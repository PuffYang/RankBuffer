from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


logger = logging.getLogger(__name__)
_FORMAT_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer>(?P<answer>.*?)</answer>", flags=re.DOTALL)


def analyze_format(text: str) -> dict[str, Any]:
    """Score the strict wrapper while independently extracting one answer block."""
    text = text or ""
    answer_matches = list(_ANSWER_RE.finditer(text))
    if len(answer_matches) != 1:
        reason = "missing_answer_block" if not answer_matches else "duplicate_answer_block"
        return {"valid": False, "answer": "", "reason": reason}
    answer = answer_matches[0].group("answer").strip()
    if not answer:
        return {"valid": False, "answer": "", "reason": "empty_answer"}

    if re.search(r"</think>\s*</think>", text):
        return {"valid": False, "answer": answer, "reason": "duplicate_think_close"}

    match = _FORMAT_RE.fullmatch(text)
    if match is None:
        return {"valid": False, "answer": answer, "reason": "invalid_outer_format"}
    if not match.group("think").strip():
        return {"valid": False, "answer": answer, "reason": "empty_think"}
    if "</think>" in match.group("think") or "</answer>" in match.group("answer"):
        return {"valid": False, "answer": answer, "reason": "duplicate_control_block"}
    return {"valid": True, "answer": answer, "reason": ""}


def extract_answer(text: str) -> str:
    return str(analyze_format(text)["answer"])


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def _json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError(f"Judge output is not an object: {cleaned[:300]}")
    return value


def _token_usage(payload: dict[str, Any], api_calls: int = 1) -> dict[str, int]:
    usage = _plain_dict(payload.get("usage"))
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "api_calls": int(api_calls),
    }


def _rubric_text(rubrics: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{idx}. [weight={item.get('weight', 1)}] {item.get('text', item.get('description', ''))}"
        for idx, item in enumerate(rubrics, start=1)
    )


def build_coarse_prompt(
    query: str,
    rubrics: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    rollout_answer: str,
    max_answer_chars: int,
) -> str:
    anchor_blocks = []
    for idx, anchor in enumerate(anchors):
        answer = str(anchor["answer"])[:max_answer_chars]
        anchor_blocks.append(f"[A{idx}]\n{answer}")
    return f"""You are a strict response-quality judge. The anchors below are already ordered from WORST to BEST.
Place the candidate response into exactly one insertion position using the query and weighted rubrics.

Position semantics for {len(anchors)} anchors:
- 0: candidate is worse than A0
- i (1 <= i < {len(anchors)}): candidate belongs between A(i-1) and Ai
- {len(anchors)}: candidate is better than A{len(anchors) - 1}

QUERY
{query}

RUBRICS
{_rubric_text(rubrics)}

ORDERED ANCHORS (worst to best)
{chr(10).join(anchor_blocks)}

CANDIDATE
{rollout_answer[:max_answer_chars]}

Return JSON only: {{"position": <integer from 0 through {len(anchors)}>}}"""


def build_fine_prompt(
    query: str,
    rubrics: list[dict[str, Any]],
    candidates: list[tuple[int, str]],
    max_answer_chars: int,
) -> str:
    blocks = [f"[R{local_id}]\n{answer[:max_answer_chars]}" for local_id, answer in candidates]
    ids = [local_id for local_id, _ in candidates]
    return f"""You are a strict response-quality judge. Rank only the candidate responses below from WORST to BEST.
Use the query and weighted rubrics. Every candidate ID must occur exactly once.

QUERY
{query}

RUBRICS
{_rubric_text(rubrics)}

CANDIDATES
{chr(10).join(blocks)}

Return JSON only. `order` must be a permutation of {ids}, from worst to best:
{{"order": {ids}}}"""


class RankBufferRewardManager(RewardManagerBase):
    """Group-level reward manager implementing persistent per-query dynamic buffers."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        cfg = _plain_dict(config.get("rank_buffer", {}))
        self.api_url = str(cfg.get("judge_api_url") or os.getenv("JUDGE_API_URL", "")).strip()
        self.api_key = str(cfg.get("judge_api_key") or os.getenv("JUDGE_API_KEY", "EMPTY"))
        self.judge_model = str(cfg.get("judge_model") or os.getenv("JUDGE_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507"))
        if not self.api_url:
            raise ValueError("RankBuffer requires rank_buffer.judge_api_url or JUDGE_API_URL")
        self.timeout = int(cfg.get("judge_timeout", 240))
        self.max_retries = int(cfg.get("judge_max_retries", 3))
        self.retry_sleep = float(cfg.get("judge_retry_sleep", 1.5))
        self.max_answer_chars = int(cfg.get("max_answer_chars", 8000))
        self.format_weight = float(cfg.get("format_reward_weight", 0.2))
        self.prune_after_epochs = max(1, int(cfg.get("prune_after_epochs", 3)))
        self.console_answer_chars = int(cfg.get("console_answer_chars", 2000))
        self.state_dir = Path(str(cfg.get("state_dir", "buffer_state"))).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir = Path(str(cfg.get("history_dir", self.state_dir / "history"))).expanduser()
        self.history_dir.mkdir(parents=True, exist_ok=True)

    async def run_single(self, data: DataProto) -> dict:
        # Kept for the abstract interface. Training config enforces batch-after-rollout.
        return (await self.run_batch(data))[0]

    async def run_batch(self, data: DataProto) -> list[dict[str, Any]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(data)):
            ground_truth = _plain_dict(data[idx].non_tensor_batch["reward_model"].get("ground_truth"))
            groups[str(ground_truth.get("query_id", idx))].append(idx)

        tasks = [self._score_group(data, indices) for indices in groups.values()]
        group_outputs = await asyncio.gather(*tasks)
        outputs: list[dict[str, Any] | None] = [None] * len(data)
        for indices, scores in zip(groups.values(), group_outputs, strict=True):
            for idx, score in zip(indices, scores, strict=True):
                outputs[idx] = score
        assert all(output is not None for output in outputs)
        return outputs  # type: ignore[return-value]

    async def _score_group(self, data: DataProto, indices: list[int]) -> list[dict[str, Any]]:
        first = data[indices[0]]
        ground_truth = _plain_dict(first.non_tensor_batch["reward_model"].get("ground_truth"))
        query_id = str(ground_truth["query_id"])
        query = str(ground_truth["query"])
        rubrics = [dict(item) for item in ground_truth["rubrics"]]
        global_step = int(data.meta_info.get("rank_buffer_global_step", 0))
        epoch = int(data.meta_info.get("rank_buffer_epoch", 0))

        state_path_existed = self._state_path(query_id).exists()
        state = self._load_state(query_id, ground_truth["anchors"])
        anchors_before = [dict(anchor) for anchor in state["anchors"]]
        anchors_at_start = [dict(anchor) for anchor in anchors_before]
        self._ensure_history_initialized(
            query_id=query_id,
            anchors=anchors_before,
            global_step=global_step,
            epoch=epoch,
            bootstrap_existing_state=state_path_existed,
        )
        responses = [self._decode_response(data[idx]) for idx in indices]
        format_infos = [analyze_format(response) for response in responses]
        answers = [str(info["answer"]) for info in format_infos]

        coarse_tasks = []
        coarse_task_ids = []
        positions = [0] * len(indices)
        judge_errors = [0] * len(indices)
        judge_prompt_tokens = [0] * len(indices)
        judge_completion_tokens = [0] * len(indices)
        judge_api_calls = [0] * len(indices)
        for local_id, info in enumerate(format_infos):
            if not info["answer"]:
                positions[local_id] = 0
                continue
            prompt = build_coarse_prompt(
                query, rubrics, anchors_before, answers[local_id], self.max_answer_chars
            )
            coarse_task_ids.append(local_id)
            coarse_tasks.append(self._judge_position(prompt, len(anchors_before)))
        if coarse_tasks:
            coarse_results = await asyncio.gather(*coarse_tasks, return_exceptions=True)
            for local_id, result in zip(coarse_task_ids, coarse_results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Coarse judge failed for query=%s rollout=%s: %s", query_id, local_id, result)
                    positions[local_id] = 0
                    judge_errors[local_id] += 1
                else:
                    position, usage = result
                    positions[local_id] = position
                    judge_prompt_tokens[local_id] += usage["prompt_tokens"]
                    judge_completion_tokens[local_id] += usage["completion_tokens"]
                    judge_api_calls[local_id] += usage["api_calls"]

        buckets: dict[int, list[int]] = defaultdict(list)
        for local_id, position in enumerate(positions):
            buckets[position].append(local_id)

        local_orders: dict[int, list[int]] = {}
        fine_tasks = []
        fine_positions = []
        for position, local_ids in buckets.items():
            unjudgeable_ids = [local_id for local_id in local_ids if not format_infos[local_id]["answer"]]
            judgeable_ids = [local_id for local_id in local_ids if format_infos[local_id]["answer"]]
            if len(judgeable_ids) < 2:
                local_orders[position] = [*unjudgeable_ids, *judgeable_ids]
                continue
            prompt = build_fine_prompt(
                query,
                rubrics,
                [(local_id, answers[local_id]) for local_id in judgeable_ids],
                self.max_answer_chars,
            )
            fine_positions.append(position)
            fine_tasks.append(self._judge_order(prompt, judgeable_ids))
        if fine_tasks:
            fine_results = await asyncio.gather(*fine_tasks, return_exceptions=True)
            for position, result in zip(fine_positions, fine_results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Fine judge failed for query=%s position=%s: %s", query_id, position, result)
                    unjudgeable_ids = [
                        local_id for local_id in buckets[position] if not format_infos[local_id]["answer"]
                    ]
                    judgeable_ids = [local_id for local_id in buckets[position] if format_infos[local_id]["answer"]]
                    local_orders[position] = [*unjudgeable_ids, *sorted(judgeable_ids)]
                    for local_id in buckets[position]:
                        judge_errors[local_id] += 1
                else:
                    unjudgeable_ids = [
                        local_id for local_id in buckets[position] if not format_infos[local_id]["answer"]
                    ]
                    order, usage = result
                    local_orders[position] = [*unjudgeable_ids, *order]
                    # A fine-ranking request belongs to the whole bucket. Attribute
                    # it once to the first rollout so summing the expanded batch is exact.
                    usage_owner = buckets[position][0]
                    judge_prompt_tokens[usage_owner] += usage["prompt_tokens"]
                    judge_completion_tokens[usage_owner] += usage["completion_tokens"]
                    judge_api_calls[usage_owner] += usage["api_calls"]

        rollout_order = [
            local_id
            for position in range(len(anchors_before) + 1)
            for local_id in local_orders.get(position, [])
        ]
        if sorted(rollout_order) != list(range(len(indices))):
            raise RuntimeError(f"Invalid assembled rollout order for query {query_id}: {rollout_order}")

        rank_by_id = {local_id: rank for rank, local_id in enumerate(rollout_order)}
        denominator = max(1, len(indices) - 1)
        rank_rewards = []
        for local_id in range(len(indices)):
            if not format_infos[local_id]["answer"]:
                # There is no uniquely extractable answer to judge. Do not create arbitrary positive
                # rewards merely from the rollout's stable list position.
                rank_rewards.append(-1.0)
            elif judge_errors[local_id] > 0:
                rank_rewards.append(0.0)
            else:
                rank_rewards.append(-1.0 + 2.0 * rank_by_id[local_id] / denominator)
        format_rewards = [self.format_weight if info["valid"] else 0.0 for info in format_infos]

        mixed_order = []
        for position in range(len(anchors_before) + 1):
            mixed_order.extend(f"R{local_id}" for local_id in local_orders.get(position, []))
            if position < len(anchors_before):
                mixed_order.append(f"A{position}")

        update = self._update_buffer(
            state=state,
            anchors_before=anchors_before,
            buckets=buckets,
            local_orders=local_orders,
            answers=answers,
            update_eligible=[bool(info["answer"]) and judge_errors[idx] == 0 for idx, info in enumerate(format_infos)],
            global_step=global_step,
            epoch=epoch,
        )
        self._save_state(query_id, state)
        self._append_history_event(
            query_id,
            {
                "schema_version": 1,
                "event": "update",
                "query_id": query_id,
                "global_step": global_step,
                "epoch": epoch,
                "buffer_size_before": len(anchors_before),
                "buffer_size_after": len(state["anchors"]),
                "anchor_order_before": [anchor["anchor_id"] for anchor in anchors_at_start],
                "anchor_order_after": [anchor["anchor_id"] for anchor in state["anchors"]],
                "inactive_epochs_before": {
                    anchor["anchor_id"]: int(anchor.get("inactive_epochs", 0))
                    for anchor in anchors_at_start
                },
                "inactive_epochs_after": {
                    anchor["anchor_id"]: int(anchor.get("inactive_epochs", 0))
                    for anchor in state["anchors"]
                },
                "bucket_hits": {
                    str(position): len(local_ids) for position, local_ids in sorted(buckets.items())
                },
                "eligible_bucket_hits": {
                    str(position): sum(
                        bool(format_infos[local_id]["answer"]) and judge_errors[local_id] == 0
                        for local_id in local_ids
                    )
                    for position, local_ids in sorted(buckets.items())
                },
                "rollout_insertions": [
                    {
                        "rollout_id": f"R{local_id}",
                        "position": positions[local_id],
                        "local_rank": rank_by_id[local_id],
                        "format_valid": bool(format_infos[local_id]["valid"]),
                        "judge_error": bool(judge_errors[local_id]),
                    }
                    for local_id in range(len(indices))
                ],
                "added": self._history_anchors(update["added"], state["anchors"]),
                "pruned": self._history_anchors(update["pruned"], anchors_before),
                "state_markers": {
                    "last_observed_epoch": state.get("last_observed_epoch", -1),
                    "last_left_expansion_epoch": state.get("last_left_expansion_epoch", -1),
                    "last_right_expansion_epoch": state.get("last_right_expansion_epoch", -1),
                },
                "skipped_duplicate_step": bool(update["skipped_duplicate_step"]),
            },
        )

        update_json = json.dumps(update, ensure_ascii=False, separators=(",", ":"))
        mixed_json = json.dumps(mixed_order, ensure_ascii=False, separators=(",", ":"))
        print(
            f"[RankBuffer] step={global_step} epoch={epoch} query_id={query_id} "
            f"buffer={len(anchors_before)}->{len(state['anchors'])} order={mixed_json} update={update_json}"
        )
        for local_id in range(len(indices)):
            preview_source = answers[local_id] if format_infos[local_id]["answer"] else responses[local_id]
            preview = preview_source[: self.console_answer_chars].replace("\n", "\\n")
            total = rank_rewards[local_id] + format_rewards[local_id]
            print(
                f"[RankBuffer.rollout] id=R{local_id} pos={positions[local_id]} rank={rank_by_id[local_id]} "
                f"rank_reward={rank_rewards[local_id]:.4f} format={format_rewards[local_id]:.1f} "
                f"total={total:.4f} format_reason={format_infos[local_id]['reason'] or 'ok'} "
                f"output={preview}"
            )

        results = []
        for local_id in range(len(indices)):
            total = rank_rewards[local_id] + format_rewards[local_id]
            extra = {
                "score": float(total),
                "rank_reward": float(rank_rewards[local_id]),
                "format_reward": float(format_rewards[local_id]),
                "format_valid": int(format_infos[local_id]["valid"]),
                "rollout_rank": int(rank_by_id[local_id]),
                "insert_position": int(positions[local_id]),
                "insert_position_norm": float(positions[local_id] / max(1, len(anchors_before))),
                "judge_error": int(judge_errors[local_id] > 0),
                "judge_prompt_tokens": int(judge_prompt_tokens[local_id]),
                "judge_completion_tokens": int(judge_completion_tokens[local_id]),
                "judge_api_calls": int(judge_api_calls[local_id]),
                "buffer_size": int(len(state["anchors"])),
                "buffer_added": int(len(update["added"])),
                "buffer_pruned": int(len(update["pruned"])),
                "query_id": query_id,
                "rollout_id": f"R{local_id}",
                "rollout_answer": answers[local_id],
                "raw_response_preview": responses[local_id][: self.console_answer_chars],
                "format_reason": str(format_infos[local_id]["reason"]),
                "mixed_order": mixed_json,
                "buffer_update": update_json,
            }
            results.append({"reward_score": float(total), "reward_extra_info": extra})
        return results

    def _decode_response(self, item) -> str:
        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_length = int(item.batch["attention_mask"][-response_length:].sum().item())
        return self.tokenizer.decode(response_ids[:valid_length], skip_special_tokens=True)

    async def _judge_position(self, prompt: str, anchor_count: int) -> tuple[int, dict[str, int]]:
        payload, usage = await asyncio.to_thread(self._judge_request, prompt, 64)
        position = int(_json_object(payload)["position"])
        if not 0 <= position <= anchor_count:
            raise ValueError(f"Position {position} outside [0, {anchor_count}]")
        return position, usage

    async def _judge_order(
        self, prompt: str, expected_ids: list[int]
    ) -> tuple[list[int], dict[str, int]]:
        payload, usage = await asyncio.to_thread(
            self._judge_request, prompt, max(64, len(expected_ids) * 12)
        )
        order = [int(value) for value in _json_object(payload)["order"]]
        if len(order) != len(expected_ids) or set(order) != set(expected_ids):
            raise ValueError(f"Invalid local permutation {order}; expected {expected_ids}")
        return order, usage

    def _judge_request(self, prompt: str, max_tokens: int) -> tuple[str, dict[str, int]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key and self.api_key != "EMPTY":
                    headers["Authorization"] = f"Bearer {self.api_key}"
                body = {
                    "model": self.judge_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Judge response quality precisely. Return only the requested JSON object.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max_tokens,
                    "seed": 0,
                }
                with requests.Session() as session:
                    session.trust_env = False
                    response = session.post(self.api_url, headers=headers, json=body, timeout=self.timeout)
                response.raise_for_status()
                response_payload = response.json()
                message = response_payload["choices"][0]["message"]
                content = str(message.get("content") or "").strip()
                if not content:
                    raise ValueError("Judge returned empty content")
                if not response_payload.get("usage"):
                    logger.warning(
                        "Judge response has no OpenAI-compatible usage field; "
                        "its input/output tokens cannot be counted."
                    )
                return content, _token_usage(response_payload)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * attempt)
        raise RuntimeError(f"Judge failed after {self.max_retries} attempts: {last_error}")

    def _state_path(self, query_id: str) -> Path:
        if re.fullmatch(r"[0-9]+", query_id) is None:
            raise ValueError(f"RankBuffer query_id must be numeric, got {query_id!r}")
        return self.state_dir / f"{query_id}.json"

    def _history_path(self, query_id: str) -> Path:
        if re.fullmatch(r"[0-9]+", query_id) is None:
            raise ValueError(f"RankBuffer query_id must be numeric, got {query_id!r}")
        return self.history_dir / f"{query_id}.jsonl"

    @staticmethod
    def _ensure_anchor_ids(anchors: list[dict[str, Any]]) -> None:
        used_ids = set()
        for index, anchor in enumerate(anchors):
            anchor_id = str(anchor.get("anchor_id", "")).strip()
            if not anchor_id:
                if anchor.get("source") == "nectar" and anchor.get("nectar_rank") is not None:
                    anchor_id = f"N{anchor['nectar_rank']}"
                else:
                    fingerprint = hashlib.sha1(  # noqa: S324 - stable identifier, not security
                        str(anchor.get("answer", "")).encode("utf-8")
                    ).hexdigest()[:12]
                    anchor_id = (
                        f"D_E{anchor.get('added_epoch', 'x')}_S{anchor.get('added_step', 'x')}_"
                        f"{fingerprint}"
                    )
            base_id = anchor_id
            suffix = 1
            while anchor_id in used_ids:
                suffix += 1
                anchor_id = f"{base_id}_{suffix}"
            anchor["anchor_id"] = anchor_id
            used_ids.add(anchor_id)

    def _append_history_event(self, query_id: str, event: dict[str, Any]) -> None:
        path = self._history_path(query_id)
        payload = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            written = 0
            while written < len(payload):
                written += os.write(descriptor, payload[written:])
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _ensure_history_initialized(
        self,
        query_id: str,
        anchors: list[dict[str, Any]],
        global_step: int,
        epoch: int,
        bootstrap_existing_state: bool,
    ) -> None:
        if self._history_path(query_id).exists():
            return
        self._append_history_event(
            query_id,
            {
                "schema_version": 1,
                "event": "initial_snapshot",
                "query_id": query_id,
                "global_step": global_step,
                "epoch": epoch,
                "bootstrap_existing_state": bool(bootstrap_existing_state),
                "buffer_size": len(anchors),
                "anchor_order": [anchor["anchor_id"] for anchor in anchors],
                "anchors": anchors,
            },
        )

    @staticmethod
    def _history_anchors(
        updates: list[dict[str, Any]], anchors: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        anchors_by_id = {anchor["anchor_id"]: anchor for anchor in anchors}
        history_items = []
        for item in updates:
            anchor_id = item.get("anchor_id")
            anchor = anchors_by_id.get(anchor_id)
            if anchor is None:
                continue
            history_items.append({**item, "anchor": dict(anchor)})
        return history_items

    def _load_state(self, query_id: str, initial_anchors: list[dict[str, Any]]) -> dict[str, Any]:
        path = self._state_path(query_id)
        if path.exists():
            with open(path, encoding="utf-8") as file:
                state = json.load(file)
            if state.get("query_id") != query_id or len(state.get("anchors", [])) < 2:
                raise ValueError(f"Invalid RankBuffer state: {path}")
            self._ensure_anchor_ids(state["anchors"])
            state["version"] = 2
            return state
        anchors = [dict(anchor) for anchor in initial_anchors]
        self._ensure_anchor_ids(anchors)
        return {
            "version": 2,
            "query_id": query_id,
            "anchors": anchors,
            "last_updated_step": -1,
            "last_updated_epoch": -1,
            "last_observed_epoch": -1,
            "last_left_expansion_epoch": -1,
            "last_right_expansion_epoch": -1,
        }

    def _save_state(self, query_id: str, state: dict[str, Any]) -> None:
        path = self._state_path(query_id)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)

    def _update_buffer(
        self,
        state: dict[str, Any],
        anchors_before: list[dict[str, Any]],
        buckets: dict[int, list[int]],
        local_orders: dict[int, list[int]],
        answers: list[str],
        update_eligible: list[bool],
        global_step: int,
        epoch: int,
    ) -> dict[str, Any]:
        self._ensure_anchor_ids(anchors_before)
        if int(state.get("last_updated_step", -1)) == global_step:
            return {"added": [], "pruned": [], "skipped_duplicate_step": True}

        anchor_count = len(anchors_before)
        prune_indices = []
        if int(state.get("last_observed_epoch", -1)) != epoch:
            for idx in range(1, anchor_count - 1):
                hit_left = any(update_eligible[local_id] for local_id in buckets.get(idx, []))
                hit_right = any(update_eligible[local_id] for local_id in buckets.get(idx + 1, []))
                inactive = (
                    0
                    if hit_left or hit_right
                    else int(anchors_before[idx].get("inactive_epochs", 0)) + 1
                )
                anchors_before[idx]["inactive_epochs"] = inactive
                if inactive >= self.prune_after_epochs:
                    prune_indices.append(idx)
            state["last_observed_epoch"] = epoch

        additions: list[tuple[float, dict[str, Any], str]] = []
        left_order = [local_id for local_id in local_orders.get(0, []) if update_eligible[local_id]]
        if left_order and int(state.get("last_left_expansion_epoch", -1)) != epoch:
            local_id = left_order[0]
            additions.append(
                (-0.5, self._rollout_anchor(answers[local_id], epoch, global_step, "left_boundary"), "left")
            )
            state["last_left_expansion_epoch"] = epoch
        right_order = [
            local_id for local_id in local_orders.get(anchor_count, []) if update_eligible[local_id]
        ]
        if right_order and int(state.get("last_right_expansion_epoch", -1)) != epoch:
            local_id = right_order[-1]
            additions.append(
                (
                    anchor_count - 0.5,
                    self._rollout_anchor(answers[local_id], epoch, global_step, "right_boundary"),
                    "right",
                )
            )
            state["last_right_expansion_epoch"] = epoch
        for position in range(1, anchor_count):
            order = [local_id for local_id in local_orders.get(position, []) if update_eligible[local_id]]
            if len(order) < 3:
                continue
            local_id = order[(len(order) - 1) // 2]
            additions.append(
                (
                    position - 0.5,
                    self._rollout_anchor(answers[local_id], epoch, global_step, "local_refinement"),
                    f"interval_{position}",
                )
            )

        merged: list[tuple[float, dict[str, Any]]] = []
        for idx, anchor in enumerate(anchors_before):
            if idx not in prune_indices:
                merged.append((float(idx), anchor))
        merged.extend((coordinate, anchor) for coordinate, anchor, _ in additions)
        merged.sort(key=lambda item: item[0])
        state["anchors"] = [anchor for _, anchor in merged]
        state["last_updated_step"] = global_step
        state["last_updated_epoch"] = epoch

        return {
            "added": [
                {
                    "anchor_id": anchor["anchor_id"],
                    "where": where,
                    "source": anchor["source"],
                    "answer_preview": anchor["answer"][:120],
                }
                for _, anchor, where in additions
            ],
            "pruned": [
                {
                    "anchor_id": anchors_before[idx]["anchor_id"],
                    "anchor_index": idx,
                    "answer_preview": anchors_before[idx]["answer"][:120],
                }
                for idx in prune_indices
            ],
            "skipped_duplicate_step": False,
        }

    @staticmethod
    def _rollout_anchor(answer: str, epoch: int, step: int, source: str) -> dict[str, Any]:
        return {
            "anchor_id": f"D_{uuid.uuid4().hex[:12]}",
            "answer": answer,
            "source": source,
            "added_epoch": epoch,
            "added_step": step,
            "inactive_epochs": 0,
        }
