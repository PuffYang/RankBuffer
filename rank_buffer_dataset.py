from __future__ import annotations

import copy
import os
import re
from typing import Any

import datasets

from verl.utils.dataset.rl_dataset import RLHFDataset


_NECTAR_TURN_RE = re.compile(r"(?:^|\n\n)(Human|Assistant):\s*", flags=re.IGNORECASE)
_EMPTY_ANSWER_SENTINEL = "[EMPTY RESPONSE]"


def _nectar_query_to_messages(query: str) -> list[dict[str, str]]:
    """Convert Nectar's ``Human:/Assistant:`` transcript into chat messages."""
    query = str(query or "").strip()
    matches = list(_NECTAR_TURN_RE.finditer(query))
    if not matches:
        return [{"role": "user", "content": query}]

    messages: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(query)
        content = query[content_start:content_end].strip()
        role = "user" if match.group(1).lower() == "human" else "assistant"
        if content:
            messages.append({"role": role, "content": content})

    # Nectar prompts normally end with an empty Assistant turn. Generation needs the
    # conversation through the last user message, not an empty assistant message.
    while messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages.pop()
    if not messages or messages[-1]["role"] != "user":
        return [{"role": "user", "content": query}]
    return messages


def _normalize_rubrics(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rubrics = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", item.get("description", ""))).strip()
        if not text:
            continue
        try:
            weight = max(1, min(5, int(item.get("weight", item.get("importance", 1)))))
        except (TypeError, ValueError):
            weight = 1
        rubrics.append({"text": text, "weight": weight})
    return rubrics


def _normalize_anchor_answers(value: Any, row_index: int) -> list[dict[str, Any]]:
    """Validate Nectar ranks and return all seven anchors from worst to best.

    Two records in the supplied 1k file contain a genuinely empty model response.
    Keeping a visible sentinel is preferable to dropping the query or presenting an
    unlabeled blank block to the judge; ``is_empty`` preserves that provenance.
    """
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError(f"Invalid Nectar sample at row {row_index}: expected exactly 7 answers")

    normalized: list[dict[str, Any]] = []
    ranks: list[int] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid Nectar answer at row {row_index}: expected an object")
        try:
            rank = int(item["rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Nectar answer rank at row {row_index}: {item.get('rank')!r}") from exc
        ranks.append(rank)
        raw_answer = str(item.get("answer") or "").strip()
        normalized.append(
            {
                "answer": raw_answer or _EMPTY_ANSWER_SENTINEL,
                "source": "nectar",
                "nectar_rank": rank,
                "model": item.get("model"),
                "is_empty": not raw_answer,
                "inactive_epochs": 0,
            }
        )

    if sorted(ranks) != list(range(1, 8)):
        raise ValueError(
            f"Invalid Nectar ranks at row {row_index}: expected each integer from 1 to 7, got {ranks}"
        )
    # Nectar rank=1 is best; RankBuffer is represented from worst to best.
    return sorted(normalized, key=lambda answer: answer["nectar_rank"], reverse=True)


def _select_initial_anchors(anchors: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Uniformly subsample ordered anchors while always preserving both boundaries."""
    if not 2 <= count <= len(anchors):
        raise ValueError(f"initial_anchor_count must be in [2, {len(anchors)}], got {count}")
    if count == len(anchors):
        return [dict(anchor) for anchor in anchors]
    last_index = len(anchors) - 1
    indices = [round(slot * last_index / (count - 1)) for slot in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError(f"Could not uniformly select {count} anchors from {len(anchors)}")
    return [dict(anchors[index]) for index in indices]


class RankBufferPromptDataset(RLHFDataset):
    """Load rubric-augmented Nectar JSONL and expose a configured anchor subset."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            raise ValueError("RankBufferPromptDataset requires `config`.")

        prompt_file = os.path.expanduser(str(config.get("system_prompt_file", ""))).strip()
        if not prompt_file or not os.path.isfile(prompt_file):
            raise FileNotFoundError(f"RankBuffer system prompt does not exist: {prompt_file}")
        with open(prompt_file, encoding="utf-8") as file:
            self.system_prompt_text = file.read().strip()
        if not self.system_prompt_text:
            raise ValueError(f"RankBuffer system prompt is empty: {prompt_file}")
        self.initial_anchor_count = int(config.get("initial_anchor_count", 7))
        if not 2 <= self.initial_anchor_count <= 7:
            raise ValueError(
                f"RankBuffer initial_anchor_count must be between 2 and 7, got {self.initial_anchor_count}"
            )
        super().__init__(*args, **kwargs)

    def _read_files_and_tokenize(self):
        frames = []
        for data_file in self.data_files:
            if not str(data_file).endswith((".json", ".jsonl")):
                raise ValueError("RankBuffer training data must be the rubric-augmented Nectar JSONL.")
            frames.append(datasets.load_dataset("json", data_files=data_file)["train"])
        raw = datasets.concatenate_datasets(frames)

        required = {"query", "rubrics", "answers"}
        missing = required - set(raw.column_names)
        if missing:
            raise ValueError(f"Nectar JSONL is missing required fields: {sorted(missing)}")

        def prepare(example: dict, index: int) -> dict:
            query = str(example["query"] or "").strip()
            rubrics = _normalize_rubrics(example["rubrics"])
            if not query or not rubrics:
                raise ValueError(f"Invalid Nectar sample at row {index}: expected query, rubrics, and 7 answers")
            all_anchors = _normalize_anchor_answers(example["answers"], index)
            anchors = _select_initial_anchors(all_anchors, self.initial_anchor_count)
            query_id = str(example.get("sample_index", index))
            messages = [
                {"role": "system", "content": self.system_prompt_text},
                *_nectar_query_to_messages(query),
            ]
            ground_truth = {
                "query_id": query_id,
                "query": query,
                "rubrics": rubrics,
                "anchors": anchors,
            }
            return {
                "prompt": messages,
                "data_source": "nectar_rank_buffer",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"index": index, "query_id": query_id},
            }

        self.dataframe = raw.map(
            prepare,
            with_indices=True,
            remove_columns=raw.column_names,
            desc="Preparing Nectar RankBuffer samples",
            load_from_cache_file=False,
        )
        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            self.dataframe = self.dataframe.select(range(self.max_samples))
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        selected_ranks = [
            anchor["nectar_rank"]
            for anchor in _select_initial_anchors(
                [{"nectar_rank": rank} for rank in range(7, 0, -1)], self.initial_anchor_count
            )
        ]
        print(
            f"RankBuffer dataset len: {len(self.dataframe)}, "
            f"initial_anchor_count={self.initial_anchor_count}, nectar_ranks={selected_ranks}"
        )

    def _build_messages(self, example: dict, key: str):
        return copy.deepcopy(example[key])
