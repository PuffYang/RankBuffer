from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)


def extract_answer(text: str) -> str:
    matches = list(_ANSWER_RE.finditer(text or ""))
    return matches[0].group(1).strip() if len(matches) == 1 and matches[0].group(1).strip() else ""


def _judge_prompt(instruction: str, output_a: str, output_b: str) -> str:
    return f"""Compare two assistant responses to the same instruction. Judge correctness, relevance,
instruction following, completeness, clarity, safety, and appropriate concision. Do not favor a response merely
because it is longer. If quality is genuinely indistinguishable, choose tie.

INSTRUCTION
{instruction}

RESPONSE A
{output_a}

RESPONSE B
{output_b}

Return JSON only: {{"winner":"A"}} or {{"winner":"B"}} or {{"winner":"tie"}}."""


def _parse_winner(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        payload = json.loads(cleaned[start : end + 1])
    winner = str(payload["winner"]).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError(f"Invalid AlpacaEval 2 winner: {winner}")
    return winner


def _request_judge(prompt: str, config: dict[str, Any]) -> str:
    api_url = str(config.get("api_url") or os.getenv("JUDGE_API_URL", ""))
    api_key = str(config.get("api_key") or os.getenv("JUDGE_API_KEY", "EMPTY"))
    model = str(config.get("model") or os.getenv("JUDGE_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507"))
    retries = int(config.get("max_retries", 3))
    retry_sleep = float(config.get("retry_sleep", 1.5))
    timeout = int(config.get("timeout", 240))
    if not api_url:
        raise ValueError("AlpacaEval 2 judge API URL is missing")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            headers = {"Content-Type": "application/json"}
            if api_key and api_key != "EMPTY":
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an impartial AlpacaEval-style pairwise judge. Return JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 32,
                "seed": 0,
            }
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"].get("content") or "")
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"AlpacaEval 2 judge failed: {last_error}")


def _evaluate_one(index: int, item: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    instruction = item["instruction"]
    candidate = item["candidate_output"]
    reference = item["reference_output"]
    # Deterministic randomization removes a fixed A/B positional advantage while
    # keeping repeated evaluations comparable across checkpoints.
    seed = int(hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:8], 16)  # noqa: S324
    candidate_is_a = bool(random.Random(seed).randrange(2) == 0)
    output_a, output_b = (candidate, reference) if candidate_is_a else (reference, candidate)
    try:
        raw = _request_judge(_judge_prompt(instruction, output_a, output_b), config)
        winner = _parse_winner(raw)
        if winner == "TIE":
            preference = 0.5
        elif (winner == "A") == candidate_is_a:
            preference = 1.0
        else:
            preference = 0.0
        error = ""
    except Exception as exc:
        raw = ""
        preference = 0.0
        error = str(exc)
    return {
        "index": index,
        **item,
        "candidate_is_a": candidate_is_a,
        "preference": preference,
        "judge_output": raw,
        "judge_error": error,
    }


def evaluate_alpacaeval2(
    records: list[dict[str, str]],
    output_dir: str | None,
    max_workers: int = 16,
    judge_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate against AlpacaEval 2's official GPT-4-Turbo reference outputs.

    The judge is intentionally configurable and is Qwen3-30B-A3B in this project,
    so this is a development win rate rather than the official GPT-4 weighted/LC score.
    """
    config = dict(judge_config or {})
    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [pool.submit(_evaluate_one, idx, item, config) for idx, item in enumerate(records)]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["index"])

    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model_outputs.json", "w", encoding="utf-8") as file:
            json.dump(
                [
                    {
                        "instruction": item["instruction"],
                        "output": item["candidate_output"],
                        "generator": "RankBuffer",
                    }
                    for item in results
                ],
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")
        with open(path / "qwen_judgments.jsonl", "w", encoding="utf-8") as file:
            for item in results:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

    valid = [item for item in results if not item["judge_error"]]
    denominator = max(1, len(results))
    return {
        "alpaca_eval2/qwen_win_rate": 100.0 * sum(item["preference"] for item in results) / denominator,
        "alpaca_eval2/qwen_tie_rate": 100.0 * sum(item["preference"] == 0.5 for item in results) / denominator,
        "alpaca_eval2/judge_error_rate": 100.0 * (len(results) - len(valid)) / denominator,
        "alpaca_eval2/format_valid_rate": 100.0
        * sum(bool(item.get("candidate_format_valid", item["candidate_output"])) for item in results)
        / denominator,
        "alpaca_eval2/candidate_mean_chars": sum(len(item["candidate_output"]) for item in results) / denominator,
        "alpaca_eval2/reference_mean_chars": sum(len(item["reference_output"]) for item in results) / denominator,
    }
