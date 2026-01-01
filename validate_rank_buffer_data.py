from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def validate_file(path: Path) -> dict[str, int]:
    """Fail fast on schema/rank defects before a distributed training job starts."""
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    rows = 0
    empty_answers = 0
    duplicate_answer_samples = 0

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                item: dict[str, Any] = json.loads(line)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            missing = {"sample_index", "query", "rubrics", "answers"} - set(item)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            query_id = str(item["sample_index"])
            query = str(item["query"] or "").strip()
            if not query or query_id in seen_ids or query in seen_queries:
                raise ValueError(f"{path}:{line_number}: empty or duplicate query/query ID {query_id!r}")
            seen_ids.add(query_id)
            seen_queries.add(query)

            rubrics = item["rubrics"]
            if not isinstance(rubrics, list) or not rubrics:
                raise ValueError(f"{path}:{line_number}: rubrics must be a non-empty list")
            for rubric in rubrics:
                if not isinstance(rubric, dict) or not str(rubric.get("text") or "").strip():
                    raise ValueError(f"{path}:{line_number}: rubric text is missing")
                weight = rubric.get("weight")
                if not isinstance(weight, int) or not 1 <= weight <= 5:
                    raise ValueError(f"{path}:{line_number}: rubric weight must be an integer in [1, 5]")

            answers = item["answers"]
            if not isinstance(answers, list) or len(answers) != 7:
                raise ValueError(f"{path}:{line_number}: expected exactly 7 answers")
            try:
                ranks = [int(answer["rank"]) for answer in answers]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: every answer needs an integer rank") from exc
            if sorted(ranks) != list(range(1, 8)):
                raise ValueError(f"{path}:{line_number}: ranks must be exactly 1..7, got {ranks}")
            texts = [str(answer.get("answer") or "").strip() for answer in answers]
            empty_answers += sum(not text for text in texts)
            duplicate_answer_samples += int(len(Counter(texts)) < 7)
            rows += 1

    if rows == 0:
        raise ValueError(f"{path}: dataset is empty")
    return {
        "rows": rows,
        "empty_answers_normalized": empty_answers,
        "samples_with_duplicate_answer_text": duplicate_answer_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rubric-augmented Nectar data for RankBuffer")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if not args.path.is_file():
        raise FileNotFoundError(args.path)
    result = validate_file(args.path)
    print("RankBuffer data validation: " + ", ".join(f"{key}={value}" for key, value in result.items()))


if __name__ == "__main__":
    main()
