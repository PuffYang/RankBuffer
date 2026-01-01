from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    verl_module = types.ModuleType("verl")
    verl_module.DataProto = object
    sys.modules["verl"] = verl_module
    for module_name in [
        "verl.experimental",
        "verl.experimental.reward_loop",
        "verl.experimental.reward_loop.reward_manager",
        "verl.utils",
        "verl.utils.dataset",
    ]:
        sys.modules[module_name] = types.ModuleType(module_name)
    reward_base = types.ModuleType("verl.experimental.reward_loop.reward_manager.base")
    reward_base.RewardManagerBase = object
    sys.modules[reward_base.__name__] = reward_base
    dataset_module = types.ModuleType("verl.utils.dataset.rl_dataset")
    dataset_module.RLHFDataset = object
    sys.modules[dataset_module.__name__] = dataset_module

from rank_buffer_dataset import _nectar_query_to_messages, _normalize_anchor_answers, _select_initial_anchors
from rank_buffer_reward import (
    RankBufferRewardManager,
    _token_usage,
    analyze_format,
    build_coarse_prompt,
    build_fine_prompt,
)
import alpaca_eval2
from validate_rank_buffer_data import validate_file
from migrate_buffer_state import migrate_buffer_state


def _anchors(count: int = 7):
    return [
        {"answer": f"anchor-{idx}", "source": "nectar", "inactive_epochs": 0}
        for idx in range(count)
    ]


def _manager():
    manager = RankBufferRewardManager.__new__(RankBufferRewardManager)
    manager.prune_after_epochs = 3
    return manager


def test_qwen_format_requires_exact_generated_blocks():
    valid = analyze_format("<think>reason</think>\n<answer>final</answer>")
    assert valid == {"valid": True, "answer": "final", "reason": ""}
    answer_only = analyze_format("<answer>final</answer>")
    assert not answer_only["valid"] and answer_only["answer"] == "final"
    assert not analyze_format("<think></think><answer>final</answer>")["valid"]
    assert analyze_format("<think>Use the answer tag.</think><answer><b>valid HTML</b></answer>")["valid"]
    assert not analyze_format("<think>x</think><answer>one</answer><answer>two</answer>")["valid"]
    duplicate_close = analyze_format("<think>reason</think></think><answer>final</answer>")
    assert duplicate_close == {"valid": False, "answer": "final", "reason": "duplicate_think_close"}


def test_start_uses_qwen3_native_thinking_template():
    start_source = (Path(__file__).parents[1] / "start.sh").read_text()
    assert "apply_chat_template_kwargs.enable_thinking=True" in start_source
    assert "custom_chat_template" not in start_source


def test_judge_url_has_an_anonymous_local_default_and_supports_override():
    start_source = (Path(__file__).parents[1] / "start.sh").read_text()
    assert 'export JUDGE_API_URL="${JUDGE_API_URL:-http://localhost:8000/v1/chat/completions}"' in start_source
    assert "JUDGE_HOST" not in start_source


def test_start_enables_durable_token_usage_log():
    start_source = (Path(__file__).parents[1] / "start.sh").read_text()
    assert 'TOKEN_USAGE_DIR="${SCRIPT_DIR}/token_usage"' in start_source
    assert '++trainer.token_usage_log="${TOKEN_USAGE_LOG}"' in start_source


def test_openai_compatible_judge_input_and_output_tokens_are_counted():
    assert _token_usage(
        {"usage": {"prompt_tokens": 101, "completion_tokens": 7, "total_tokens": 108}}
    ) == {
        "prompt_tokens": 101,
        "completion_tokens": 7,
        "api_calls": 1,
    }
    assert _token_usage({}) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "api_calls": 1,
    }


def test_training_judge_call_metrics_count_only_successfully_parsed_results():
    trainer_source = (Path(__file__).parents[1] / "verl/verl/trainer/ppo/ray_trainer.py").read_text()
    reward_source = (Path(__file__).parents[1] / "rank_buffer_reward.py").read_text()
    assert '"train_judge_calls"' in trainer_source
    assert '"judge_api_calls"' in reward_source
    assert 'judge_api_calls[local_id] += usage["api_calls"]' in reward_source
    assert 'getattr(result, "usage"' not in reward_source
    assert 'cumulative_usage["api_calls"]' not in reward_source


def test_buffer_state_files_use_numeric_query_ids_and_migrate_legacy_names():
    with tempfile.TemporaryDirectory() as directory:
        state_dir = Path(directory)
        query_id = "158031"
        legacy = state_dir / f"{hashlib.sha1(query_id.encode()).hexdigest()}.json"
        legacy.write_text(json.dumps({"query_id": query_id, "anchors": [{}, {}]}), encoding="utf-8")
        assert migrate_buffer_state(state_dir) == 1
        assert not legacy.exists()
        assert (state_dir / "158031.json").is_file()

        manager = RankBufferRewardManager.__new__(RankBufferRewardManager)
        manager.state_dir = state_dir
        manager.history_dir = state_dir / "history"
        manager.history_dir.mkdir()
        assert manager._state_path(query_id) == state_dir / "158031.json"


def test_buffer_history_is_append_only_and_keeps_query_evolution():
    with tempfile.TemporaryDirectory() as directory:
        manager = RankBufferRewardManager.__new__(RankBufferRewardManager)
        manager.state_dir = Path(directory)
        manager.history_dir = manager.state_dir / "history"
        manager.history_dir.mkdir()
        anchors = _anchors(3)
        manager._ensure_anchor_ids(anchors)

        manager._ensure_history_initialized(
            query_id="53360",
            anchors=anchors,
            global_step=1,
            epoch=0,
            bootstrap_existing_state=False,
        )
        manager._append_history_event(
            "53360",
            {
                "schema_version": 1,
                "event": "update",
                "query_id": "53360",
                "global_step": 1,
                "epoch": 0,
                "anchor_order_before": [anchor["anchor_id"] for anchor in anchors],
                "anchor_order_after": [anchor["anchor_id"] for anchor in anchors],
            },
        )
        manager._append_history_event(
            "53360",
            {
                "schema_version": 1,
                "event": "update",
                "query_id": "53360",
                "global_step": 1001,
                "epoch": 1,
                "anchor_order_before": [anchor["anchor_id"] for anchor in anchors],
                "anchor_order_after": [anchor["anchor_id"] for anchor in anchors],
            },
        )

        history_path = manager.history_dir / "53360.jsonl"
        events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
        assert [event["event"] for event in events] == ["initial_snapshot", "update", "update"]
        assert [event["epoch"] for event in events] == [0, 0, 1]
        assert events[0]["anchors"][0]["anchor_id"] == anchors[0]["anchor_id"]


def test_start_places_numeric_history_files_under_buffer_state():
    start_source = (Path(__file__).parents[1] / "start.sh").read_text()
    assert 'BUFFER_HISTORY_DIR="${BUFFER_HISTORY_DIR:-${BUFFER_STATE_DIR}/history}"' in start_source
    assert '++rank_buffer.history_dir="${BUFFER_HISTORY_DIR}"' in start_source


def test_rankbuffer_metadata_names_do_not_collide_with_vllm_global_steps():
    trainer_source = (Path(__file__).parents[1] / "verl/verl/trainer/ppo/ray_trainer.py").read_text()
    reward_source = (Path(__file__).parents[1] / "rank_buffer_reward.py").read_text()
    assert 'batch.meta_info["rank_buffer_global_step"]' in trainer_source
    assert 'data.meta_info.get("rank_buffer_global_step"' in reward_source


def test_nectar_transcript_becomes_chat_without_empty_assistant_turn():
    query = "\n\nHuman: first\n\nAssistant: previous\n\nHuman: follow-up\n\nAssistant: "
    assert _nectar_query_to_messages(query) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "previous"},
        {"role": "user", "content": "follow-up"},
    ]


def test_real_nectar_anchor_order_and_empty_response_normalization():
    answers = [
        {"rank": rank, "model": f"m{rank}", "answer": "" if rank == 6 else f"answer-{rank}"}
        for rank in range(1, 8)
    ]
    anchors = _normalize_anchor_answers(answers, row_index=0)
    assert [anchor["nectar_rank"] for anchor in anchors] == [7, 6, 5, 4, 3, 2, 1]
    assert anchors[1]["answer"] == "[EMPTY RESPONSE]"
    assert anchors[1]["is_empty"] is True


def test_k4_ablation_selects_evenly_spaced_anchors():
    anchors = [{"nectar_rank": rank} for rank in range(7, 0, -1)]
    selected = _select_initial_anchors(anchors, 4)
    assert [anchor["nectar_rank"] for anchor in selected] == [7, 5, 3, 1]
    start_source = (Path(__file__).parents[1] / "start.sh").read_text()
    assert "INITIAL_BUFFER_SIZE=4" in start_source
    assert '++data.initial_anchor_count="${INITIAL_BUFFER_SIZE}"' in start_source
    assert 'trainer.experiment_name="RankBuffer_K4_${TIMESTAMP}"' in start_source


def test_optional_local_training_file_passes_full_preflight():
    data_path = Path(__file__).parents[1] / "data/train.jsonl"
    if data_path.is_file():
        result = validate_file(data_path)
        assert result == {
            "rows": 1000,
            "empty_answers_normalized": 2,
            "samples_with_duplicate_answer_text": 81,
        }


def test_judge_prompts_use_compact_integer_protocols():
    anchors = _anchors(3)
    coarse = build_coarse_prompt("q", [{"text": "correct", "weight": 5}], anchors, "candidate", 100)
    fine = build_fine_prompt("q", [{"text": "correct", "weight": 5}], [(2, "x"), (5, "y")], 100)
    assert '"position"' in coarse and "0 through 3" in coarse
    assert '"order"' in fine and "[2, 5]" in fine


def test_buffer_expands_boundaries_and_refines_with_valid_rollouts_only():
    manager = _manager()
    anchors = _anchors(3)
    state = {
        "anchors": copy.deepcopy(anchors),
        "last_updated_step": -1,
        "last_observed_epoch": -1,
        "last_left_expansion_epoch": -1,
        "last_right_expansion_epoch": -1,
    }
    answers = ["bad", "invalid", "middle-low", "middle-mid", "middle-high", "great"]
    update = manager._update_buffer(
        state=state,
        anchors_before=copy.deepcopy(anchors),
        buckets={0: [0, 1], 1: [2, 3, 4], 3: [5]},
        local_orders={0: [1, 0], 1: [2, 3, 4], 3: [5]},
        answers=answers,
        update_eligible=[True, False, True, True, True, True],
        global_step=1,
        epoch=0,
    )
    assert [anchor["answer"] for anchor in state["anchors"]] == [
        "bad",
        "anchor-0",
        "middle-mid",
        "anchor-1",
        "anchor-2",
        "great",
    ]
    assert [item["where"] for item in update["added"]] == ["left", "right", "interval_1"]


def test_internal_anchors_prune_after_three_epoch_misses_but_boundaries_remain():
    manager = _manager()
    state = {
        "anchors": _anchors(4),
        "last_updated_step": -1,
        "last_observed_epoch": -1,
        "last_left_expansion_epoch": -1,
        "last_right_expansion_epoch": -1,
    }
    update = None
    for epoch in range(3):
        before = copy.deepcopy(state["anchors"])
        update = manager._update_buffer(
            state=state,
            anchors_before=before,
            buckets={},
            local_orders={},
            answers=[],
            update_eligible=[],
            global_step=epoch + 1,
            epoch=epoch,
        )
    assert [anchor["answer"] for anchor in state["anchors"]] == ["anchor-0", "anchor-3"]
    assert len(update["pruned"]) == 2


def test_alpaca_eval2_randomized_pairwise_mapping_and_outputs():
    original = alpaca_eval2._request_judge

    def fake_judge(prompt, config):
        del config
        response_a = prompt.split("RESPONSE A\n", 1)[1].split("\n\nRESPONSE B", 1)[0]
        return '{"winner":"A"}' if response_a == "candidate" else '{"winner":"B"}'

    alpaca_eval2._request_judge = fake_judge
    try:
        with tempfile.TemporaryDirectory() as directory:
            metrics = alpaca_eval2.evaluate_alpacaeval2(
                [
                    {
                        "instruction": "instruction",
                        "candidate_output": "candidate",
                        "reference_output": "reference",
                    }
                ],
                output_dir=directory,
                max_workers=1,
            )
            assert metrics["alpaca_eval2/qwen_win_rate"] == 100.0
            assert Path(directory, "model_outputs.json").is_file()
            assert Path(directory, "qwen_judgments.jsonl").is_file()
    finally:
        alpaca_eval2._request_judge = original


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
