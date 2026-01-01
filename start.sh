#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="${SCRIPT_DIR}/verl"
CONFIG_PATH="${VERL_DIR}/examples/sglang_multiturn/config"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/models/Qwen3-8B}"
TRAIN_DATA="${TRAIN_DATA:-${SCRIPT_DIR}/data/train.jsonl}"
PROMPT_FILE="${SCRIPT_DIR}/prompt.txt"
DATASET_CLASS_FILE="${SCRIPT_DIR}/rank_buffer_dataset.py"
REWARD_MANAGER_FILE="${SCRIPT_DIR}/rank_buffer_reward.py"
ALPACA_EVAL_FILE="${SCRIPT_DIR}/data/alpaca_eval2/alpaca_eval_gpt4_baseline.json"
ALPACA_EVALUATOR_FILE="${SCRIPT_DIR}/alpaca_eval2.py"
DATA_VALIDATOR_FILE="${SCRIPT_DIR}/validate_rank_buffer_data.py"
BUFFER_MIGRATOR_FILE="${SCRIPT_DIR}/migrate_buffer_state.py"
JUDGE_PREFLIGHT_FILE="${SCRIPT_DIR}/check_judge_endpoint.py"

for required_file in \
    "${TRAIN_DATA}" \
    "${PROMPT_FILE}" \
    "${DATASET_CLASS_FILE}" \
    "${REWARD_MANAGER_FILE}" \
    "${ALPACA_EVAL_FILE}" \
    "${ALPACA_EVALUATOR_FILE}" \
    "${DATA_VALIDATOR_FILE}" \
    "${BUFFER_MIGRATOR_FILE}" \
    "${JUDGE_PREFLIGHT_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done
python3 "${DATA_VALIDATOR_FILE}" "${TRAIN_DATA}"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing Qwen3-8B model directory: ${MODEL_PATH}" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs"
ROLLOUT_DIR="${SCRIPT_DIR}/rollouts/${TIMESTAMP}"
CKPT_DIR="${SCRIPT_DIR}/checkpoint"
WANDB_DIR="${SCRIPT_DIR}/wandb"
BUFFER_STATE_DIR="${BUFFER_STATE_DIR:-${SCRIPT_DIR}/buffer_state}"
BUFFER_HISTORY_DIR="${BUFFER_HISTORY_DIR:-${BUFFER_STATE_DIR}/history}"
ALPACA_OUTPUT_DIR="${SCRIPT_DIR}/alpaca_eval2_results"
TOKEN_USAGE_DIR="${SCRIPT_DIR}/token_usage"
TOKEN_USAGE_LOG="${TOKEN_USAGE_DIR}/train_${TIMESTAMP}.jsonl"

# A failed run can write buffer state before the first model checkpoint exists.
# Reset only when explicitly requested, and archive instead of deleting it.
if [[ "${RESET_BUFFER_STATE:-0}" == "1" && -d "${BUFFER_STATE_DIR}" ]]; then
    BUFFER_STATE_BACKUP="${BUFFER_STATE_DIR}_before_${TIMESTAMP}"
    mv -- "${BUFFER_STATE_DIR}" "${BUFFER_STATE_BACKUP}"
    echo "Archived previous buffer state to: ${BUFFER_STATE_BACKUP}"
fi
mkdir -p \
    "${LOG_DIR}" \
    "${ROLLOUT_DIR}" \
    "${CKPT_DIR}" \
    "${WANDB_DIR}" \
    "${BUFFER_STATE_DIR}" \
    "${BUFFER_HISTORY_DIR}" \
    "${ALPACA_OUTPUT_DIR}" \
    "${TOKEN_USAGE_DIR}"
python3 "${BUFFER_MIGRATOR_FILE}" "${BUFFER_STATE_DIR}"

export PYTHONUNBUFFERED=1
export WANDB_DIR
export VLLM_USE_V1=1
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
if [[ -n "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

unset CC CXX CUDAHOSTCXX NVCC_APPEND_FLAGS
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export NVCC_APPEND_FLAGS="-ccbin /usr/bin/g++"

export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}"
export JUDGE_API_URL="${JUDGE_API_URL:-http://localhost:8000/v1/chat/completions}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
if [[ "${SKIP_JUDGE_PREFLIGHT:-0}" != "1" ]]; then
    python3 "${JUDGE_PREFLIGHT_FILE}"
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-20}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_LOG_SAMPLE_COUNT="${ROLLOUT_LOG_SAMPLE_COUNT:-$((TRAIN_BATCH_SIZE * ROLLOUT_N))}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
INITIAL_BUFFER_SIZE=4

LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"

cd "${VERL_DIR}"
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"
nohup python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name=search_multiturn_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${TRAIN_DATA}" \
    data.val_max_samples=1 \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.shuffle=True \
    data.prompt_key=prompt \
    ++data.apply_chat_template_kwargs.enable_thinking=True \
    data.custom_cls.path="${DATASET_CLASS_FILE}" \
    data.custom_cls.name=RankBufferPromptDataset \
    ++data.system_prompt_file="${PROMPT_FILE}" \
    ++data.initial_anchor_count="${INITIAL_BUFFER_SIZE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    reward.num_workers=8 \
    reward.batch_after_rollout=True \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=RankBufferRewardManager \
    reward.reward_manager.module.path="${REWARD_MANAGER_FILE}" \
    reward.reward_manager.module.name=RankBufferRewardManager \
    ++rank_buffer.judge_api_url="${JUDGE_API_URL}" \
    ++rank_buffer.judge_api_key="${JUDGE_API_KEY}" \
    ++rank_buffer.judge_model="${JUDGE_MODEL_NAME}" \
    ++rank_buffer.state_dir="${BUFFER_STATE_DIR}" \
    ++rank_buffer.history_dir="${BUFFER_HISTORY_DIR}" \
    ++rank_buffer.prune_after_epochs=3 \
    ++rank_buffer.format_reward_weight=0.2 \
    ++rank_buffer.max_answer_chars=8000 \
    ++rank_buffer.console_answer_chars=500 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=True \
    trainer.run_standard_validation=False \
    trainer.balance_batch=False \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.project_name=RankBuffer \
    trainer.experiment_name="RankBuffer_K4_${TIMESTAMP}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPT_DIR}" \
    ++trainer.token_usage_log="${TOKEN_USAGE_LOG}" \
    trainer.alpaca_eval2.enable=True \
    trainer.alpaca_eval2.reference_file="${ALPACA_EVAL_FILE}" \
    trainer.alpaca_eval2.evaluator_path="${ALPACA_EVALUATOR_FILE}" \
    trainer.alpaca_eval2.output_dir="${ALPACA_OUTPUT_DIR}" \
    trainer.alpaca_eval2.batch_size=64 \
    trainer.alpaca_eval2.do_sample=True \
    trainer.alpaca_eval2.max_workers=16 \
    trainer.alpaca_eval2.judge.model="${JUDGE_MODEL_NAME}" \
    trainer.alpaca_eval2.judge.api_url="${JUDGE_API_URL}" \
    trainer.alpaca_eval2.judge.api_key="${JUDGE_API_KEY}" \
    ++trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    ++trainer.rollout_log_freq=10 \
    ++trainer.rollout_log_sample_count="${ROLLOUT_LOG_SAMPLE_COUNT}" \
    ++trainer.extra_wandb_metrics='["format_valid","insert_position_norm","judge_error","buffer_size","buffer_added","buffer_pruned"]' \
    > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started RankBuffer GRPO with PID $(cat "${PID_FILE}")"
echo "Model: ${MODEL_PATH}"
echo "Training data: ${TRAIN_DATA}"
echo "Initial buffer: K=${INITIAL_BUFFER_SIZE}, Nectar ranks=[7,5,3,1]"
echo "Rollouts logged per dump: ${ROLLOUT_LOG_SAMPLE_COUNT}"
echo "Judge: ${JUDGE_MODEL_NAME} at ${JUDGE_API_URL}"
echo "Buffer state: ${BUFFER_STATE_DIR}"
echo "Buffer history: ${BUFFER_HISTORY_DIR}"
echo "Token usage: ${TOKEN_USAGE_LOG}"
echo "Log: ${LOG_FILE}"
