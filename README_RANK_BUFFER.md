# RankBuffer training

The training path is now single-turn Qwen3-8B GRPO with group-level dynamic
buffers. No web-search/Jina tool is initialized. Qwen3's native chat template
runs with `enable_thinking=True`; `prompt.txt` only requires the final response
to be wrapped in `<answer>...</answer>`.

## Data

Training expects the path in `TRAIN_DATA` (default: `data/train.jsonl`). The
training file is not included; obtain Nectar from its public source and prepare
the 1,000-example split described in the paper.
Each row must contain `query`, `rubrics`, and exactly seven ranked `answers`.
The generated Nectar rank is best-to-worst, while buffers are stored
worst-to-best. `start.sh` validates the complete JSONL before allocating the
training job and never falls back to rubric-free parquet data. The supplied 1k
file contains two genuinely empty historical responses; they remain in their
original rank positions but are rendered as `[EMPTY RESPONSE]` so the judge
does not receive an ambiguous blank anchor. Duplicate answer text is retained
because every initial buffer must preserve all seven Nectar rank positions.

AlpacaEval 2 uses the checked-in official 805-example GPT-4-Turbo reference
file. Every 50 steps, sampled policy outputs are extracted only from the sole
`<answer>` block and compared pairwise with Qwen3-30B-A3B. This is a Qwen-judged
development win rate, not the official GPT-4 length-controlled score.

## Buffer and reward

Every query starts with its seven Nectar answers. Any rollout with exactly one
extractable `<answer>` block is judged even when its outer thinking format is
invalid; only exact native-think-plus-answer output receives the format reward.
Coarse judgments choose an
insertion position from `0..len(buffer)`. Only buckets containing at least two
valid rollouts invoke the local worst-to-best listwise judge. The assembled
rollout order maps linearly to `[-1, 1]`, then adds a `0.2` valid-format reward.

At most one valid rollout expands each boundary per query per epoch. An internal
bucket with at least three valid rollouts inserts its lower median. Internal
anchors whose left and right intervals both miss for three consecutive observed
epochs are pruned; boundary anchors are never pruned. State is persisted under
`buffer_state/`, so it should be transferred/restored together with checkpoints.

## 8×H800 defaults

The defaults use 20 prompts × 8 rollouts (160 trajectories) per step, 4,096-token
responses, 8,192 total model length, TP=1, and 30 epochs. Compared with the old
18k-token research-agent rollout, this gives much higher concurrency without
inflating the policy batch. With 1,000 unfiltered samples this is 50 steps per
epoch and about 240,000 training rollouts over 30 epochs. Override the shell
variables (`TRAIN_BATCH_SIZE`, `ROLLOUT_N`, `TOTAL_EPOCHS`) after observing GPU
memory and response-length histograms.

Sampling uses Qwen3's thinking-mode recommendations: temperature 0.6, top-p
0.95, and top-k 20. AlpacaEval generation also samples because Qwen explicitly
warns against greedy decoding for the original Qwen3 thinking model.

Launch with `MODEL_PATH=/path/to/model TRAIN_DATA=/path/to/train.jsonl
./start.sh`. Set `JUDGE_API_URL`, `JUDGE_MODEL_NAME`, `JUDGE_API_KEY`, or
`BUFFER_STATE_DIR` in the environment as needed. Logging defaults to the
console. To enable Weights & Biases, set `WANDB_API_KEY` in the environment and
set `TRAINER_LOGGER='["console","wandb"]'`; never store credentials in this
repository.
The launcher sends one minimal judge request before loading the policy model so
an incorrect host, port, route, or judge model fails fast. Set
`SKIP_JUDGE_PREFLIGHT=1` only when this check is intentionally unavailable.
If a job fails before its first checkpoint but has already judged rollouts,
restart once with `RESET_BUFFER_STATE=1 ./start.sh`; the old state directory is
archived rather than deleted.
