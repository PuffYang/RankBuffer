# AlpacaEval 2 reference data

`alpaca_eval_gpt4_baseline.json` is the official 805-example AlpacaEval 2
evaluation set paired with the GPT-4-Turbo (`gpt4_1106_preview`) reference
outputs.

- Dataset: https://huggingface.co/datasets/tatsu-lab/alpaca_eval
- Upstream project: https://github.com/tatsu-lab/alpaca_eval
- Downloaded: 2026-08-12
- SHA-256: `83db546b872ddebee8965fd05fa48461ee3c32bc695c62fb57f2d214ff741ec4`

This project uses the official prompts and references, but substitutes the
configured Qwen3-30B-A3B deployment for AlpacaEval's official GPT-4 annotator.
The reported metric is therefore named `alpaca_eval2/qwen_win_rate`; it is a
development win rate and must not be presented as the official length-controlled
AlpacaEval 2 leaderboard score.
