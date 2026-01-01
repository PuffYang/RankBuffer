# Anonymous ICLR 2027 Code Submission

This repository contains the anonymized implementation accompanying the ICLR
2027 submission. It includes the RankBuffer data pipeline, reward manager,
training launcher, evaluation utilities, tests, and the required modifications
to a vendored snapshot of [verl](https://github.com/verl-project/verl) at
commit `68227b90735a1c65b76933652c8340f5c4f5e324`.

No submission-author names, affiliations, private endpoints, credentials,
local paths, experiment logs, checkpoints, or Git history are included.
Public third-party attribution and license notices are retained.

## Quick start

1. Install the dependencies required by `verl/` for your accelerator and
   rollout backend.
2. Obtain the public Qwen3-8B model and prepare the Nectar training split in
   the JSONL format documented in `README_RANK_BUFFER.md`.
3. Export machine-local paths and judge configuration. See `.env.example` for
   the required variables; do not commit a populated `.env` file.
4. Run `pytest -q tests/test_rank_buffer.py` for the lightweight tests.
5. Run `./start.sh` to validate the data and launch training.

The launcher defaults to console-only logging and a judge endpoint on
`localhost`. External experiment tracking is opt-in and all credentials must be
provided through the environment.

## Details

See [README_RANK_BUFFER.md](README_RANK_BUFFER.md) for the algorithm, data
schema, training defaults, persistence behavior, and evaluation setup. The
third-party `verl/` snapshot retains its original license and notices.

## Anonymity

Please report reproducibility questions through the anonymous review channel.
Identity-bearing acknowledgements and contact information will be added only
after the review period.
