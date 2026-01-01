from __future__ import annotations

import os

import requests


def main() -> None:
    api_url = os.environ["JUDGE_API_URL"]
    api_key = os.environ.get("JUDGE_API_KEY", "EMPTY")
    model = os.environ["JUDGE_MODEL_NAME"]
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 1,
    }
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(api_url, headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    print(f"Judge endpoint preflight passed: model={model}, url={api_url}")


if __name__ == "__main__":
    main()
