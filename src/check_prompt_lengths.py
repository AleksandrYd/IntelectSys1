import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import make_client, timed_chat

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    endpoint = yaml.safe_load((ROOT / "endpoint.yaml").read_text(encoding="utf-8"))
    client = make_client(endpoint["base_url"], endpoint["api_key"])
    seen = set()
    checked = 0
    with (ROOT / "data" / "requests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            length = row["target_prompt_tokens"]
            if length in seen:
                continue
            seen.add(length)
            result = timed_chat(
                client,
                endpoint["model"],
                row["messages"],
                max_tokens=1,
                temperature=0,
            )
            got = result["prompt_tokens"]
            print(
                f"{row['id']}: сервер {got}, локально {row['local_prompt_tokens']}, "
                f"TTFT {result['ttft_s'] * 1000:.0f} мс"
            )
            if got != row["local_prompt_tokens"]:
                raise SystemExit("длина на сервере не совпала с локальным подсчётом")
            checked += 1
    print(f"проверено длин: {checked}")


if __name__ == "__main__":
    main()