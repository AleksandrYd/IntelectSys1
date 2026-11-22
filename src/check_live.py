import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import make_client, timed_chat

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    endpoint = yaml.safe_load((ROOT / "endpoint.yaml").read_text(encoding="utf-8"))
    client = make_client(endpoint["base_url"], endpoint["api_key"])
    result = timed_chat(
        client,
        endpoint["model"],
        [
            {
                "role": "user",
                "content": "Перечисли числа от 1 до 20 через пробел.",
            }
        ],
        max_tokens=64,
        temperature=endpoint["temperature"],
    )
    print(f"prompt_tokens = {result['prompt_tokens']}")
    print(f"output_tokens = {result['output_tokens']}")
    print(f"finish_reason = {result['finish_reason']}")
    print(f"TTFT = {result['ttft_s'] * 1000:.0f} мс")
    tpot = result["tpot_s"]
    if tpot is None:
        print("TPOT = нет (в ответе меньше двух токенов)")
    else:
        print(f"TPOT = {tpot * 1000:.1f} мс")
    print(f"полное время = {result['e2e_s']:.2f} с")
    print("--- текст ---")
    print(result["text"])


if __name__ == "__main__":
    main()