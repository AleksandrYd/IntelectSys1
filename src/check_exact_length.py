import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import make_client, timed_chat

ROOT = Path(__file__).resolve().parents[1]
MESSAGES = [
    {
        "role": "user",
        "content": "Reply with the single word: pong",
    }
]


def main() -> None:
    endpoint = yaml.safe_load((ROOT / "endpoint.yaml").read_text(encoding="utf-8"))
    client = make_client(endpoint["base_url"], endpoint["api_key"])
    first = timed_chat(client, endpoint["model"], MESSAGES, max_tokens=32, temperature=0)
    second = timed_chat(client, endpoint["model"], MESSAGES, max_tokens=32, temperature=0)
    same = first["text"] == second["text"]
    print("первый ответ:", first["text"].replace("\n", " "))
    print("второй ответ:", second["text"].replace("\n", " "))
    print("тексты совпали:", "да" if same else "нет")

    exact = timed_chat(
        client,
        endpoint["model"],
        MESSAGES,
        max_tokens=64,
        temperature=0,
        ignore_eos=True,
    )
    print(f"output_tokens = {exact['output_tokens']}")
    print(f"finish_reason = {exact['finish_reason']}")
    print(f"TTFT = {exact['ttft_s'] * 1000:.0f} мс")
    if exact["tpot_s"] is None:
        print("TPOT = нет")
    else:
        print(f"TPOT = {exact['tpot_s'] * 1000:.1f} мс")
    if exact["output_tokens"] != 64:
        raise SystemExit("ожидалось ровно 64 токена")
    print("проверка длины пройдена")


if __name__ == "__main__":
    main()