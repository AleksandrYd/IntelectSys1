import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INSTRUCTION = "Summarize the following text in one sentence."


def render(content: str) -> str:
    return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"


def prompt_length(tokenizer: Tokenizer, content: str) -> int:
    return len(tokenizer.encode(render(content), add_special_tokens=False).ids)


def fit_content(
    tokenizer: Tokenizer,
    book_ids: list[int],
    label: str,
    target: int,
    offset: int,
) -> tuple[str, int]:
    lo = 0
    hi = len(book_ids) - offset
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        excerpt = tokenizer.decode(book_ids[offset : offset + mid])
        content = f"{label}\n{INSTRUCTION}\n{excerpt}"
        if prompt_length(tokenizer, content) <= target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    excerpt = tokenizer.decode(book_ids[offset : offset + best])
    content = f"{label}\n{INSTRUCTION}\n{excerpt}"
    got = prompt_length(tokenizer, content)
    if got != target:
        raise SystemExit(
            f"{label}: после подбора получилось {got} токенов, нужно {target}"
        )
    return content, best


def add_row(
    rows: list[dict],
    tokenizer: Tokenizer,
    book_ids: list[int],
    request_id: str,
    series: str,
    target: int,
    offset: int,
) -> None:
    content, used = fit_content(tokenizer, book_ids, request_id, target, offset)
    rows.append(
        {
            "id": request_id,
            "series": series,
            "target_prompt_tokens": target,
            "local_prompt_tokens": prompt_length(tokenizer, content),
            "book_offset": offset,
            "book_tokens": used,
            "messages": [{"role": "user", "content": content}],
        }
    )


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["pride_and_prejudice"]["tokenizer"]["sha256"]
    tokenizer_path = DATA / "tokenizer.json"
    digest = __import__("hashlib").sha256(tokenizer_path.read_bytes()).hexdigest()
    if digest != recorded:
        raise SystemExit("SHA-256 data/tokenizer.json не совпадает с манифестом")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    book = (DATA / "pride_body.txt").read_text(encoding="utf-8")
    book_ids = tokenizer.encode(book, add_special_tokens=False).ids
    rows: list[dict] = []

    for length in (128, 512, 1024, 2048, 4096):
        for repeat in range(5):
            add_row(
                rows,
                tokenizer,
                book_ids,
                f"e1-len{length}-r{repeat}",
                "exp1",
                length,
                0,
            )

    for out_len in (16, 64, 256, 1024):
        for repeat in range(5):
            add_row(
                rows,
                tokenizer,
                book_ids,
                f"e2-out{out_len}-r{repeat}",
                "exp2",
                1024,
                0,
            )

    running_offset = 0
    for clients in (1, 2, 4):
        count = max(16, 4 * clients)
        for index in range(count):
            add_row(
                rows,
                tokenizer,
                book_ids,
                f"e3-c{clients}-i{index:02d}",
                "exp3",
                1024,
                running_offset,
            )
            running_offset += 64

    out = DATA / "requests.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"запросов: {len(rows)}")
    print(f"токенов книги: {len(book_ids)}")
    print(f"файл: {out}")


if __name__ == "__main__":
    main()