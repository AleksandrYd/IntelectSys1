import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BOOK_URL = "https://gutenberg.pglaf.org/cache/epub/1342/pg1342.txt"
TOKENIZER_URL = (
    "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/tokenizer.json"
)
OPENING = "It is a truth universally acknowledged"
END_MARK = "*** END OF THE PROJECT GUTENBERG EBOOK"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"уже есть: {path.name}")
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    path.write_bytes(urllib.request.urlopen(request, timeout=120).read())
    print(f"скачан: {path.name}")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    raw_path = DATA / "pg1342.txt"
    body_path = DATA / "pride_body.txt"
    tokenizer_path = DATA / "tokenizer.json"
    download(BOOK_URL, raw_path)
    download(TOKENIZER_URL, tokenizer_path)

    raw = raw_path.read_text(encoding="utf-8")
    start = raw.find(OPENING)
    end = raw.find(END_MARK)
    if start < 0 or end < 0 or end <= start:
        raise SystemExit("В файле книги не найдены начало романа и конец Project Gutenberg")
    body = raw[start:end].replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    body_path.write_text(body, encoding="utf-8")

    manifest_path = DATA / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pride_and_prejudice"] = {
        "source": "Project Gutenberg, ebook 1342, Jane Austen, Pride and Prejudice",
        "url": BOOK_URL,
        "version": "зеркало gutenberg.pglaf.org; gutenberg.org на момент загрузки отвечал 503",
        "rule": (
            "Сохранён исходный файл. В запросы берётся текст от первого "
            "'It is a truth universally acknowledged' до маркера конца книги. "
            "Предисловие Gutenberg и лицензия в конец файла отброшены. "
            "Переводы строк приведены к \\n."
        ),
        "n_raw_chars": len(raw),
        "n_body_chars": len(body),
        "n_excluded_chars": len(raw) - len(body),
        "tokenizer": {
            "source": "Hugging Face Qwen/Qwen3-4B-Instruct-2507",
            "file": "data/tokenizer.json",
            "sha256": sha256(tokenizer_path),
        },
        "files": {
            "data/pg1342.txt": {"sha256": sha256(raw_path), "n_records": 1},
            "data/pride_body.txt": {
                "sha256": sha256(body_path),
                "n_records": 1,
                "n_chars": len(body),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"тело романа: {len(body)} символов")
    print(f"отброшено: {len(raw) - len(body)} символов")
    print(f"sha256 книги: {manifest['pride_and_prejudice']['files']['data/pg1342.txt']['sha256']}")


if __name__ == "__main__":
    main()