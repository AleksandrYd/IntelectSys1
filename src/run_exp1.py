import argparse
import json
import platform
import statistics
import sys
import traceback
import uuid
from importlib.metadata import version
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import ROOT, append_jsonl, commit_hash, file_sha256, now, setup_mlflow, write_json
from llm_client import make_client, timed_chat

import mlflow


def load_rows(length: int) -> list[dict]:
    found = []
    with (ROOT / "data" / "requests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["series"] == "exp1" and row["target_prompt_tokens"] == length:
                found.append(row)
    return found


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    found = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                found.add(json.loads(line)["id"])
    return found


def record(client, endpoint, run_dir: Path, request_id: str, condition: str, messages, cfg, warmup: bool) -> None:
    results = run_dir / "results.jsonl"
    if request_id in done_ids(results):
        print(f"уже есть {request_id}")
        return
    row = {
        "id": request_id,
        "series": cfg["series"],
        "condition": condition,
        "warmup": warmup,
        "prompt_tokens": None,
        "output_tokens": None,
        "ttft_s": None,
        "tpot_s": None,
        "e2e_s": None,
        "finish_reason": None,
        "status": "failed",
        "error": None,
    }
    try:
        measured = timed_chat(
            client,
            endpoint["model"],
            messages,
            max_tokens=cfg["output_tokens"],
            temperature=cfg["temperature"],
            ignore_eos=cfg["ignore_eos"],
        )
        row.update(
            prompt_tokens=measured["prompt_tokens"],
            output_tokens=measured["output_tokens"],
            ttft_s=measured["ttft_s"],
            tpot_s=measured["tpot_s"],
            e2e_s=measured["e2e_s"],
            finish_reason=measured["finish_reason"],
            status="ok",
        )
        if measured["output_tokens"] != cfg["output_tokens"]:
            row["status"] = "failed"
            row["error"] = f"output_tokens={measured['output_tokens']}"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    append_jsonl(results, row)
    print(
        f"{request_id}: status={row['status']} prompt={row['prompt_tokens']} "
        f"out={row['output_tokens']} TTFT={row['ttft_s']}"
    )


def summarize(run_dir: Path) -> dict:
    rows = []
    with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    by_length = {}
    cache = []
    for row in rows:
        if row.get("warmup") or row["status"] != "ok":
            continue
        if row["condition"] == "prefix_cache":
            cache.append(row["ttft_s"])
            continue
        by_length.setdefault(row["condition"], []).append(row)
    metrics = {"n_rows": len(rows), "n_failed": sum(row["status"] != "ok" for row in rows)}
    for condition, group in sorted(by_length.items()):
        ttft = [row["ttft_s"] for row in group]
        speed = [row["prompt_tokens"] / row["ttft_s"] for row in group]
        metrics[condition] = {
            "n": len(group),
            "ttft_median_s": statistics.median(ttft),
            "ttft_min_s": min(ttft),
            "ttft_max_s": max(ttft),
            "prefill_tps_median": statistics.median(speed),
        }
    if cache:
        metrics["prefix_cache_ttft_s"] = cache
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "exp1.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    endpoint = yaml.safe_load((ROOT / "endpoint.yaml").read_text(encoding="utf-8"))
    commit = "debug" if args.debug else commit_hash()
    run_id = args.run_id or uuid.uuid4().hex
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"experiment": cfg, "endpoint": endpoint}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "run_id": run_id,
            "work": "lr1",
            "series": cfg["series"],
            "status": "running",
            "started_at": now(),
            "ended_at": None,
            "commit": commit,
            "debug": args.debug,
            "versions": {
                "python": platform.python_version(),
                "openai": version("openai"),
                "mlflow": version("mlflow"),
            },
            "device": "cpu",
            "seed": 0,
            "temperature": cfg["temperature"],
            "requests_sha256": file_sha256(ROOT / "data" / "requests.jsonl"),
            "model": endpoint["model"],
            "server": endpoint["server"],
            "server_version": endpoint["server_version"],
            "weight_format": endpoint["weight_format"],
            "weight_sha256": endpoint["weight_sha256"],
        }
        write_json(metadata_path, metadata)

    client = make_client(endpoint["base_url"], endpoint["api_key"])
    setup_mlflow("lr1")
    planned = []
    for length in cfg["lengths"]:
        rows = load_rows(length)[: cfg["repeats"]]
        if len(rows) < cfg["repeats"]:
            raise SystemExit(f"для длины {length} в requests.jsonl меньше {cfg['repeats']} запросов")
        for row in rows:
            planned.append((row["id"], f"len_{length}", row["messages"], False))
    cache_row = load_rows(cfg["cache_length"])[0]
    for index in range(1, cfg["cache_repeats"] + 1):
        planned.append((f"cache-{cfg['cache_length']}-{index}", "prefix_cache", cache_row["messages"], False))
    if args.limit is not None:
        planned = planned[: args.limit]

    try:
        with mlflow.start_run(run_name=run_id):
            mlflow.set_tags({"work": "lr1", "series": cfg["series"], "commit": commit, "model": endpoint["source_model"]})
            mlflow.log_params({"output_tokens": cfg["output_tokens"], "temperature": cfg["temperature"], "server": endpoint["server_version"]})
            record(
                client,
                endpoint,
                run_dir,
                "warmup",
                "warmup",
                [{"role": "user", "content": "Reply with the single word: pong"}],
                cfg,
                True,
            )
            for request_id, condition, messages, warmup in planned:
                record(client, endpoint, run_dir, request_id, condition, messages, cfg, warmup)
            metrics = summarize(run_dir)
            write_json(run_dir / "metrics.json", metrics)
            metadata["status"] = "completed"
            metadata["ended_at"] = now()
            write_json(metadata_path, metadata)
            flat = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
            if flat:
                mlflow.log_metrics(flat)
            mlflow.log_artifacts(str(run_dir))
    except Exception:
        metadata["status"] = "failed"
        metadata["ended_at"] = now()
        metadata["traceback"] = traceback.format_exc()
        write_json(metadata_path, metadata)
        raise
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"run_id = {run_id}")


if __name__ == "__main__":
    main()