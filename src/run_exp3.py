import argparse
import asyncio
import json
import platform
import sys
import time
import traceback
import uuid
from importlib.metadata import version
from pathlib import Path

import yaml
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
from harness import ROOT, append_jsonl, commit_hash, file_sha256, now, setup_mlflow, write_json
from metrics import e2e_s, tpot_s, ttft_s


def load_level(clients: int) -> list[dict]:
    found = []
    prefix = f"e3-c{clients}-"
    with (ROOT / "data" / "requests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["series"] == "exp3" and row["id"].startswith(prefix):
                found.append(row)
    found.sort(key=lambda row: row["id"])
    count = max(16, 4 * clients)
    if len(found) < count:
        raise SystemExit(f"для {clients} клиентов в requests.jsonl только {len(found)} запросов")
    return found[:count]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


async def timed_chat(client: AsyncOpenAI, model: str, messages, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True},
    )
    t1 = None
    t_last = None
    usage = None
    finish_reason = None
    async for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        content = choice.delta.content if choice.delta is not None else None
        if content:
            moment = time.perf_counter()
            if t1 is None:
                t1 = moment
            t_last = moment
    if usage is None or t1 is None or t_last is None:
        raise RuntimeError("сервер не прислал текст или поле usage")
    n = usage.completion_tokens
    return {
        "t0": t0,
        "t_last": t_last,
        "ttft_s": ttft_s(t0, t1),
        "tpot_s": tpot_s(t1, t_last, n),
        "e2e_s": e2e_s(t0, t_last),
        "prompt_tokens": usage.prompt_tokens,
        "output_tokens": n,
        "finish_reason": finish_reason,
    }


async def run_level(client, model, requests, out_len: int, results_path: Path, clients: int) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for row in requests:
        queue.put_nowait(row)

    async def worker() -> None:
        while True:
            item = await queue.get()
            row = {
                "id": item["id"],
                "series": "exp3_load",
                "condition": f"clients_{clients}",
                "warmup": False,
                "clients": clients,
                "prompt_tokens": None,
                "output_tokens": None,
                "ttft_s": None,
                "tpot_s": None,
                "e2e_s": None,
                "t0": None,
                "t_last": None,
                "finish_reason": None,
                "status": "failed",
                "error": None,
            }
            try:
                measured = await timed_chat(client, model, item["messages"], out_len)
                row.update(
                    prompt_tokens=measured["prompt_tokens"],
                    output_tokens=measured["output_tokens"],
                    ttft_s=measured["ttft_s"],
                    tpot_s=measured["tpot_s"],
                    e2e_s=measured["e2e_s"],
                    t0=measured["t0"],
                    t_last=measured["t_last"],
                    finish_reason=measured["finish_reason"],
                    status="ok",
                )
                if measured["output_tokens"] != out_len:
                    row["status"] = "failed"
                    row["error"] = f"output_tokens={measured['output_tokens']}"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            append_jsonl(results_path, row)
            print(
                f"{row['id']}: status={row['status']} TTFT={row['ttft_s']} TPOT={row['tpot_s']}",
                flush=True,
            )
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(clients)]
    await queue.join()
    for task in workers:
        task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


def summarize(rows: list[dict], cfg: dict) -> dict:
    metrics = {"n_rows": len(rows), "n_failed": sum(row["status"] != "ok" for row in rows), "levels": {}}
    for clients in cfg["clients"]:
        group = [row for row in rows if row.get("clients") == clients]
        ok = [row for row in group if row["status"] == "ok" and row["t0"] is not None and row["t_last"] is not None]
        level = {"n": len(group), "n_ok": len(ok)}
        if ok:
            duration = max(row["t_last"] for row in ok) - min(row["t0"] for row in ok)
            tokens = sum(row["output_tokens"] for row in ok)
            ttft = [row["ttft_s"] for row in ok]
            tpot = [row["tpot_s"] for row in ok if row["tpot_s"] is not None]
            good = [
                row for row in ok
                if row["ttft_s"] <= cfg["ttft_limit_s"] and row["tpot_s"] is not None and row["tpot_s"] <= cfg["tpot_limit_s"]
            ]
            level.update(
                duration_s=duration,
                output_tokens_per_s=tokens / duration if duration else None,
                requests_per_s=len(ok) / duration if duration else None,
                ttft_median_s=percentile(ttft, 0.5),
                ttft_p90_s=percentile(ttft, 0.9),
                tpot_median_s=percentile(tpot, 0.5) if tpot else None,
                n_good=len(good),
                goodput_rps=len(good) / duration if duration else None,
            )
        metrics["levels"][str(clients)] = level
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "exp3.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    endpoint = yaml.safe_load((ROOT / "endpoint.yaml").read_text(encoding="utf-8"))
    commit = "debug" if args.debug else commit_hash()
    run_id = uuid.uuid4().hex
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"experiment": cfg, "endpoint": endpoint}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
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
        "ttft_limit_s": cfg["ttft_limit_s"],
        "tpot_limit_s": cfg["tpot_limit_s"],
        "n_slots_expected": 4,
    }
    write_json(run_dir / "metadata.json", metadata)
    setup_mlflow("lr1")
    results_path = run_dir / "results.jsonl"

    async def all_levels() -> None:
        client = AsyncOpenAI(base_url=endpoint["base_url"], api_key=endpoint["api_key"])
        for clients in cfg["clients"]:
            print(f"уровень {clients}", flush=True)
            await run_level(
                client, endpoint["model"], load_level(clients), cfg["output_tokens"], results_path, clients,
            )
        await client.close()

    try:
        with mlflow.start_run(run_name=run_id):
            mlflow.set_tags({"work": "lr1", "series": cfg["series"], "commit": commit, "model": endpoint["source_model"]})
            mlflow.log_params({
                "output_tokens": cfg["output_tokens"],
                "ttft_limit_s": cfg["ttft_limit_s"],
                "tpot_limit_s": cfg["tpot_limit_s"],
                "clients": ",".join(str(c) for c in cfg["clients"]),
            })
            asyncio.run(all_levels())
            rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            metrics = summarize(rows, cfg)
            write_json(run_dir / "metrics.json", metrics)
            metadata["status"] = "completed"
            metadata["ended_at"] = now()
            write_json(run_dir / "metadata.json", metadata)
            mlflow.log_metrics({"n_rows": metrics["n_rows"], "n_failed": metrics["n_failed"]})
            mlflow.log_artifacts(str(run_dir))
    except Exception:
        metadata["status"] = "failed"
        metadata["ended_at"] = now()
        metadata["traceback"] = traceback.format_exc()
        write_json(run_dir / "metadata.json", metadata)
        raise
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"run_id = {run_id}")


if __name__ == "__main__":
    main()
