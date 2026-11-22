import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run_id
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    lengths = []
    medians = []
    lows = []
    highs = []
    for length in (128, 512, 1024, 2048, 4096):
        group = [
            row["ttft_s"]
            for row in rows
            if row["condition"] == f"len_{length}" and row["status"] == "ok" and not row["warmup"]
        ]
        if not group:
            continue
        lengths.append(length)
        medians.append(statistics.median(group))
        lows.append(min(group))
        highs.append(max(group))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        lengths,
        medians,
        yerr=[[m - lo for m, lo in zip(medians, lows)], [hi - m for m, hi in zip(medians, highs)]],
        fmt="o",
        capsize=4,
    )
    ax.set_xlabel("длина запроса, токены")
    ax.set_ylabel("TTFT, с")
    ax.set_title(args.run_id)
    fig.tight_layout()
    out = run_dir / "ttft_vs_length.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()