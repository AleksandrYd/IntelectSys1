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
    ok = [row for row in rows if row["status"] == "ok" and not row["warmup"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for out_len in (16, 64, 256, 1024):
        group = [row for row in ok if row["target_output_tokens"] == out_len]
        if not group:
            continue
        xs = [out_len - 1] * len(group)
        ys = [row["e2e_s"] for row in group]
        ax.scatter(xs, ys, label=f"{out_len} токенов")
    if len(ok) >= 2:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        intercept = metrics["fit_intercept_s"]
        slope = metrics["fit_slope_s"]
        x_line = [0, 1023]
        ax.plot(x_line, [intercept + slope * x for x in x_line], color="black", label="прямая")
    ax.set_xlabel("токены ответа после первого, N − 1")
    ax.set_ylabel("полное время, с")
    ax.legend()
    fig.tight_layout()
    out = run_dir / "e2e_vs_output.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()