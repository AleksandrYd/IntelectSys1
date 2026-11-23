import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run_id
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    clients = []
    throughput = []
    ttft = []
    for key, level in metrics["levels"].items():
        if level.get("output_tokens_per_s") is None:
            continue
        clients.append(int(key))
        throughput.append(level["output_tokens_per_s"])
        ttft.append(level["ttft_p90_s"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(throughput, ttft, marker="o")
    for c, x, y in zip(clients, throughput, ttft):
        ax.annotate(f"{c} клиента", (x, y))
    ax.set_xlabel("пропускная способность, выходные токены/с")
    ax.set_ylabel("TTFT, 90-й перцентиль, с")
    fig.tight_layout()
    out = run_dir / "throughput_vs_ttft.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
