"""Summarize local metrics and optionally plot them; never uploads logs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--plot", action="store_true", help="Requires the plots extra")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (Path(args.run) / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Empty metrics file")
    print(
        json.dumps(
            {
                "rounds": len(rows),
                "optimizer_updates": rows[-1]["optimizer_updates"],
                "mean_reward": sum(r["reward"] for r in rows) / len(rows),
                "zero_active_group_rounds": sum(r["active_groups"] == 0 for r in rows),
                "total_seconds": sum(r["seconds"] for r in rows),
                "peak_allocated_mib": max(r["peak_allocated_mib"] for r in rows),
            },
            indent=2,
        )
    )
    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for ax, key in zip(axes, ("reward", "optimizer_updates", "peak_allocated_mib")):
            ax.plot([r["step"] for r in rows], [r[key] for r in rows])
            ax.set_ylabel(key)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("Rollout round")
        fig.tight_layout()
        fig.savefig(Path(args.run) / "learning.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
