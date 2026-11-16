# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["swanlab[dashboard]==0.10.4"]
# ///
"""Read local metrics into SwanLab's local dashboard, without touching training.

Run with uv to install the optional viewer in its own cached environment:
    uv run local_rl/swanlab_export.py outputs/grpo-20 --watch
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def read_metrics(path, offset=0, last_step=0, follow=False):
    """Read complete JSONL records; leave a live writer's unfinished tail for later."""
    path = Path(path)
    if not path.exists() and follow:
        return [], offset
    with path.open("rb") as stream:
        if path.stat().st_size < offset:
            raise ValueError("Metrics were truncated; stop editing logs and restart the importer")
        stream.seek(offset)
        records = []
        while line := stream.readline():
            if follow and not line.endswith(b"\n"):
                break
            offset = stream.tell()
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Invalid JSON in {path} near byte {offset}") from exc
            if (
                not isinstance(row, dict)
                or not {"step", "reward", "loss", "optimizer_updates"} <= row.keys()
            ):
                raise ValueError("Each metric row needs step, reward, loss and optimizer_updates")
            if type(row["step"]) is not int or row["step"] <= last_step:
                raise ValueError(
                    "Metric steps must be positive integers in strictly increasing order"
                )
            for key, value in row.items():
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"Metric {key!r} must be a finite number")
            last_step = row["step"]
            records.append(row)
    return records, offset


def local_session(logdir, name, config, source):
    try:
        import swanlab
        import swanboard  # noqa: F401 -- fail early if the dashboard extra is absent
    except ImportError as exc:
        raise RuntimeError(
            "Viewer dependency missing. Run: uv run local_rl/swanlab_export.py <run-directory>"
        ) from exc
    # Never report the importer's current hardware utilization as historical training data.
    settings = swanlab.Settings(
        interactive=False,
        terminal={"proxy_type": "none"},
        probe={
            "hardware": False,
            "runtime": False,
            "requirements": False,
            "conda": False,
            "git": False,
            "swanlab": False,
            "monitor": False,
        },
    )
    session = swanlab.init(
        project="agentic-rl-local",
        name=name,
        mode="local",  # Explicit mode takes precedence over environment/account defaults.
        log_dir=str(logdir),
        config={"training": config, "imported_from": str(source), "source_status_unchecked": True},
        settings=settings,
    )
    return session


def export(run, logdir="outputs/swanlog", name=None, follow=False, poll_interval=1.0):
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("poll-interval must be finite and positive")
    source = Path(run).resolve()
    config_path = source / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing {config_path}; start training and wait for config.json first")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")
    target_step = config.get("steps")
    if type(target_step) is not int or target_step < 1:
        raise ValueError("config.json must contain a positive integer steps value")
    metrics_path = source / "metrics.jsonl"
    records, offset = read_metrics(metrics_path, follow=follow)
    if not records and not follow:
        raise ValueError("No metrics to import; wait for at least one completed training round")
    destination = Path(logdir).resolve()
    if destination == source or source.is_relative_to(destination):
        raise ValueError("Use a separate dashboard directory, such as outputs/swanlog")
    session = local_session(destination, name or source.name, config, source)
    count = last_step = 0
    state = "success"
    print(f"Reading {metrics_path}; SwanLab mode=local. No training is started.", flush=True)
    try:
        while True:
            for row in records:
                session.log(
                    {key: value for key, value in row.items() if key != "step"}, step=row["step"]
                )
                last_step = row["step"]
                count += 1
            if records:
                print(f"Imported {count} rounds; last step={last_step}.", flush=True)
            if not follow or last_step >= target_step:
                break
            time.sleep(poll_interval)
            records, offset = read_metrics(metrics_path, offset, last_step, follow=True)
    except KeyboardInterrupt:
        state = "aborted"
        raise
    except Exception:
        state = "crashed"
        raise
    finally:
        session.finish(state=state)
    print(
        "Import finished. Dashboard status describes this import, not training completion. "
        "Check the original checkpoint and evaluation separately.",
        flush=True,
    )
    return count


def watch(logdir, port=5092):
    destination = Path(logdir).resolve()
    if not (destination / "runs.swanlab").is_file():
        raise ValueError("No local dashboard database; import at least one experiment first")
    print(
        f"Starting viewer at http://127.0.0.1:{port}. "
        "If that port is busy, use the address printed by SwanLab. "
        "Ctrl+C stops only the viewer.",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "swanlab",
            "watch",
            str(destination),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", nargs="?", help="Training output directory with config.json/metrics.jsonl"
    )
    parser.add_argument("--logdir", default="outputs/swanlog")
    parser.add_argument(
        "--name", help="Dashboard experiment name; default is the source folder name"
    )
    parser.add_argument(
        "--follow", action="store_true", help="Read new rounds until configured steps or Ctrl+C"
    )
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--watch", action="store_true", help="Start the local dashboard after importing"
    )
    parser.add_argument(
        "--watch-only", action="store_true", help="Reopen existing dashboard without reimporting"
    )
    parser.add_argument("--port", type=int, default=5092)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.watch_only:
        if args.run or args.follow or args.watch or args.name:
            parser.error("--watch-only is used without a run, --follow, --watch or --name")
    elif not args.run:
        parser.error("provide a training output directory, or use --watch-only")
    elif args.follow and args.watch:
        parser.error("Use --follow here and --watch-only in a separate terminal")
    try:
        if not args.watch_only:
            export(args.run, args.logdir, args.name, args.follow, args.poll_interval)
        if args.watch or args.watch_only:
            watch(args.logdir, args.port)
    except KeyboardInterrupt:
        print("Stopped this importer/viewer; separate training processes are unaffected.")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Local SwanLab: {exc}") from exc


if __name__ == "__main__":
    main()
