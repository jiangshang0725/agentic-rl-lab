"""Check live JSONL handling and the real, optional local dashboard file format."""

import json
from pathlib import Path
import sqlite3

import pytest

from local_rl import swanlab_export as viewer


def row(step, reward=0.5):
    return {"step": step, "reward": reward, "loss": 0.0, "optimizer_updates": step - 1}


def test_live_tail_is_not_lost_or_reimported(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = (json.dumps(row(21)) + "\n").encode()
    second = (json.dumps(row(22, 0.75)) + "\n").encode()
    path.write_bytes(first + second[:20])
    records, offset = viewer.read_metrics(path, follow=True)
    assert records == [row(21)]
    assert offset == len(first)
    with path.open("ab") as stream:
        stream.write(second[20:])
    records, offset = viewer.read_metrics(path, offset, last_step=21, follow=True)
    assert records == [row(22, 0.75)]
    assert viewer.read_metrics(path, offset, last_step=22, follow=True) == ([], offset)


def test_snapshot_accepts_final_record_without_newline(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(row(1)), encoding="utf-8")
    assert viewer.read_metrics(path)[0] == [row(1)]
    assert viewer.read_metrics(path, follow=True) == ([], 0)


@pytest.mark.parametrize(
    "content",
    [
        json.dumps(row(0)),
        json.dumps(row(True)),
        json.dumps(row(1.5)),
        json.dumps(row(1, float("nan"))),
        json.dumps(row(1, float("inf"))),
        json.dumps(row(1, "0.5")),
        json.dumps({"step": 1, "reward": 1}),
        json.dumps(row(2)) + "\n" + json.dumps(row(2)),
        json.dumps(row(2)) + "\n" + json.dumps(row(1)),
        '{"step": broken}\n',
    ],
)
def test_invalid_history_is_rejected(tmp_path, content):
    path = tmp_path / "metrics.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        viewer.read_metrics(path)


def test_truncated_live_file_is_rejected(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="truncated"):
        viewer.read_metrics(path, offset=100, follow=True)


def test_follow_waits_for_a_complete_round_and_stops_at_target(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"steps": 22}', encoding="utf-8")
    metrics = source / "metrics.jsonl"
    calls = []
    states = []

    class Session:
        def log(self, data, step):
            calls.append((step, data))

        def finish(self, state):
            states.append(state)

    monkeypatch.setattr(viewer, "local_session", lambda *args: Session())
    pending = iter([json.dumps(row(21)) + "\n", json.dumps(row(22)), "\n"])

    def append(_):
        with metrics.open("a", encoding="utf-8") as stream:
            stream.write(next(pending))

    monkeypatch.setattr(viewer.time, "sleep", append)
    assert viewer.export(source, tmp_path / "board", follow=True) == 2
    assert [step for step, _ in calls] == [21, 22]
    assert states == ["success"]


def test_real_local_sdk_preserves_values_without_http(tmp_path, monkeypatch):
    pytest.importorskip("swanlab")
    pytest.importorskip("swanboard")
    import requests
    import yaml

    # An old account/environment setting must not silently upload this import.
    monkeypatch.setenv("SWANLAB_MODE", "online")

    def no_http(*args, **kwargs):
        raise AssertionError("Local logging must not make HTTP requests")

    monkeypatch.setattr(requests.sessions.Session, "request", no_http)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"steps": 22, "device": "cuda"}', encoding="utf-8")
    expected = [row(21, 0.25), row(22, 0.75)]
    (source / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in expected), encoding="utf-8"
    )
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    destination = tmp_path / "board"
    assert viewer.export(source, destination, name="historical-test") == 2
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    with sqlite3.connect(destination / "runs.swanlab") as connection:
        run_id, name = connection.execute("select run_id, name from experiment").fetchone()
        tags = connection.execute("select folder, name from tag").fetchall()
    assert name == "historical-test"
    assert {name for _, name in tags} == {"reward", "loss", "optimizer_updates"}
    for folder_name, key in tags:
        folder = destination / run_id / "logs" / folder_name
        actual = [
            json.loads(line)
            for path in sorted(folder.glob("*.log"))
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert [(r["index"], r["data"]) for r in actual] == [
            (r["step"], r[key]) for r in expected
        ]
    config = yaml.safe_load((destination / run_id / "files/config.yaml").read_text("utf-8"))
    assert config["training"]["value"] == {"steps": 22, "device": "cuda"}
    assert Path(config["imported_from"]["value"]) == source
    assert config["source_status_unchecked"]["value"] is True
