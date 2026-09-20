import json

import numpy as np

from artifactbench.bench import deterministic_select
from artifactbench.data.manifest import load_manifest
from artifactbench.metrics.source_level import measure_source, summarize
from artifactbench.models.artifactnet import ArtifactNetModel


class MeanModel:
    def forward(self, audio):
        return float(np.mean(audio))


class NonFiniteModel:
    def forward(self, audio):
        return float("nan")


def test_deterministic_selection_is_order_independent():
    rows = [
        {"track_id": f"track-{index}", "source": "s", "label": "ai"}
        for index in range(20)
    ]
    left = deterministic_select(rows, 7, seed=42)
    right = deterministic_select(list(reversed(rows)), 7, seed=42)
    assert [row["track_id"] for row in left] == [row["track_id"] for row in right]
    assert len(deterministic_select(rows, 0, seed=42)) == 20


def test_tracks_manifest_is_supported(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "tracks": [
            {"track_id": "a", "source": "one", "label": "ai", "bench_origin": "test"},
            {"track_id": "b", "source": "two", "label": "real", "bench_origin": "train"},
        ]
    }))
    rows, by_source = load_manifest(manifest, bench_origin="test")
    assert [row["track_id"] for row in rows] == ["a"]
    assert set(by_source) == {"one"}


def test_failures_are_not_silently_dropped(tmp_path):
    missing = tmp_path / "missing.wav"
    rows = [
        {"track_id": "missing", "source": "real", "label": "real", "path": str(missing)},
        {"track_id": "unresolved", "source": "real", "label": "real"},
    ]
    scored, failures = measure_source(rows, MeanModel(), verbose=False)
    assert scored == []
    assert {failure["stage"] for failure in failures} == {"decode", "resolve"}


def test_summary_records_attempts_failures_and_threshold():
    summary = summarize(
        [
            {"prob": 0.4, "label": "ai"},
            {"prob": 0.8, "label": "ai"},
        ],
        attempted=3,
        threshold=0.7,
    )
    assert summary["n"] == 2
    assert summary["attempted"] == 3
    assert summary["failures"] == 1
    assert summary["threshold"] == 0.7
    assert summary["tpr"] == 0.5


def test_artifactnet_uses_frozen_chunk_count_for_long_tracks():
    model = ArtifactNetModel(n_chunks=7)
    calls = []
    model._forward_chunk = lambda chunk: calls.append(len(chunk)) or 0.25
    probability = model.forward(np.zeros(44100 * 180, dtype=np.float32))
    assert probability == 0.25
    assert len(calls) == 7


def test_nonfinite_probability_is_an_inference_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "artifactbench.metrics.source_level.load_audio_mono",
        lambda path: np.zeros(32, dtype=np.float32),
    )
    rows = [{"track_id": "nan", "source": "ai", "label": "ai", "path": str(tmp_path / "x.wav")}]
    scored, failures = measure_source(rows, NonFiniteModel(), verbose=False)
    assert scored == []
    assert failures[0]["stage"] == "inference"
    assert failures[0]["error_type"] == "ValueError"
