import json

import pytest

from scripts.analyze_frozen_protocol import (
    choose_threshold,
    failure_as_error_metrics,
)
from scripts.audit_manifest import audit
from scripts.build_public_release import public_row
from scripts.build_public_results import assert_public_safe


def test_threshold_selection_uses_fpr_constraint_and_lowest_tie():
    rows = [
        {"label": 0, "prob": 0.10},
        {"label": 0, "prob": 0.20},
        {"label": 1, "prob": 0.70},
        {"label": 1, "prob": 0.90},
    ]

    threshold, metrics = choose_threshold(rows, max_fpr=0.0)

    assert threshold == 0.70
    assert metrics["tpr"] == 1.0
    assert metrics["fpr"] == 0.0


def test_failure_as_error_keeps_missing_rows_out_of_score_metrics():
    scored = [
        {"label": 0, "prob": 0.10},
        {"label": 1, "prob": 0.90},
    ]
    missing = [{"label": "ai"}, {"label": "real"}]

    result = failure_as_error_metrics(scored, threshold=0.5, missing_rows=missing)

    assert result["n_attempted"] == 4
    assert result["n_scored"] == 2
    assert result["n_failures_imputed_as_errors"] == 2
    assert result["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert "auroc" not in result
    assert "auprc" not in result


def test_public_row_strips_private_fields_and_hashes_identity():
    private = {
        "track_id": "private-track-name",
        "label": "ai",
        "source": "aime_musicgen_large",
        "generator": "MusicGen Large",
        "generator_family": "MusicGen",
        "generator_version": "large",
        "bench_origin": "test",
        "protocol_split": "test",
        "audio_sha256": "a" * 64,
        "audio_bytes": 1234,
        "audio_format": "wav",
        "lineage_id": "private-lineage",
        "runtime_path": "/private/audio.wav",
        "title": "private title",
    }

    public = public_row(private)

    assert public["track_id"] == "ab2-" + "a" * 20
    assert public["lineage_id"].startswith("ab2l-")
    assert public["legacy_track_id_sha256"] != private["track_id"]
    assert public["retrieval"]["upstream"] == "disco-eth/AIME"
    assert "runtime_path" not in public
    assert "title" not in public


@pytest.mark.parametrize(
    "value",
    [
        {"runtime_path": "relative.wav"},
        {"error": "failed at /Users/researcher/private.wav"},
        {"error": "contact researcher@example.org"},
        {"nested": [{"artist": "private creator"}]},
    ],
)
def test_public_safety_rejects_private_fields_and_paths(value):
    with pytest.raises(ValueError):
        assert_public_safe(value)


def test_public_safety_allows_model_filenames_and_repository_ids():
    assert_public_safe(
        {
            "hf_filename": "artifactnet_v94_full.onnx",
            "mert_repo": "m-a-p/MERT-v1-95M",
            "score_file": "raw/artifactnet/track_probs.json",
        }
    )


def test_manifest_audit_reads_v2_top_level_counts(tmp_path):
    manifest = {
        "schema_version": 2,
        "total": 2,
        "labels": {"ai": 1, "real": 1},
        "sources": {"synthetic": 1, "real": 1},
        "tracks": [
            {
                "track_id": "ab2-a",
                "label": "ai",
                "source": "synthetic",
                "bench_origin": "test",
                "audio_sha256": "a" * 64,
                "lineage_id": "ab2l-a",
            },
            {
                "track_id": "ab2-b",
                "label": "real",
                "source": "real",
                "bench_origin": "test",
                "audio_sha256": "b" * 64,
                "lineage_id": "ab2l-b",
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    result = audit(path)

    assert result["comparisons"]["total"]["status"] == "match"
    assert result["comparisons"]["ai"]["status"] == "match"
    assert result["comparisons"]["real"]["status"] == "match"
    assert result["comparisons"]["n_sources"]["status"] == "match"
    assert result["release_ready"] is True
