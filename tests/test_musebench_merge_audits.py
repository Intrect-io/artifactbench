# Created: 2026-11-04
# Purpose: Cover scripts/musebench/merge_audits.py (sharding and lossless merge of audits).
# Dependencies: pytest
# Test Status: passing
import pytest

from scripts.musebench import merge_audits as ma


def failures(n):
    return [{"track_id": f"t{i}", "stage": "inference"} for i in range(n)] + \
        [{"track_id": "dec", "stage": "decode"}]


def audit_of(ids, finite=5):
    return {"tracks": [{"track_id": t, "valid_chunks": finite, "total_chunks": 7,
                        "chunks": [{"finite": j < finite, "rms_dbfs": -90.0 - j} for j in range(7)]}
                       for t in ids]}


def test_split_round_robin_skips_non_inference():
    parts = ma.split(failures(10), 3)
    assert [len(p) for p in parts] == [4, 3, 3]
    assert sorted(f["track_id"] for p in parts for f in p) == sorted(f"t{i}" for i in range(10))


def test_merge_restores_order_and_recomputes_summary():
    fl = failures(6)
    parts = ma.split(fl, 2)
    audits = [audit_of([f["track_id"] for f in parts[0]], finite=5),
              audit_of([f["track_id"] for f in parts[1]], finite=2)]
    m = ma.merge(fl, audits)
    assert [t["track_id"] for t in m["tracks"]] == [f"t{i}" for i in range(6)]
    assert m["tracks_audited"] == 6 and m["chunks_audited"] == 42
    assert m["nonfinite_chunks"] == 3 * 2 + 3 * 5
    assert m["tracks_with_at_least_four_finite_chunks"] == 3
    assert m["nonfinite_rms_dbfs"]["max"] == -92.0 and m["merged_from_shards"] == 2


def test_merge_rejects_missing_duplicate_or_extra_tracks():
    fl = failures(3)
    with pytest.raises(ValueError, match="not audited"):
        ma.merge(fl, [audit_of(["t0", "t1"])])
    with pytest.raises(ValueError, match="twice"):
        ma.merge(fl, [audit_of(["t0", "t1", "t2"]), audit_of(["t0"])])
    with pytest.raises(ValueError, match="not inference failures"):
        ma.merge(fl, [audit_of(["t0", "t1", "t2", "zz"])])
