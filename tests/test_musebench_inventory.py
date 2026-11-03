# Created: 2026-11-04
# Purpose: Cover scripts/musebench/inventory.py grouping and Markdown rendering.
# Dependencies: pytest
# Test Status: passing
from scripts.musebench import inventory as inv


def rec(path, size, shard="part1_001.tar", typ="file"):
    return {"path": path, "size": size, "shard": shard, "type": typ}


def test_summarize_groups_by_depth_and_skips_non_files():
    records = [
        rec("Jamendo/a.mp3", 10),
        rec("Jamendo/b.mp3", 20, shard="part1_002.tar"),
        rec("Musicgen_Generated_Music/Jamedo/a.mp3.wav", 5),
        rec("Musicgen_Generated_Music/Genius-Lyrics/x.mp3.wav", 7),
        rec("aac_generated_music/udio_generated_music/fake_1_udio_0.m4a", 3),
        rec("Jamendo", 0, typ="dir"),
    ]
    out = inv.summarize(records, depth=2)
    assert out["Jamendo"] == {"files": 2, "bytes": 30, "ext": {"mp3": 2},
                              "shards": ["part1_001.tar", "part1_002.tar"]}
    assert out["Musicgen_Generated_Music/Jamedo"]["ext"] == {"wav": 1}
    assert "aac_generated_music/udio_generated_music" in out
    assert sum(g["files"] for g in out.values()) == 5


def test_markdown_has_total_row():
    md = inv.to_markdown(inv.summarize([rec("OOD FMA/1.mp3", 2_000_000_000)]))
    assert "| `OOD FMA` | 1 | 2.0 | mp3 1 | 1 |" in md
    assert md.splitlines()[-1].startswith("| **Total** | **1** | **2.0**")
