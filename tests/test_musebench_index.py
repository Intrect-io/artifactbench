# Created: 2026-11-04
# Purpose: Cover scripts/musebench/index_shards.py (tar header walk over a range reader)
#          on synthetic GNU, PAX and USTAR archives, plus resume after an interrupted walk.
# Dependencies: pytest
# Test Status: passing
import io
import json
import tarfile

import pytest

from scripts.musebench import index_shards as ix


def make_tar(members, fmt):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tf:
        for name, data in members:
            if data is None:
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
                continue
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 1_700_000_000
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def reader(blob, calls=None):
    def fetch(offset, length):
        if calls is not None:
            calls.append((offset, length))
        return blob[offset:offset + length]
    return fetch


def walk_all(blob, start=0):
    gen = ix.walk_tar(reader(blob), start)
    recs = []
    while True:
        try:
            recs.append(next(gen))
        except StopIteration as stop:
            return recs, stop.value


def check_against_tarfile(blob, recs):
    """Every record must locate exactly the bytes tarfile reads for that member."""
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        expected = [(m.name, m.size, m.offset_data) for m in tf.getmembers() if m.isfile()]
    got = [(r["path"], r["size"], r["data_offset"]) for r in recs if r["type"] == "file"]
    assert got == expected


LONG = "MagNet_Generated_Music/Genius-Lyrics/" + "x" * 150 + ".mp3.wav"
UNICODE = "Jamendo/Часть 1_4 - Ilyin_Books & ‘quote’.mp3"
PREFIXED = "ai_human_co-created_music/Jamendo/" + "d" * 60 + "/" + "f" * 90 + ".wav"


@pytest.mark.parametrize("fmt", [tarfile.GNU_FORMAT, tarfile.PAX_FORMAT, tarfile.USTAR_FORMAT])
def test_walk_matches_tarfile(fmt):
    members = [
        ("Caption/captions-genius.csv", b"a,b\n" * 300),
        ("Jamendo", None),
        ("Jamendo/empty.mp3", b""),
        ("Jamendo/exact.mp3", b"z" * 1024),
        (PREFIXED, b"p" * 700),
    ]
    if fmt != tarfile.USTAR_FORMAT:  # USTAR cannot store these names
        members += [(LONG, b"l" * 513), (UNICODE, b"u" * 5)]
    blob = make_tar(members, fmt)
    recs, end = walk_all(blob)
    check_against_tarfile(blob, recs)
    names = [r["path"] for r in recs]
    assert PREFIXED in names
    if fmt != tarfile.USTAR_FORMAT:
        assert LONG in names and UNICODE in names
    assert recs[0]["offset"] == 0
    for prev, cur in zip(recs, recs[1:]):
        assert cur["offset"] == ix.record_end(prev)  # contiguous, extensions included
    assert end == ix.record_end(recs[-1])
    assert blob[end:end + 1024] == b"\0" * 1024


def test_gnu_long_name_record_starts_at_extension_header():
    blob = make_tar([("a.wav", b"1"), (LONG, b"2")], tarfile.GNU_FORMAT)
    recs, _ = walk_all(blob)
    long_rec = recs[1]
    assert long_rec["path"] == LONG
    # The record offset is the ././@LongLink header, one header + one data block earlier.
    assert long_rec["data_offset"] - long_rec["offset"] == 3 * ix.BLOCK


def test_gnu_header_tail_is_not_read_as_prefix():
    # GNU headers store atime/ctime where POSIX keeps the prefix; it must be ignored.
    blob = bytearray(make_tar([("short.wav", b"abc")], tarfile.GNU_FORMAT))
    assert bytes(blob[257:263]) == b"ustar "
    blob[345:357] = b"14672366670\0"  # fake atime
    blob[148:156] = b"        "
    chk = sum(blob[:512])
    blob[148:156] = b"%06o\0 " % chk
    recs, _ = walk_all(bytes(blob))
    assert recs[0]["path"] == "short.wav"


def test_checksum_mismatch_raises():
    blob = bytearray(make_tar([("a.wav", b"1")], tarfile.GNU_FORMAT))
    blob[0] = ord("b")
    with pytest.raises(ix.TarFormatError, match="checksum"):
        walk_all(bytes(blob))


def test_truncated_archive_raises():
    blob = make_tar([("a.wav", b"1" * 2000)], tarfile.GNU_FORMAT)
    with pytest.raises(ix.TarFormatError, match="truncated"):
        walk_all(blob[:512 + 2048])


def test_parse_number_octal_and_base256():
    assert ix.parse_number(b"0000644\0") == 0o644
    assert ix.parse_number(b"   12 \0") == 0o12
    assert ix.parse_number(b"\0" * 8) == 0
    big = 20 * 1024 ** 3
    field = bytes([0x80]) + big.to_bytes(11, "big")
    assert ix.parse_number(field) == big


def pax_record(key, value):
    body = b" " + key + b"=" + value + b"\n"
    n = len(body) + 1
    while len(str(n)) + len(body) != n:
        n += 1
    return str(n).encode() + body


def test_parse_pax_records():
    data = pax_record(b"mtime", b"1700000000.5") + pax_record(b"path", "Jamendo/Ч.mp3".encode())
    out = ix.parse_pax(data + b"\0" * 7)
    assert out["path"] == "Jamendo/Ч.mp3".encode()
    assert out["mtime"] == b"1700000000.5"
    with pytest.raises(ix.TarFormatError):
        ix.parse_pax(b"99 path=x\n")


def test_one_fetch_per_plain_member():
    members = [(f"Jamendo/{i}.mp3", b"m" * 5000) for i in range(5)]
    blob = make_tar(members, tarfile.GNU_FORMAT)
    calls = []
    gen = ix.walk_tar(reader(blob, calls))
    assert len(list(gen)) == 5
    assert len(calls) == 6  # five headers + end marker


def shard_for(blob, stem="part9_001"):
    return {"part": 9, "repo": "r", "revision": "abc", "file": f"{stem}.tar",
            "size": len(blob), "lfs_sha256": None}


def test_index_shard_writes_jsonl_and_done(tmp_path):
    blob = make_tar([(f"suno_generated_music/fake_{i:05d}_suno_0.mp3", b"s" * 900)
                     for i in range(4)], tarfile.GNU_FORMAT)
    info = ix.index_shard(shard_for(blob), tmp_path, reader(blob))
    lines = (tmp_path / "part9_001.jsonl").read_text().splitlines()
    assert info["members"] == len(lines) == 4
    done = json.loads((tmp_path / "part9_001.done").read_text())
    assert done["end_offset"] == ix.record_end(json.loads(lines[-1]))
    assert all(json.loads(l)["shard"] == "part9_001.tar" for l in lines)
    # Idempotent: a second call does not touch the network.
    assert ix.index_shard(shard_for(blob), tmp_path, lambda o, n: 1 / 0) == done


def test_resume_after_interruption_drops_torn_line(tmp_path):
    members = [(f"udio_generated_music/fake_{i:05d}_udio_1.mp3", b"u" * (300 + i)) for i in range(6)]
    blob = make_tar(members, tarfile.GNU_FORMAT)
    shard = shard_for(blob)
    fetch = reader(blob)

    class Boom(Exception):
        pass

    n = [0]

    def flaky(offset, length):
        n[0] += 1
        if n[0] == 4:
            raise Boom
        return fetch(offset, length)

    with pytest.raises(Boom):
        ix.index_shard(shard, tmp_path, flaky)
    jsonl = tmp_path / "part9_001.jsonl"
    with open(jsonl, "a") as f:
        f.write('{"torn": ')  # simulate a crash mid-write
    assert not (tmp_path / "part9_001.done").exists()
    ix.index_shard(shard, tmp_path, fetch)
    recs = [json.loads(l) for l in jsonl.read_text().splitlines()]
    assert [r["path"] for r in recs] == [m[0] for m in members]
    check_against_tarfile(blob, recs)


def test_end_offset_beyond_listed_size_is_rejected(tmp_path):
    blob = make_tar([("a.wav", b"1")], tarfile.GNU_FORMAT)
    shard = shard_for(blob)
    shard["size"] = 100
    with pytest.raises(ix.TarFormatError, match="beyond"):
        ix.index_shard(shard, tmp_path, reader(blob))
