"""검출 점수 없이 메타데이터 정합성과 후보 추출을 검사한다."""
import json

import pytest

from artifactbench.v12.inventory import exposure_index, survey_sample
from artifactbench.v12.suno_metadata import extract_suno
from artifactbench.v12.udio_metadata import extract_udio
from artifactbench.v12.official_demos import demo_urls
from artifactbench.v12.prepare_selection import select_native


MID = "6e3d6379-e1f1-449a-bc80-99f0cb88e086"
OTHER = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"


def flight(value, split=False):
    data = "4f:" + json.dumps(value) + "\n"
    chunks = [data[:len(data)//2], data[len(data)//2:]] if split else [data]
    return "".join("<script>self.__next_f.push(" + json.dumps([1, chunk]) + ")</script>" for chunk in chunks)


def clip(mid=MID, version="v5.5"):
    return {"id": mid, "major_model_version": version, "model_name": "chirp-flounder",
            "metadata": {"task": "upsample", "is_remix": True}, "handle": "artist"}


def test_requested_id_not_recommended_song():
    page = flight({"recommended": clip(OTHER, "v4"), "clip": clip()})
    result = extract_suno(page, MID)
    assert result["generator_version"] == "v5.5"
    assert result["version_field"] == "major_model_version"
    assert result["generation_task"] == "upsample"
    assert result["is_remix"] is True


def test_flight_across_scripts():
    assert extract_suno(flight({"clip": clip()}, split=True), MID)["generator_version"] == "v5.5"


def test_length_delimited_lyrics_then_json_without_newline():
    lyrics = '한글 가사\n41:' + json.dumps(clip(version="v4"))
    data = ':HL["/style.css","style"]\n' + f"52:T{len(lyrics.encode('utf-8')):x}," + lyrics
    data += '41:' + json.dumps({"clip": clip()}) + '\n'
    page = '<script>self.__next_f.push(' + json.dumps([1, data]) + ')</script>'
    assert extract_suno(page, MID)["generator_version"] == "v5.5"


def test_truncated_flight_text_is_explicit_error():
    page = '<script>self.__next_f.push(' + json.dumps([1, '52:Tff,short']) + ')</script>'
    with pytest.raises(ValueError, match="Truncated"):
        extract_suno(page, MID)


def test_recommendation_only_is_not_evidence():
    with pytest.raises(ValueError, match="requested ID"):
        extract_suno(flight({"clip": clip(OTHER)}), MID)


def test_conflicting_version_fails():
    with pytest.raises(ValueError, match="Conflicting clip"):
        extract_suno(flight([clip(), clip(version="v4")]), MID)


def test_badge_conflict_fails():
    obj = clip()
    obj["metadata"]["model_badges"] = {"songcard": {"display_name": "v4"}}
    with pytest.raises(ValueError, match="Conflicting version"):
        extract_suno(flight(obj), MID)


def test_plain_json_and_unknown_version():
    obj = clip()
    del obj["major_model_version"]
    result = extract_suno('<script type="application/json">'+json.dumps(obj)+"</script>", MID)
    assert result["generator_version"] is None


def test_all_training_splits_are_exposure():
    docs = {"model": {split: [{"track_id": split, "creator_group": "Suno:@Artist"}]
                      for split in ("train", "val", "test")}}
    tracks, groups = exposure_index(docs, ["suno:@selected"])
    assert set(tracks) == {"train", "val", "test"}
    assert len(groups["suno:@artist"]) == 3
    assert groups["suno:@selected"] == {"native_sealed_selection_pool"}


def test_survey_is_order_invariant_and_keeps_variants():
    rows = [{"track_id": f"udio:{i}", "media_id": str(i), "variant_id": v,
             "creator_group": f"udio:artist:{i//4}"} for i in range(12) for v in ("mp3", "mp4")]
    first = survey_sample(rows, 50, creator_cap=2)
    assert first == survey_sample(list(reversed(rows)), 50, creator_cap=2)
    assert len(first) == 6
    assert all(len(row["variants"]) == 2 for row in first)


def test_conflicting_variant_creator_fails():
    rows = [{"track_id": "udio:1", "media_id": "1", "variant_id": str(i),
             "creator_group": f"udio:{i}"} for i in range(2)]
    with pytest.raises(ValueError, match="Conflicting creator"):
        survey_sample(rows, 1)


def test_udio_matches_cdn_id_not_page_recommendation():
    mid = "8b5339ed7ba74d35a1d3467a20ac4a77"
    song = {"id": "different-song-uuid", "generation_id": "generation",
            "song_path": f"https://storage.googleapis.com/bucket/samples/{mid}/2/song.mp3",
            "user_id": "creator-uuid", "artist": "Artist"}
    result = extract_udio(flight({"song": song, "suggestion": clip()}), mid)
    assert result["song_id"] == "different-song-uuid"
    assert result["creator_id"] == "creator-uuid"
    assert result["generator_version"] is None


def test_udio_id_substring_is_not_a_match():
    mid = "8b5339ed7ba74d35a1d3467a20ac4a77"
    song = {"generation_id": "generation", "song_path": f"https://example.com/{mid}extra/x.mp3"}
    with pytest.raises(ValueError, match="requested media ID"):
        extract_udio(flight(song), mid)


def test_demo_heading_can_contain_formatting():
    page = '<h1><span><strong>Stable Audio 3.0</strong></span></h1><div data-media-src="/s/demo.mp3"></div>'
    assert demo_urls("stable_audio", page) == ["https://stability.ai/s/demo.mp3"]


def test_demo_version_outside_heading_is_not_evidence():
    page = '<h1>Stable Audio 2.0</h1><p>Stable Audio 3.0</p><div data-media-src="/s/demo.mp3"></div>'
    with pytest.raises(ValueError, match="version heading"):
        demo_urls("stable_audio", page)


def test_lyria_light_dark_duplicates_do_not_inflate_recordings():
    url = 'https://storage.googleapis.com/demo/lyria-3-5__track__test.webm#t=0.1'
    page = '<section id=lyria-35><h2>Introducing Lyria 3.5</h2><article>'
    page += f'<source src="{url}"><source data-src="{url}"></article></section><hr>'
    assert len(demo_urls("lyria", page)) == 1


def test_enriched_creator_alias_excludes_same_id_other_handle():
    candidates = [{"track_id": f"suno:{i}", "provider": "suno"} for i in (1, 2)]
    metadata = {f"suno:{i}": {"outcome": "verified", "generator_version": "v5.5",
                              "creator_id": "same-creator", "creator_handle": handle}
                for i, handle in ((1, "old_handle"), (2, "new_handle"))}
    selected, audit = select_native(candidates, metadata, {"suno:@old_handle"})
    assert not selected
    assert all(r["reason"] == "enriched_creator_exposure" for r in audit)


def test_udio_old_creation_date_is_excluded_before_quota():
    candidate = {"track_id": "udio:1", "provider": "udio"}
    metadata = {"udio:1": {"outcome": "unknown_version", "created_at": "2025-12-31T23:59:00Z"}}
    selected, audit = select_native([candidate], metadata, set())
    assert not selected
    assert audit[0]["reason"] == "udio_created_before_2026_or_unknown"
