"""읽기 전용 코퍼스 목록에서 점수와 무관한 메타데이터 조사 후보를 고정한다."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .common import digest, manifest_rows, rank, write_once


NATIVE_INPUTS = (
    "native_cdn_260831/manifest.current.json",
    "native_cdn_260831/manifest_delta_260831.json",
)
EXPOSURE_INPUTS = (
    "bigset_260720/train_manifest.json",
    "native_cdn_train_260831/train_manifest_final.json",
    "native_cdn_train_260831_v2/train_manifest_v3.json",
    "native_cdn_train_260831_v4/train_manifest_v4.json",
    "s0_manifest_baseline_260904.json",
)
SEALED_INPUT = "native_cdn_saturation_260831/sealed_creator_groups_260831.json"


def exposure_index(documents, selection_groups=()):
    tracks, creators = defaultdict(set), defaultdict(set)
    for name, doc in documents.items():
        for split, row in manifest_rows(doc):
            reason = f"{name}:{split}"
            if row.get("track_id"):
                tracks[row["track_id"]].add(reason)
            if row.get("creator_group"):
                creators[row["creator_group"].casefold()].add(reason)
    for group in selection_groups:
        creators[group.casefold()].add("native_sealed_selection_pool")
    return tracks, creators


def survey_sample(rows, limit, creator_cap=2, seed=260905):
    """곡 단위 순위를 고정하고 동일 곡의 모든 전송 변형을 함께 보존한다."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["track_id"]].append(row)
    selected, counts = [], Counter()
    for track_id in sorted(grouped, key=lambda key: rank(key, seed)):
        variants = grouped[track_id]
        groups = {r["creator_group"].casefold() for r in variants}
        if len(groups) != 1:
            raise ValueError(f"Conflicting creator groups: {track_id}")
        group = groups.pop()
        if counts[group] >= creator_cap:
            continue
        selected.append({
            "track_id": track_id, "media_id": variants[0]["media_id"],
            "provider": track_id.split(":", 1)[0],
            "creator_group": group,
            "creator_evidence": "song_fallback" if ":song:" in group else "crawl_metadata",
            "variants": sorted(variants, key=lambda r: r["variant_id"]),
        })
        counts[group] += 1
        if len(selected) >= limit:
            break
    return selected


def build(root, output, suno_limit=1200, udio_limit=300):
    paths = [root / name for name in (*NATIVE_INPUTS, *EXPOSURE_INPUTS, SEALED_INPUT)]
    docs = {str(p.relative_to(root)): json.loads(p.read_text()) for p in paths}
    track_exposure, group_exposure = exposure_index(
        {name: docs[name] for name in EXPOSURE_INPUTS}, docs[SEALED_INPUT])
    native = {}
    for name in NATIVE_INPUTS:
        for split, row in manifest_rows(docs[name]):
            item = dict(row, original_manifest=name, original_split=split)
            if row["path"] in native:
                old = native[row["path"]]
                for key in ("track_id", "variant_id", "creator_group", "source_url"):
                    if item[key] != old[key]:
                        raise ValueError(f"Conflicting input {key}: {row['path']}")
                continue
            native[row["path"]] = item
    audit, eligible = [], defaultdict(list)
    for row in native.values():
        reasons = sorted(track_exposure.get(row["track_id"], set()) |
                         group_exposure.get(row["creator_group"].casefold(), set()))
        audit.append({"track_id": row["track_id"], "variant_id": row["variant_id"],
                      "creator_group": row["creator_group"], "exclusion_evidence": reasons})
        if not reasons:
            eligible[row["track_id"].split(":", 1)[0]].append(row)
    selected = []
    for provider, limit in (("suno", suno_limit), ("udio", udio_limit)):
        selected.extend(survey_sample(eligible[provider], limit))
    input_hashes = {str(p): digest(p) for p in paths}
    summary = {
        "schema_version": "artifactbench-1.2-survey-1",
        "input_hashes": input_hashes,
        "tool_sha256": digest(__file__),
        "common_tool_sha256": digest(Path(__file__).with_name("common.py")),
        "seed": 260905, "provisional_creator_cap": 2,
        "requested_survey_tracks": {"suno": suno_limit, "udio": udio_limit},
        "native_variants": len(native),
        "native_tracks": len({r["track_id"] for r in native.values()}),
        "eligible_tracks": {p: len({r['track_id'] for r in rr}) for p, rr in eligible.items()},
        "survey_tracks": dict(Counter(r["provider"] for r in selected)),
        "known_exposure_groups": sorted(group_exposure),
        "known_exposure_tracks": sorted(track_exposure),
        "scope": "Metadata survey, not a frozen scored benchmark. Native corpus was previously mined; unknown baseline exposure is not clean exposure.",
    }
    write_once(output / "inventory.json", summary)
    write_once(output / "exposure_audit.json", audit)
    write_once(output / "survey_candidates.local.json", selected)
    print(json.dumps({k: summary[k] for k in ("native_variants", "native_tracks", "eligible_tracks", "survey_tracks")}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifactnet-outputs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--suno-limit", type=int, default=1200)
    ap.add_argument("--udio-limit", type=int, default=300)
    args = ap.parse_args()
    if min(args.suno_limit, args.udio_limit) < 1:
        ap.error("Survey limits must be positive")
    build(args.artifactnet_outputs, args.output, args.suno_limit, args.udio_limit)


if __name__ == "__main__":
    main()
