"""사전 선언한 규칙으로 v1.2 검증 대상 목록을 만든다. 검출 점수는 읽지 않는다."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .common import digest, rank, write_once
from .correct_native_identity import choose_udio_variant


def creator_alias(provider, metadata):
    if provider == "suno":
        handle = metadata.get("creator_handle")
        return "suno:@" + handle.casefold().lstrip("@") if handle else None
    artist = " ".join((metadata.get("artist") or "").casefold().split())
    return "udio:artist:" + artist if artist else None


def select_native(candidates, metadata, exposed_groups, quota=200):
    # 같은 user ID로 결합된 별칭 중 하나라도 노출된 경우 전체 제작자를 제외한다.
    exposed_ids = set()
    for candidate in candidates:
        m = metadata[candidate["track_id"]]
        alias = creator_alias(candidate["provider"], m)
        if alias in exposed_groups and m.get("creator_id"):
            exposed_ids.add((candidate["provider"], m["creator_id"]))
    counts, creator_counts = Counter(), Counter()
    selected, audit = [], []
    for candidate in sorted(candidates, key=lambda r: rank(r["track_id"])):
        provider = candidate["provider"]
        m = metadata[candidate["track_id"]]
        reason = None
        creator = m.get("creator_id")
        alias = creator_alias(provider, m)
        if m["outcome"] == "error":
            reason = "metadata_error"
        elif provider == "suno" and m.get("generator_version") != "v5.5":
            reason = "not_verified_suno_v5.5"
        elif provider == "udio" and (m.get("created_at") or "") < "2026-01-01":
            reason = "udio_created_before_2026_or_unknown"
        elif not creator or not alias:
            reason = "creator_identity_missing"
        elif alias in exposed_groups or (provider, creator) in exposed_ids:
            reason = "enriched_creator_exposure"
        elif creator_counts[(provider, creator)] >= 2:
            reason = "enriched_creator_cap"
        elif counts[provider] >= quota:
            reason = "provider_quota"
        audit.append({"track_id": candidate["track_id"], "creator_alias": alias,
                      "creator_id": creator, "reason": reason or "selected_for_validation"})
        if reason:
            continue
        variants = sorted(candidate["variants"], key=lambda r: (r["container"] != "mp3", r["variant_id"]))
        if provider == 'udio':
            matched, _ = choose_udio_variant(variants, m)
            variants = [variant for variant, _ in matched]
        primary = variants[0]
        entry = dict(primary)
        entry.update(
            source="suno_v5.5_native_260905" if provider == "suno" else "udio_2026_version_unknown_native_260905",
            partition="contemporary_native", provider=provider,
            generator_version=m.get("generator_version"), generator=provider,
            created_at=m.get("created_at"), creator_group=f"{provider}:user:{creator}",
            creator_alias=alias, label_scope=m["label_scope"],
            generation_task=m.get("generation_task"), generation_type=m.get("generation_type"),
            audio_conditioning_type=m.get("audio_conditioning_type"),
            lineage=m.get("lineage"), song_id=m.get("song_id"),
            metadata_source_url=m["source_url"], metadata_page_sha256=m["page_sha256"],
            metadata_retrieved_at=m["retrieved_at"],
            version_evidence=m.get("version_evidence") or m.get("version_field"),
            additional_native_variants=variants[1:],
            exposure_scope="Known local train/selection creator exclusion; retrospectively mined source pool; external checkpoint exposure unknown",
            rights="metadata_only; audio redistribution permission not established",
        )
        selected.append(entry)
        counts[provider] += 1
        creator_counts[(provider, creator)] += 1
    return selected, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--suno-metadata", required=True, type=Path)
    ap.add_argument("--udio-metadata", required=True, type=Path)
    ap.add_argument("--legacy", required=True, type=Path)
    ap.add_argument("--demos", required=True, type=Path)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    inventory = json.loads((args.inventory / "inventory.json").read_text())
    candidates_path = args.inventory / "survey_candidates.local.json"
    candidates = json.loads(candidates_path.read_text())
    metadata, metadata_hashes = {}, {}
    for provider, directory in (("suno", args.suno_metadata), ("udio", args.udio_metadata)):
        summary = json.loads((directory / "summary.json").read_text())
        expected = [r for r in candidates if r["provider"] == provider]
        if summary["attempted"] != len(expected):
            raise ValueError(f"Incomplete metadata survey: {provider}")
        for candidate in expected:
            path = directory / "metadata" / f"{candidate['media_id']}.json"
            m = json.loads(path.read_text())
            if m["track_id"] != candidate["track_id"]:
                raise ValueError(f"Metadata ID mismatch: {path}")
            if m.get("page_sha256") and digest(directory / m["page_file"]) != m["page_sha256"]:
                raise ValueError(f"Metadata page changed: {path}")
            metadata[m["track_id"]] = m
            metadata_hashes[str(path)] = digest(path)
    selected, audit = select_native(candidates, metadata, set(inventory["known_exposure_groups"]))
    legacy_manifest = args.legacy / "manifest.json"
    legacy_provenance = args.legacy / "provenance.json"
    legacy = json.loads(legacy_manifest.read_text())["bench"]
    recovery = json.loads(legacy_provenance.read_text())["recovered_audio_paths"]
    for entry in legacy:
        old_path = entry["path"]
        entry.update(path=recovery.get(old_path, old_path), original_path=old_path,
                     partition="legacy", creator_group="recording:" + entry["source"] + ":" + entry["track_id"],
                     creator_evidence="unknown; recording fallback", label_scope="legacy dataset label",
                     exposure_scope="retrospective; model-specific exposure audit required")
    demos = json.loads(args.demos.read_text())
    entries = legacy + selected + demos
    variant_keys = [(r["source"], r["track_id"]) for r in entries]
    if len(variant_keys) != len(set(variant_keys)):
        raise ValueError("Duplicate primary evaluation IDs")
    input_paths = [args.inventory / "inventory.json", candidates_path, legacy_manifest,
                   legacy_provenance, args.demos, args.protocol, Path(__file__)]
    provenance = {
        "status": "selection fixed before scoring; full validation/duplicate audit pending",
        "schema_version": "artifactbench-1.2-selection-1", "seed": 260905,
        "input_hashes": {str(p): digest(p) for p in input_paths},
        "metadata_hashes": metadata_hashes,
        "partitions": dict(Counter(r["partition"] for r in entries)),
        "source_counts": dict(Counter(r["source"] for r in entries)),
        "native_creator_counts": dict(Counter(provider for provider, _ in {
            (r["provider"], r["creator_group"]) for r in selected})),
        "native_exclusion_counts": dict(Counter(r["reason"] for r in audit)),
        "native_selected": dict(Counter(r["provider"] for r in selected)),
    }
    write_once(args.output / "selection_provenance.json", provenance)
    write_once(args.output / "native_selection_audit.json", audit)
    write_once(args.output / "selection.local.json", {"bench": entries})
    print(json.dumps({k: provenance[k] for k in ("partitions", "native_selected", "native_creator_counts", "native_exclusion_counts")}, indent=2))


if __name__ == "__main__":
    main()
