"""공개 baseline 자산을 검증된 HF 리비전에 고정하여 전용 캐시에 받는다."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .common import digest, write_once


# 2026-09-05 HfApi.model_info로 조회한 실제 리비전. main으로 대체하지 않는다.
REPOS = {
    "awsaf49/sonics-spectttra-alpha-120s": "094b32a5545098a71c113f6ae9d5c55310564268",
    "awsaf49/sonics-spectttra-beta-5s": "fb311b279eabd9b80918874f085e3c1ad7e46f4a",
    "m-a-p/MERT-v1-95M": "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
    "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres": "caf978c8328a2cec4229b3eb0b41b162e379caa1",
    "AI-Music-Detection/ai_music_detection_large_60s": "5e29461bbfc41dc5fe4ed5f7fadc961aa84a7e3e",
    "MIT/ast-finetuned-audioset-10-10-0.4593": "f826b80d28226b62986cc218e5cec390b1096902",
    "lofcz/ai-music-detector": "d2180598fed79e3f917e8050a00439982466e5c6",
    "DeepFense/FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42": "c8ec432e71e94babaa83e0ef681c24bfa540ada7",
    "worstchan/EAT-large_epoch20_pretrain": "1109aaae544915a2b82182c643c1894b1b4d9f27",
}


def fetch_one(repo, revision, cache):
    patterns = ["README.md", "*.json", "*.py", "pytorch_model.bin", "model.safetensors", "*.onnx", "best_model.pth", "config.yaml"]
    if repo.startswith("MIT/"):
        # AST baseline은 이 모델의 feature extractor 설정만 사용한다.
        patterns = ["README.md", "config.json", "preprocessor_config.json"]
    path = Path(snapshot_download(repo_id=repo, revision=revision, cache_dir=cache,
                                  allow_patterns=patterns, token=False, max_workers=2))
    files = {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError(f"Empty model snapshot: {repo}")
    return {"repo": repo, "revision": revision, "snapshot": str(path.resolve()), "files": files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    write_once(args.output / "fetch_inputs.json", {"repos": REPOS, "tool_sha256": digest(__file__)})
    rows = []
    # 동시 snapshot_download가 tqdm의 전역 lock을 제거하는 경쟁을 피한다.
    # 한 저장소 내부의 파일 다운로드는 max_workers=2로 병렬성을 유지한다.
    for repo, revision in REPOS.items():
        row = fetch_one(repo, revision, args.cache)
        write_once(args.output / (row["repo"].replace("/", "--") + ".json"), row)
        rows.append(row)
        print(json.dumps({"repo": row["repo"], "revision": row["revision"], "files": len(row["files"])}), flush=True)
    write_once(args.output / "snapshots.json", sorted(rows, key=lambda r: r["repo"]))


if __name__ == "__main__":
    main()
