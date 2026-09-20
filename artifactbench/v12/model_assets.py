"""동결된 로컬 snapshot만 사용하는 모델 구성과 실제 코드/가중치 출처."""
import json
import os
from pathlib import Path
import subprocess

from .common import digest


def asset_location(variable, default):
    """경로만 기계별로 바꾸며, 실제 구현과 가중치의 해시 검사는 유지한다."""
    return Path(os.environ.get(variable, default)).expanduser().resolve()


def source_identity(path):
    path = Path(path)
    return {'root': str(path), 'git_head': subprocess.check_output(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip(),
        'python_files': {str(p): digest(p) for p in sorted(path.rglob('*.py'))
                         if not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(path).parts)}}


def build_pinned(name, snapshots_path):
    from artifactbench.models import (ArtifactNetModel, SpecTTTraModel, SpecTTTraVariantModel,
        CLAMModel, DeezerISMIRModel, FSTModel, ASTMusicDetectionModel, DeepFenseModel)

    snapshots = {r['repo']: r for r in json.loads(Path(snapshots_path).read_text())}
    evidence = {'snapshots_file_sha256': digest(snapshots_path), 'snapshots': {}, 'files': {}, 'source_repos': {}}
    def snapshot(repo):
        row = snapshots[repo]
        directory = Path(row['snapshot'])
        if directory.name != row['revision']:
            raise ValueError(f'Snapshot revision mismatch: {repo}')
        for filename, expected in row['files'].items():
            if digest(directory / filename) != expected:
                raise ValueError(f'Model asset changed: {repo}/{filename}')
        evidence['snapshots'][repo] = row
        return str(directory)
    def local(path):
        path = Path(path).resolve()
        evidence['files'][str(path)] = digest(path)
        return str(path)
    if name == 'artifactnet':
        import sys
        root = asset_location('ARTIFACTBENCH_ARTIFACTNET_REPO', '/home/unohee/dev/ArtifactNet')
        sys.path.insert(0, str(root))
        from src.pipeline import model_config as cfg
        import src.pipeline.infer as infer
        infer.GPU_BATCH_MAX = 1
        for path in (cfg.DEFAULT_CNN, cfg.DEFAULT_UNET, cfg.DEFAULT_RESCUE_LGBM):
            local(root / path)
        evidence['source_repos']['ArtifactNet'] = source_identity(root / 'src')
        evidence['operating_thresholds'] = {'cnn': cfg.DEFAULT_THRESHOLD, 'rescue': cfg.DEFAULT_RESCUE_TAU}
        evidence['rescue_checkpoint'] = str(root / cfg.DEFAULT_RESCUE_LGBM)
        model = ArtifactNetModel(repo_dir=str(root))
    elif name == 'spectttra':
        model = SpecTTTraModel(hf_repo=snapshot('awsaf49/sonics-spectttra-alpha-120s'))
    elif name == 'spectttra_beta5s':
        model = SpecTTTraVariantModel(hf_repo=snapshot('awsaf49/sonics-spectttra-beta-5s'))
    elif name == 'ast_60s':
        model = ASTMusicDetectionModel(hf_repo=snapshot('AI-Music-Detection/ai_music_detection_large_60s'),
                    extractor_repo=snapshot('MIT/ast-finetuned-audioset-10-10-0.4593'))
    elif name == 'deepfense':
        root = Path(snapshot('DeepFense/FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42'))
        model = DeepFenseModel(config_path=str(root / 'config.yaml'), checkpoint_path=str(root / 'best_model.pth'),
                    frontend_path=snapshot('worstchan/EAT-large_epoch20_pretrain'))
    elif name == 'deezer_ismir':
        model = DeezerISMIRModel(onnx_path=str(Path(snapshot('lofcz/ai-music-detector')) / 'ai_music_detector.onnx'))
    elif name == 'clam':
        root = asset_location('ARTIFACTBENCH_CLAM_REPO', '/home/unohee/dev/MoM-CLAM')
        evidence['source_repos']['MoM-CLAM'] = source_identity(root)
        model = CLAMModel(clam_repo=str(root), clam_ckpt=local(root / 'model_wts/best_model_triplet_loss_margin_0.2.pth'),
                    mert_repo=snapshot('m-a-p/MERT-v1-95M'),
                    wav2vec_repo=snapshot('m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres'))
    elif name == 'fst':
        root = asset_location('ARTIFACTBENCH_FST_REPO', '/home/unohee/dev/FST-AI-music-detection')
        evidence['source_repos']['FST'] = source_identity(root)
        model = FSTModel(fst_repo=str(root), stage1_ckpt=local(root / 'checkpoints/backbone_stage1.ckpt'),
                    stage2_ckpt=local(root / 'checkpoints/classifier_stage2.ckpt'),
                    mert_path=snapshot('m-a-p/MERT-v1-95M'),
                    beat_checkpoint=local(asset_location('ARTIFACTBENCH_BEAT_CHECKPOINT', '.tools/beat-this/final0.ckpt')))
        evidence['beat_checkpoint_url'] = 'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt'
    else:
        raise ValueError(f'Unknown pinned model: {name}')
    return model, evidence
