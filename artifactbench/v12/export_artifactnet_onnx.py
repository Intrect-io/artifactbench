"""완료 rc2 run의 최신 UNet/CNN을 읽기 전용으로 고정해 CPU ONNX 후보를 만든다."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

from .common import digest, write_once


def reference_assets(reference_run):
    reference_run = Path(reference_run)
    identity_path, summary_path = reference_run/'identity.json', reference_run/'summary.json'
    identity = json.loads(identity_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (identity['model'] != 'artifactnet' or identity['smoke_per_source'] != 0
            or summary['status'] != 'full fixed-checkpoint run complete'
            or summary['identity_sha256'] != digest(identity_path)
            or summary['attempted'] != summary['expected'] or summary['expected'] != 2579):
        raise ValueError('Export requires the completed rc2 ArtifactNet run')
    expected_manifest = 'feab7c4c3d037919fd784dc40e46199470088abde532c5181e34573f20dceefd'
    if identity['public_manifest_sha256'] != expected_manifest:
        raise ValueError('Export requires the corrected rc2 release')
    sources = identity['model_assets']['source_repos']['ArtifactNet']
    architecture = Path(sources['root'])/'models/artifact_unet.py'
    selected = {path: checksum for path, checksum in identity['model_assets']['files'].items()
                if Path(path).name in ('unet_codec4_best.pt', 'cnn_bigset_best.pt')}
    if len(selected) != 2:
        raise ValueError('Expected exactly the evaluated UNet and bigset CNN')
    selected[str(architecture)] = sources['python_files'][str(architecture)]
    if any(digest(path) != checksum for path, checksum in selected.items()):
        raise ValueError('Evaluated model weights or architecture changed')
    weights = {Path(path).name: Path(path) for path in selected if path.endswith('.pt')}
    return identity, architecture, weights, selected


def load_graph(reference_run):
    import torch
    from .artifactnet_onnx_graph import RawSongGraph
    identity, architecture, weights, hashes = reference_assets(reference_run)
    specification = importlib.util.spec_from_file_location('artifactbench_export_architecture', architecture)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    unet = module.ArtifactUNet(base_channels=32, mask_max=.5).eval()
    cnn = module.ResidualCNN7ch().eval()
    unet.load_state_dict(torch.load(weights['unet_codec4_best.pt'], map_location='cpu', weights_only=True))
    cnn.load_state_dict(torch.load(weights['cnn_bigset_best.pt'], map_location='cpu', weights_only=True))
    mel = module.DifferentiableMel(sr=44100, n_fft=2048, n_mels=128, top_db=80.).eval()
    return RawSongGraph(unet, cnn, mel).eval(), identity, hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-run', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only export requires CUDA_VISIBLE_DEVICES set to an empty string')
    import torch
    import onnx
    from . import artifactnet_onnx, artifactnet_onnx_graph
    from .artifactnet_onnx import CHUNK_SAMPLES, RawOnnx
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(260905)
    graph, identity, hashes = load_graph(args.reference_run)
    args.output.mkdir(parents=True, exist_ok=False)
    own_sources = [Path(__file__), Path(artifactnet_onnx.__file__), Path(artifactnet_onnx_graph.__file__)]
    inputs = {'reference_identity_sha256': digest(args.reference_run/'identity.json'),
              'reference_summary_sha256': digest(args.reference_run/'summary.json'),
              'source_file_sha256': dict(hashes, **{str(p): digest(p) for p in own_sources}),
              'python': sys.version, 'packages': sorted((d.metadata['Name'], d.version)
                    for d in importlib.metadata.distributions()), 'threads': 2, 'device': 'cpu',
              'scope': 'Local export candidate only; full real-input parity and publication pending',
              'parity_policy': {'full_scored_reference_required': True, 'absolute_score_tolerance': 1e-3,
                                'maximum_raw_05_decision_flips': 0, 'failed_reference_rows': 'not comparable; retained'}}
    write_once(args.output/'export_inputs.local.json', inputs)
    example = torch.randn(2, CHUNK_SAMPLES)*.05
    example[1] *= 1e-3
    with torch.inference_mode():
        stft_reference = torch.stft(example, 2048, 512, window=graph.window, return_complex=True).abs().unsqueeze(1)
        stft_error = float((graph.stft_magnitude(example)-stft_reference).abs().max())
        expected = graph(example).numpy()
    if stft_error > 1e-3:
        raise ValueError('DFT-convolution STFT component tolerance exceeded')
    artifact = args.output/'artifactnet_raw.onnx'
    # 기존 E2E 경로의 TorchScript exporter를 명시한다. 2.13 기본값은 dynamo=True다.
    torch.onnx.export(graph, (example,), str(artifact), dynamo=False, opset_version=18,
                      input_names=['audio_chunks'], output_names=['p_ai'],
                      dynamic_axes={'audio_chunks': {0: 'chunks'}}, external_data=False)
    exported = onnx.load(str(artifact))
    onnx.checker.check_model(exported, full_check=True)
    if (len(exported.graph.input) != 1 or len(exported.graph.output) != 1
            or any(t.data_location == onnx.TensorProto.EXTERNAL for t in exported.graph.initializer)):
        raise ValueError('Expected one input, one output and embedded weights')
    metadata = {'schema': 'artifactnet-raw-onnx/1', 'onnx_sha256': digest(artifact),
        'input': {'name': 'audio_chunks', 'dtype': 'float32', 'shape': ['chunks', CHUNK_SAMPLES],
                  'sample_rate': 44100, 'chunks_min': 1, 'chunks_max': 15,
                  'axis_meaning': 'All non-overlapping chunks of ONE song, not independent songs'},
        'output': {'name': 'p_ai', 'dtype': 'float64', 'shape': [1], 'meaning': 'raw RMS-weighted sigmoid score'},
        'preprocessing': 'First 60 seconds; pad to four seconds; discard tail below two seconds; pad other tails',
        'mel_normalization': 'Shared global top-dB maximum across all chunks and residual/H/P components',
        'precision': 'FP32 network and RMS, FP64 probability aggregation',
        'exclusions': ['audio decoder/resampler', 'LGBM rescue', 'AcoustID', 'codec TTA', 'calibration guarantee'],
        'source_sha256': {Path(path).name: checksum for path, checksum in hashes.items()},
        'reference_public_manifest_sha256': identity['public_manifest_sha256'],
        'status': 'local export candidate; full benchmark parity pending; not uploaded'}
    write_once(artifact.with_suffix('.json'), metadata)
    session = RawOnnx(artifact)
    got = session.predict_chunks(example.numpy())
    error = abs(got-float(expected[0]))
    if error > 1e-3:
        raise ValueError('Initial CPU ONNX engineering-vector tolerance exceeded')
    if any(digest(path) != checksum for path, checksum in inputs['source_file_sha256'].items()):
        raise ValueError('Export source or weights changed during export')
    if digest(args.reference_run/'identity.json') != inputs['reference_identity_sha256']:
        raise ValueError('Reference identity changed during export')
    result = {'status': metadata['status'], 'onnx_sha256': metadata['onnx_sha256'],
              'onnx_bytes': artifact.stat().st_size, 'stft_max_absolute_error': stft_error,
              'engineering_vector': {'scope': 'seeded noise, not benchmark audio',
                'torch_graph_p_ai': float(expected[0]), 'onnx_p_ai': got, 'absolute_error': error},
              'full_parity_complete': False, 'providers': session.session.get_providers()}
    write_once(args.output/'export_summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
