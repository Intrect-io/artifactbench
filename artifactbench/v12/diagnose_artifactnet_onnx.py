"""고정된 한 곡의 GPU/CPU/DFT/ONNX 차이를 측정한다. 원본과 활성 실행은 수정하지 않는다."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .artifactnet_onnx import RawOnnx, song_chunks
from .common import digest, write_once
from .export_artifactnet_onnx import load_graph
from .model_assets import build_pinned
from .report import load_completed
from .run import seed_record
from .validate_artifactnet_onnx import compare
from .verify_release import verify


def describe(value):
    value = np.asarray(value)
    if not value.size or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError('Expected nonempty finite floating-point diagnostic array')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            'minimum': float(value.min()), 'maximum': float(value.max())}


def difference(left, right):
    describe(left)
    describe(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('Diagnostic arrays have different shapes or dtypes')
    delta = right.astype(np.float64)-left.astype(np.float64)
    return {'exact': bool(np.array_equal(left, right)), 'max_absolute': float(np.abs(delta).max()),
            'mean_absolute': float(np.abs(delta).mean()), 'rms': float(np.sqrt(np.mean(delta**2)))}


def capture(unet, cnn, call):
    """여러 UNet microbatch를 순서대로 모으고 예외 시에도 모든 hook을 제거한다."""
    captured = {key: [] for key in ('magnitude', 'mask', 'features', 'logits')}
    def hook(input_key, output_key):
        def observe(module, inputs, output):
            captured[input_key].append(inputs[0].detach().cpu().numpy().copy())
            captured[output_key].append(output.detach().cpu().numpy().copy())
        return observe
    handles = []
    try:
        handles.append(unet.register_forward_hook(hook('magnitude', 'mask')))
        handles.append(cnn.register_forward_hook(hook('features', 'logits')))
        with torch.no_grad():
            score = float(call())
        if not np.isfinite(score) or not 0 <= score <= 1 or any(not v for v in captured.values()):
            raise ValueError('Incomplete or invalid diagnostic forward')
        arrays = {key: np.concatenate(values, axis=0) for key, values in captured.items()}
        result = {'p_ai': score, 'unet_calls': len(captured['mask']),
                  'tensors': {key: describe(value) for key, value in arrays.items()},
                  'chunk_logits': arrays['logits'].reshape(-1).tolist()}
        return arrays, result
    finally:
        for handle in handles:
            handle.remove()


def fft_magnitude(chunks, window, split=False):
    """원래 torch.stft를 사용하며 split=True는 rc2 factory의 batch=1을 재현한다."""
    def transform(value):
        return torch.stft(value, 2048, 512, window=window, return_complex=True).abs().unsqueeze(1)
    return torch.cat([transform(chunk[None]) for chunk in chunks]) if split else transform(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference-run', 'release', 'snapshots', 'parity-run', 'output'):
        ap.add_argument('--'+name, required=True, type=Path)
    ap.add_argument('--record-id', required=True)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Diagnostic is CPU-only; set CUDA_VISIBLE_DEVICES to an empty string')
    from .audio import load_audio_float
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verify(args.release)
    public = {r['id']: r for r in json.loads((args.release/'manifest.public.json').read_text())['bench']}
    local = {r['id']: r for r in json.loads((args.release/'manifest.local.json').read_text())['bench']}
    references, identity = load_completed(args.reference_run, public)
    entry, original = public[args.record_id], references[args.record_id]
    prior_path = args.parity_run/'records'/(args.record_id+'.json')
    prior = json.loads(prior_path.read_text())
    bound = json.loads((args.parity_run/'inputs.local.json').read_text())
    expected = dict(bound['source_file_sha256'])
    if any(digest(path) != checksum for path, checksum in expected.items()):
        raise ValueError('Pinned parity source/input drift')
    if (original['outcome'] != 'scored' or prior['id'] != args.record_id
            or prior['audio_sha256'] != entry['sha256'] or prior['original_gpu_p_ai'] != original['prob']
            or prior['original_gpu_comparison'] != compare(prior['onnx_p_ai'], original['prob'], .001)):
        raise ValueError('Diagnostic reference mismatch')
    expected.update({str(path): digest(path) for path in
                     (Path(__file__), args.snapshots, prior_path, args.parity_run/'inputs.local.json')})
    audio_path = Path(local[args.record_id]['path'])
    if digest(audio_path) != entry['sha256']:
        raise ValueError('Frozen audio changed')
    expected[str(audio_path)] = entry['sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output/'inputs.local.json', {'id': args.record_id, 'source_file_sha256': expected,
        'selection': 'Outcome-selected numerical diagnostic, not a performance or parity sample',
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'torchaudio', 'numpy', 'onnxruntime', 'soundfile')},
        'device': 'cpu', 'threads': 2, 'original_gpu_score': original['prob'],
        'scope': 'No original score, candidate, threshold, active source or weight changes'})
    waveform = load_audio_float(audio_path)
    if len(waveform) != original['decoded_frames']:
        raise ValueError('Decoder frame count changed')
    chunks = song_chunks(waveform)
    write_once(args.output/'input_arrays.json', {'waveform': describe(waveform), 'chunks': describe(chunks)})
    seed_record('model-initialization')
    adapter, assets = build_pinned('artifactnet', args.snapshots)
    if assets != identity['model_assets']:
        raise ValueError('Original factory asset mismatch')
    adapter.load('cpu')
    # Factory의 batch=1과 기본 adapter의 batch=8은 구분해야 한다.
    import src.pipeline.infer as infer
    if infer.GPU_BATCH_MAX != 1:
        raise ValueError('Expected the original rc2 microbatch condition')
    traces, results = {}, {}
    def execute(name, unet, cnn, call):
        seed_record(args.record_id)
        started = time.monotonic()
        trace, result = capture(unet, cnn, call)
        result['seconds'] = time.monotonic()-started
        result['original_gpu_comparison'] = compare(result['p_ai'], original['prob'], .001)
        traces[name], results[name] = trace, result
        write_once(args.output/(name+'.json'), result)
        print(json.dumps({'condition': name, 'p_ai': result['p_ai'], 'seconds': result['seconds']}), flush=True)
    execute('original_cpu_batch1', adapter._net._unet, adapter._net._cnn, lambda: adapter.forward(waveform))
    graph, _, _ = load_graph(args.reference_run)
    tensor = torch.from_numpy(chunks)
    execute('graph_dft', graph.unet, graph.cnn, lambda: graph(tensor)[0])
    with patch.object(graph, 'stft_magnitude', lambda x: fft_magnitude(x, graph.window)):
        execute('graph_fft', graph.unet, graph.cnn, lambda: graph(tensor)[0])
    forward_unet = graph.unet.forward
    with patch.object(graph, 'stft_magnitude', lambda x: fft_magnitude(x, graph.window, split=True)), \
            patch.object(graph.unet, 'forward', lambda x: torch.cat([forward_unet(chunk[None]) for chunk in x])):
        execute('graph_fft_batch1', graph.unet, graph.cnn, lambda: graph(tensor)[0])
    started = time.monotonic()
    replay = RawOnnx(args.model).predict_chunks(chunks)
    results['onnx_replay'] = {'p_ai': replay, 'seconds': time.monotonic()-started,
        'previous_onnx_comparison': compare(replay, prior['onnx_p_ai'], .001),
        'original_gpu_comparison': compare(replay, original['prob'], .001)}
    write_once(args.output/'onnx_replay.json', results['onnx_replay'])
    pairs = [('original_cpu_batch1', 'graph_fft_batch1'), ('graph_fft_batch1', 'graph_fft'),
             ('graph_fft', 'graph_dft')]
    comparisons = {left+'__'+right: {key: difference(traces[left][key], traces[right][key])
                    for key in traces[left]} for left, right in pairs}
    scores = {key: value['p_ai'] for key, value in results.items()}
    scores['original_gpu'], scores['prior_onnx'] = original['prob'], prior['onnx_p_ai']
    if any(digest(path) != checksum for path, checksum in expected.items()):
        raise ValueError('Diagnostic source/input changed during execution')
    summary = {'status': 'single-recording CPU numerical diagnostic complete', 'id': args.record_id,
        'scores': scores, 'trace_comparisons': comparisons,
        'result_sha256': {name: digest(args.output/(name+'.json')) for name in results},
        'inputs_sha256': digest(args.output/'inputs.local.json'),
        'limits': 'Outcome-selected one-recording diagnostic. No full-parity success or new GPU execution claim.'}
    write_once(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
