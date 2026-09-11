"""보존된 v2 ONNX의 STFT 부분만 별도 복사본에서 교체하는 CPU 수치 실험."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .artifactnet_onnx import RawOnnx, song_chunks
from .common import digest, write_once
from .diagnose_artifactnet_onnx import describe, difference, fft_magnitude
from .validate_artifactnet_onnx import compare

PARENT_SHA256 = '590d992b0839ac49900f6eed820cd2c9e68af76304af932dcff37e5bfd7b4621'
MAGNITUDE = '/Unsqueeze_1_output_0'


def native_prefix(window):
    import onnx
    from onnx import helper as h, numpy_helper as nh
    window = np.asarray(window)
    if window.shape != (2048,) or window.dtype != np.float32 or not np.isfinite(window).all():
        raise ValueError('Expected the exact FP32 2048-sample reference window')
    prefix = '__native_stft/'
    values = {'pads': np.array([0, 1024, 0, 1024], dtype=np.int64),
              'complex_axis': np.array([2], dtype=np.int64), 'channel_axis': np.array([1], dtype=np.int64),
              'last_axis': np.array([-1], dtype=np.int64), 'step': np.array(512, dtype=np.int64),
              'length': np.array(2048, dtype=np.int64), 'window': window}
    initializers = [nh.from_array(value, prefix+name) for name, value in values.items()]
    def node(kind, inputs, output, **attrs):
        return h.make_node(kind, inputs, [output], name=prefix+kind, **attrs)
    nodes = [
        node('Pad', ['audio_chunks', prefix+'pads'], prefix+'padded', mode='reflect'),
        node('Unsqueeze', [prefix+'padded', prefix+'complex_axis'], prefix+'signal'),
        node('STFT', [prefix+'signal', prefix+'step', prefix+'window', prefix+'length'], prefix+'complex', onesided=1),
        node('Mul', [prefix+'complex', prefix+'complex'], prefix+'squared'),
        node('ReduceSum', [prefix+'squared', prefix+'last_axis'], prefix+'power', keepdims=0),
        node('Sqrt', [prefix+'power'], prefix+'magnitude'),
        node('Transpose', [prefix+'magnitude'], prefix+'frequency_time', perm=[0, 2, 1]),
        h.make_node('Unsqueeze', [prefix+'frequency_time', prefix+'channel_axis'], [MAGNITUDE], name=prefix+'Output'),
    ]
    return nodes, initializers


def replace_prefix(parent, window):
    import onnx
    child = onnx.ModelProto()
    child.CopyFrom(parent)
    if sum(MAGNITUDE in n.output for n in child.graph.node) != 1:
        raise ValueError('Expected one known v2 magnitude boundary')
    nodes, initializers = native_prefix(window)
    original = [n for n in child.graph.node if MAGNITUDE not in n.output]
    # 새 magnitude를 기준으로 출력의 선행 노드만 남긴다. 공유 상수는 보존된다.
    needed, kept = {v.name for v in child.graph.output}, []
    for node in reversed(nodes+original):
        if any(name in needed for name in node.output):
            kept.append(node)
            needed.update(node.input)
    kept.reverse()
    retained = [t for t in child.graph.initializer if t.name in needed]
    removed = {t.name for t in child.graph.initializer if t.name not in needed}
    if removed != {'cosine', 'sine'}:
        raise ValueError('Replacement would remove more than the DFT bases')
    prior_tensors = {t.name: t.SerializeToString() for t in parent.graph.initializer}
    if any(t.SerializeToString() != prior_tensors[t.name] for t in retained):
        raise ValueError('Learned initializer changed')
    del child.graph.node[:]
    child.graph.node.extend(kept)
    del child.graph.initializer[:]
    child.graph.initializer.extend(retained+initializers)
    live_values = {v.name for v in child.graph.input} | {o for n in kept for o in n.output}
    infos = [v for v in child.graph.value_info if v.name in live_values]
    del child.graph.value_info[:]
    child.graph.value_info.extend(infos)
    onnx.checker.check_model(child, full_check=True)
    return child, {'unchanged_parent_initializers': len(retained), 'removed_initializers': sorted(removed),
                   'standard_stft_nodes': sum(n.op_type == 'STFT' for n in kept), 'nodes': len(kept)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'diagnostic', 'release', 'output'):
        ap.add_argument('--'+name, required=True, type=Path)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('STFT probe is CPU-only')
    import onnx
    from .audio import load_audio_float
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if digest(args.model) != PARENT_SHA256:
        raise ValueError('Probe requires the preserved v2 ONNX')
    diagnostic = json.loads((args.diagnostic/'summary.json').read_text())
    inputs = json.loads((args.diagnostic/'inputs.local.json').read_text())
    if digest(args.diagnostic/'inputs.local.json') != diagnostic['inputs_sha256']:
        raise ValueError('Diagnostic identity changed')
    expected = dict(inputs['source_file_sha256'])
    for name, checksum in diagnostic['result_sha256'].items():
        expected[str(args.diagnostic/(name+'.json'))] = checksum
    if any(digest(path) != checksum for path, checksum in expected.items()):
        raise ValueError('Diagnostic source/input changed')
    identifier = diagnostic['id']
    public = {r['id']: r for r in json.loads((args.release/'manifest.public.json').read_text())['bench']}
    local = {r['id']: r for r in json.loads((args.release/'manifest.local.json').read_text())['bench']}
    audio_path = Path(local[identifier]['path'])
    if digest(audio_path) != public[identifier]['sha256']:
        raise ValueError('Frozen input changed')
    expected.update({str(path): digest(path) for path in (Path(__file__), args.model,
        args.model.with_suffix('.json'), args.diagnostic/'summary.json', args.diagnostic/'input_arrays.json')})
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output/'inputs.local.json', {'source_file_sha256': expected, 'id': identifier,
        'scope': 'Outcome-selected one-recording STFT intervention, not a release export or full parity'})
    waveform = load_audio_float(audio_path)
    chunks = song_chunks(waveform)
    if describe(chunks) != json.loads((args.diagnostic/'input_arrays.json').read_text())['chunks']:
        raise ValueError('Diagnostic chunk bytes differ')
    window = torch.hann_window(2048)
    child, changes = replace_prefix(onnx.load(str(args.model)), window.numpy())
    artifact = args.output/'stft_probe.onnx'
    onnx.save(child, str(artifact))
    metadata = json.loads(args.model.with_suffix('.json').read_text())
    metadata.update(onnx_sha256=digest(artifact), status='single-recording CPU STFT probe; not a release candidate',
                    stft='Standard ONNX STFT-17 with explicit reflect centering and FP32 reference Hann window',
                    parent_onnx_sha256=PARENT_SHA256)
    write_once(artifact.with_suffix('.json'), metadata)
    # 원래 raw runtime 옵션 그대로 실행한다. 공개 성공을 표시할 export_summary는 만들지 않는다.
    runtime = RawOnnx(artifact)
    begin = time.monotonic()
    score = runtime.predict_chunks(chunks)
    elapsed = time.monotonic()-begin
    # 같은 실제 청크의 STFT 경계도 별도 작은 표준 그래프로 측정한다.
    nodes, initializers = native_prefix(window.numpy())
    from onnx import helper as h, TensorProto as T
    stft_only = h.make_model(h.make_graph(nodes, 'STFT-only diagnostic',
        [h.make_tensor_value_info('audio_chunks', T.FLOAT, ['chunks', 176400])],
        [h.make_tensor_value_info(MAGNITUDE, T.FLOAT, ['chunks', 1, 1025, 345])], initializers),
        opset_imports=[h.make_opsetid('', 18)], ir_version=8)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(stft_only.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    magnitude = session.run([MAGNITUDE], {'audio_chunks': chunks})[0]
    with torch.no_grad():
        reference_mag = fft_magnitude(torch.from_numpy(chunks), window, split=True).numpy()
    if any(digest(path) != checksum for path, checksum in expected.items()):
        raise ValueError('STFT probe inputs changed during execution')
    summary = {'status': 'single-recording STFT probe measured', 'id': identifier, 'p_ai': score,
        'seconds': elapsed, 'model_bytes': artifact.stat().st_size, 'model_sha256': digest(artifact),
        'changes': changes, 'stft_vs_cpu_reference': difference(reference_mag, magnitude),
        'comparisons': {name: compare(score, value, .001) for name, value in diagnostic['scores'].items()},
        'limits': 'CPU provider only; outcome-selected one recording; no global superiority or full-parity claim.'}
    write_once(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
