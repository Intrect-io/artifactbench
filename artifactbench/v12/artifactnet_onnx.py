"""학습 코드 없이 NumPy/ORT로 곡 단위 raw P(AI)를 실행한다. 디코더는 별도다."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import digest


SR = 44100
CHUNK_SAMPLES = 4*SR
MAX_SAMPLES = 60*SR


def song_chunks(audio):
    """rc2 adapter의 prefix/padding/50%-tail 정책. 한 호출에 반드시 한 곡만 넣는다."""
    audio = np.asarray(audio)
    if audio.ndim != 1 or not audio.size:
        raise ValueError('Expected finite nonempty mono waveform')
    if not np.issubdtype(audio.dtype, np.floating) or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
        raise ValueError('Expected decoded floating-point waveform clipped to [-1, 1]')
    audio = np.ascontiguousarray(audio[:MAX_SAMPLES], dtype=np.float32)
    if len(audio) < CHUNK_SAMPLES:
        audio = np.pad(audio, (0, CHUNK_SAMPLES-len(audio)))
    chunks = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start:start+CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES//2:
            continue
        chunks.append(np.pad(chunk, (0, CHUNK_SAMPLES-len(chunk))))
    return np.ascontiguousarray(np.stack(chunks), dtype=np.float32)


class RawOnnx:
    """입력 청크 축은 독립 곡 batch가 아니라 단일 곡의 1–15개 청크다."""

    def __init__(self, artifact, threads=2):
        import onnxruntime as ort
        artifact = Path(artifact)
        metadata = json.loads(artifact.with_suffix('.json').read_text())
        if metadata['schema'] != 'artifactnet-raw-onnx/1' or digest(artifact) != metadata['onnx_sha256']:
            raise ValueError('ONNX artifact identity mismatch')
        if threads < 1:
            raise ValueError('Thread count must be positive')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry('session.use_deterministic_compute', '1')
        self.session = ort.InferenceSession(str(artifact), sess_options=options, providers=['CPUExecutionProvider'])
        self.session.disable_fallback()
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != 'audio_chunks' or inputs[0].type != 'tensor(float)'
                or inputs[0].shape != ['chunks', CHUNK_SAMPLES] or len(outputs) != 1
                or outputs[0].name != 'p_ai' or outputs[0].type != 'tensor(double)' or outputs[0].shape != [1]):
            raise ValueError('Unexpected ONNX input/output contract')

    def predict_chunks(self, chunks):
        chunks = np.asarray(chunks)
        if (chunks.dtype != np.float32 or chunks.ndim != 2 or chunks.shape[1] != CHUNK_SAMPLES
                or not 1 <= len(chunks) <= 15 or not np.isfinite(chunks).all() or np.max(np.abs(chunks)) > 1):
            raise ValueError('Expected 1–15 float32 four-second chunks from one clipped mono song')
        result = np.asarray(self.session.run(['p_ai'], {'audio_chunks': np.ascontiguousarray(chunks)})[0])
        if result.shape != (1,) or not np.isfinite(result).all() or not 0 <= result[0] <= 1:
            raise ValueError('Invalid raw P(AI) output')
        return float(result[0])

    def predict(self, audio_44100_mono):
        return self.predict_chunks(song_chunks(audio_44100_mono))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--waveform-npy', required=True, type=Path,
                    help='Already decoded 44.1 kHz mono floating-point waveform; no implicit resampling')
    args = ap.parse_args()
    audio = np.load(args.waveform_npy, allow_pickle=False)
    print(json.dumps({'p_ai': RawOnnx(args.model).predict(audio)}))


if __name__ == '__main__':
    main()
