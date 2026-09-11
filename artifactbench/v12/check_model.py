"""가중치 실제 로딩만 검증한다. 합성 입력이나 검출 정확도 결과를 만들지 않는다."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from .common import digest, write_once
from .model_assets import build_pinned


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--snapshots', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('Run with HF_HUB_OFFLINE=1 so model loading cannot fetch moving hub assets')
    torch.set_num_threads(2)
    report = {'model': args.model, 'python': sys.version, 'torch': torch.__version__,
              'tool_sha256': digest(__file__), 'device': args.device, 'scope': 'real weight loading only; no inference score'}
    start = time.monotonic()
    try:
        model, report['assets'] = build_pinned(args.model, args.snapshots)
        model.load(args.device)
        report.update(outcome='loaded', model_info=model.info(), elapsed_seconds=time.monotonic()-start)
        if args.device == 'cuda':
            report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        report.update(outcome='load_error', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        write_once(args.output, report)
        raise
    write_once(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'assets'}), flush=True)


if __name__ == '__main__':
    main()
