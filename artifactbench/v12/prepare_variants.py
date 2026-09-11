"""확정 선택의 부가 native 전송 파일만 별도 검증 목록으로 추출한다."""
import argparse
import json
from pathlib import Path

from .common import digest, write_once


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    variants = []
    for entry in json.loads(args.selection.read_text())['bench']:
        for variant in entry.get('additional_native_variants', []):
            variants.append(dict(variant, primary_path=entry['path'],
                                 primary_source=entry['source'], partition='native_paired_variant'))
    write_once(args.output, {'selection_sha256': digest(args.selection), 'bench': variants})
    print(f'Additional native variants: {len(variants)}')


if __name__ == '__main__':
    main()
