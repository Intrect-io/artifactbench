"""해시와 재실행 시 입력 변경 방지를 공통으로 처리한다."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, value):
    """같은 내용의 재실행만 허용하며, 기존 산출물을 덮어쓰지 않는다."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)


def rank(key, seed=260905):
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def manifest_rows(doc):
    for split in ("train", "val", "test", "bench"):
        for row in doc.get(split, []):
            yield split, row
