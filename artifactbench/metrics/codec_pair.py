"""Codec pair invariance 측정."""
from ..data.manifest import WAV_SOURCES, encode_variant, load_audio_mono

CODECS = ["wav", "mp3_128", "aac_128", "opus_128"]


def codec_pair_measure(entries, model, n_pair=50, verbose=True):
    """WAV source 트랙에서 4-way codec invariance 측정."""
    results = []
    fail_counts = {c: 0 for c in CODECS}
    visited = 0

    for e in entries:
        if len(results) >= n_pair:
            break
        visited += 1
        if e["source"] not in WAV_SOURCES:
            continue
        path = e.get("runtime_path", e.get("path"))
        if not path:
            continue
        audio = load_audio_mono(path)
        if audio is None:
            continue

        probs = {}
        ok = True
        for c in CODECS:
            variant = encode_variant(audio, c)
            if variant is None:
                fail_counts[c] += 1
                ok = False
                break
            probs[c] = model.forward(variant)
        if not ok:
            continue

        p_values = list(probs.values())
        delta = max(p_values) - min(p_values)
        results.append({
            "track_id": e["track_id"],
            "source": e["source"],
            "label": e["label"],
            "probs": probs,
            "delta": delta,
        })
        if verbose and len(results) % 10 == 0:
            print(f"  codec pair: {len(results)}/{n_pair} (visited={visited})", flush=True)

    if verbose:
        print(f"  codec pair DONE: {len(results)} pairs", flush=True)
    return results
