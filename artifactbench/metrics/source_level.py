"""Source-level 측정: source별 n sample forward → prob/rate stats."""
import numpy as np
from ..data.manifest import load_audio_mono


def measure_source(entries, model, max_n=100, verbose=True):
    """source entries → forward → raw results list."""
    results = []
    for i, e in enumerate(entries[:max_n]):
        audio = load_audio_mono(e["path"])
        if audio is None:
            continue
        try:
            prob = model.forward(audio)
            results.append({
                "prob": prob,
                "label": e["label"],
                "path": e["path"],
                "source": e["source"],
            })
        except Exception as ex:
            if verbose:
                print(f"  skip {e['path']}: {ex}")
        if verbose and (i + 1) % 20 == 0:
            print(f"    {i+1}/{min(max_n, len(entries))} done", flush=True)
    return results


def summarize(results):
    """raw results → source summary dict."""
    if not results:
        return None
    probs = np.array([r["prob"] for r in results])
    labels = np.array([1 if r["label"] == "ai" else 0 for r in results])
    pred = (probs >= 0.5).astype(int)
    n = len(probs)
    is_ai = int(labels[0])
    rate = float(pred.mean())
    rate_name = "tpr" if is_ai else "fpr"
    return {
        "n": n,
        "label": "ai" if is_ai else "real",
        "prob_mean": float(probs.mean()),
        "prob_median": float(np.median(probs)),
        "prob_p10": float(np.percentile(probs, 10)),
        "prob_p90": float(np.percentile(probs, 90)),
        rate_name: rate,
    }
