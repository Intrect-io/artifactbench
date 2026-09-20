"""Source-level 측정: source별 n sample forward → prob/rate stats."""
import numpy as np

from ..data.manifest import load_audio_mono


def measure_source(entries, model, verbose=True):
    """Score a preselected source cohort and retain every failure separately."""
    results = []
    failures = []
    for i, e in enumerate(entries):
        path = e.get("runtime_path", e.get("path"))
        if not path:
            failures.append({
                "track_id": e["track_id"],
                "source": e["source"],
                "stage": "resolve",
                "error": "manifest row has no runtime_path or path",
            })
            continue
        audio = load_audio_mono(path)
        if audio is None:
            failures.append({
                "track_id": e["track_id"],
                "source": e["source"],
                "stage": "decode",
                "error": "audio decode failed",
            })
            continue
        try:
            prob = model.forward(audio)
            if not np.isfinite(prob):
                raise ValueError(f"model returned non-finite probability: {prob}")
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"model returned probability outside [0, 1]: {prob}")
            results.append({
                "prob": prob,
                "label": e["label"],
                "track_id": e["track_id"],
                "source": e["source"],
                "path": path,
            })
        except Exception as ex:
            failures.append({
                "track_id": e["track_id"],
                "source": e["source"],
                "stage": "inference",
                "error_type": type(ex).__name__,
                "error": str(ex),
            })
            if verbose:
                print(f"  skip {e['track_id']}: {ex}")
        if verbose and (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(entries)} attempted", flush=True)
    return results, failures


def summarize(results, attempted=None, threshold=0.5):
    """raw results → source summary dict."""
    if not results:
        return None
    probs = np.array([r["prob"] for r in results])
    labels = np.array([1 if r["label"] == "ai" else 0 for r in results])
    pred = (probs >= threshold).astype(int)
    n = len(probs)
    is_ai = int(labels[0])
    rate = float(pred.mean())
    rate_name = "tpr" if is_ai else "fpr"
    return {
        "n": n,
        "attempted": attempted if attempted is not None else n,
        "failures": (attempted - n) if attempted is not None else 0,
        "threshold": threshold,
        "label": "ai" if is_ai else "real",
        "prob_mean": float(probs.mean()),
        "prob_median": float(np.median(probs)),
        "prob_p10": float(np.percentile(probs, 10)),
        "prob_p90": float(np.percentile(probs, 90)),
        rate_name: rate,
    }
