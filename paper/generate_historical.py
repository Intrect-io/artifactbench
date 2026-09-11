"""검증된 과거 실행만으로 초안의 historical audit 표와 그림을 생성한다."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    paper = Path(__file__).resolve().parent
    run = paper.parent / "out/current_production_260905_recovered"
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()]
    summary = json.loads((run / "summary.json").read_text())
    scored = [r for r in rows if "error" not in r]
    assert len(rows) == summary["attempted"] == summary["expected"] == 2164
    assert len(scored) == summary["scored"] == 2163
    labels = np.array([r["label"] == "ai" for r in scored])
    probs = np.array([r["prob"] for r in scored])
    assert np.isfinite(probs).all()
    modes = [("Raw, $t=0.5$", "raw_05", probs >= 0.5),
             ("Raw, $t=0.225$", "raw_operating", probs >= 0.225),
             ("CNN + matched rescue", "production", np.array([r["production_ai"] for r in scored]))]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Operating point & TP & FP & FN & TN & F1 & FPR (\%) \\", r"\midrule"]
    for title, key, predicted in modes:
        counts = {"tp": int(np.sum(labels & predicted)), "fp": int(np.sum(~labels & predicted)),
                  "fn": int(np.sum(labels & ~predicted)), "tn": int(np.sum(~labels & ~predicted))}
        assert all(counts[k] == summary[key][k] for k in counts)
        m = summary[key]
        lines.append(f"{title} & {m['tp']} & {m['fp']} & {m['fn']} & {m['tn']} & {m['f1']:.4f} & {100*m['fpr']:.2f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (paper / "generated").mkdir(exist_ok=True)
    (paper / "figures").mkdir(exist_ok=True)
    (paper / "generated/historical_table.tex").write_text("\n".join(lines) + "\n")
    codec = json.loads((run / "codec_pair.json").read_text())
    delta = np.sort([r["delta"] for r in codec])
    assert len(delta) == 50
    for r in codec:
        assert set(r["probs"]) == {"wav", "mp3_128", "aac_128", "opus_128"}
        assert np.isclose(r["delta"], max(r["probs"].values()) - min(r["probs"].values()))
    fig, ax = plt.subplots(figsize=(5.2, 2.7), layout="constrained")
    ax.step(delta, np.arange(1, len(delta)+1)/len(delta), where="post", color="#255c85", linewidth=1.7)
    ax.set(xlabel="Within-recording probability range", ylabel="Empirical CDF", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.18)
    fig.savefig(paper / "figures/historical_codec_cdf.pdf")
    plt.close(fig)
    data = {"status": "historical pilot only; not v1.2 results", "expected": len(rows), "scored": len(scored),
            "auc": roc_auc_score(labels, probs), "codec_n": len(codec),
            "codec_mean": float(np.mean(delta)), "codec_median": float(np.median(delta)),
            "codec_max": float(np.max(delta)),
            "input_hashes": {str(p.relative_to(paper.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (run/"rows.jsonl", run/"summary.json", run/"codec_pair.json", Path(__file__))}}
    (paper / "generated/historical_evidence.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
