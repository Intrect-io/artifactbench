#!/usr/bin/env python3
# Created: 2026-11-05
# Purpose: Compare a candidate MuseBench report with a baseline report built on the same cohort and the same
#          seen/unseen field, write comparison.json / comparison.md, and evaluate a gate spec fixed in advance.
# Dependencies: standard library only (imports scripts/musebench/report.py from this repository).
# Test Status: tests/test_musebench_compare.py
"""Compare two ``musebench_report.json`` files and evaluate gates.

Usage::

    python -I scripts/musebench/compare.py --baseline BASE.json --candidate CAND.json \\
        --out DIR [--gates SPEC.json] [--section primary]

Same-population rule: the two reports must describe the identical cohort and the identical
seen/unseen partition, otherwise the comparison is refused (exit 2). The cohort is
``run.cohort.sha256`` (a cohort-restricted report) or, for a report on the whole sealed eval
cohort, the pair (``run.input_manifest_sha256``, ``coverage.eval_rows``). The partition is
``run.seen_field`` (``seen_in_training`` when a report predates that key). To compare detectors
with different training lineages, build both reports with the same ``--seen-field``.

Output per split (all / seen / unseen): every headline metric and every generator's clean FNR and
AUROC, with baseline, candidate, delta (candidate - baseline) and the Wilson 95 % interval of each
side; rate deltas also carry Newcombe's hybrid-score interval, which treats the two runs as
independent samples (on one shared cohort it is usually wider than a paired interval).

Gate spec: ``{"rules": [{"metric": ..., "split": "all", "generator": ..., "max": ...}, ...]}``,
comparators ``max`` / ``min`` on the candidate value and ``max_increase`` / ``min_decrease`` on the
change from the baseline, all in rate units (0.05 = 5 points). Exit codes: 0 every rule passes (or no
gate spec), 3 at least one rule fails, 2 invalid input (cohort or seen-field mismatch, malformed spec,
unknown metric, or a metric with no rows in either report).
"""
from __future__ import annotations

import os as _os
import sys as _sys

# This directory holds select.py, which shadows the stdlib ``select`` module when the
# directory is on sys.path (``python scripts/musebench/x.py`` without -I). Drop it.
if _sys.path and _os.path.abspath(_sys.path[0] or ".") == _os.path.dirname(_os.path.abspath(__file__)):
    _sys.path.pop(0)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.musebench.report import GENERATOR_ORDER, SEEN_FIELD, SPLITS, TRANSFORM_ORDER  # noqa: E402

SCHEMA = "musebench-comparison/v1"
EXIT_PASS, EXIT_INVALID, EXIT_GATE_FAIL = 0, 2, 3
# Headline metrics of a report split: rate blocks ({n, k, rate, ci95}) and scalars.
RATE_METRICS = ("human_jamendo_fpr", "fma_ood_accuracy", "clean_ai_fnr", "co_created_fnr")
SCALAR_METRICS = ("balanced_accuracy_clean", "auroc_clean_ai_vs_jamendo")
# Per-generator metrics, read from per_generator_clean_fnr[<generator>].
GENERATOR_METRICS = {"clean_fnr": "rate", "auroc_vs_jamendo": "scalar"}
COMPARATORS = ("max", "min", "max_increase", "min_decrease")
RULE_KEYS = {"metric", "split", "generator", "name", "note", *COMPARATORS}


class InvalidInput(ValueError):
    """The comparison cannot be made: exit 2."""


# --------------------------------------------------------------------------- inputs

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path, what: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidInput(f"cannot read {what} {path}: {exc}") from exc


def section_of(report: dict, section: str) -> dict:
    """The report body that holds ``splits``: ``report[section]`` for a full report, else the report itself."""
    body = report.get(section) if section in report else (report if "splits" in report else None)
    if not isinstance(body, dict) or not isinstance(body.get("splits"), dict):
        raise InvalidInput(f"report has no '{section}' section with splits")
    missing = [s for s in SPLITS if s not in body["splits"]]
    if missing:
        raise InvalidInput(f"report section '{section}' lacks splits {missing}")
    return body


def cohort_identity(report: dict, body: dict) -> dict:
    run = report.get("run") or {}
    cohort = run.get("cohort")
    if isinstance(cohort, dict) and cohort.get("sha256"):
        return {"kind": "cohort_file", "sha256": cohort["sha256"], "size": cohort.get("size")}
    if run.get("cohort_sha256"):
        return {"kind": "cohort_file", "sha256": run["cohort_sha256"], "size": run.get("cohort_size")}
    if run.get("input_manifest_sha256"):
        return {"kind": "full_eval_cohort", "input_manifest_sha256": run["input_manifest_sha256"],
                "eval_rows": (body.get("coverage") or {}).get("eval_rows")}
    raise InvalidInput("report records no cohort (run.cohort.sha256) and no run.input_manifest_sha256")


def seen_field_of(report: dict) -> str:
    return (report.get("run") or {}).get("seen_field") or SEEN_FIELD


def side_info(path: Path, report: dict, body: dict) -> dict:
    run = report.get("run") or {}
    return {"report": str(path), "report_sha256": sha256_file(path), "run_dir": run.get("run_dir"),
            "threshold": body.get("threshold", run.get("threshold")), "decision_rule": body.get("decision_rule"),
            "coverage": {k: (body.get("coverage") or {}).get(k)
                         for k in ("eval_rows", "scored", "failures", "missing", "seen_rows")}}


# --------------------------------------------------------------------------- statistics

def newcombe(cand: dict, base: dict) -> list[float] | None:
    """Hybrid-score 95 % interval of cand.rate - base.rate from the two Wilson intervals (Newcombe 1998, #10)."""
    if not (cand.get("n") and base.get("n")) or cand.get("ci95") is None or base.get("ci95") is None:
        return None
    p1, (l1, u1) = cand["rate"], cand["ci95"]
    p2, (l2, u2) = base["rate"], base["ci95"]
    d = p1 - p2
    return [d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]


def rate_block(x) -> dict:
    if not isinstance(x, dict):
        return {"n": 0, "k": 0, "rate": None, "ci95": None}
    return {k: x.get(k) for k in ("n", "k", "rate", "ci95")}


def entry(kind: str, base, cand) -> dict:
    if kind == "rate":
        b, c = rate_block(base), rate_block(cand)
        delta = c["rate"] - b["rate"] if c["rate"] is not None and b["rate"] is not None else None
        return {"kind": "rate", "baseline": b, "candidate": c, "delta": delta, "delta_ci95": newcombe(c, b)}
    b = base if isinstance(base, (int, float)) else None
    c = cand if isinstance(cand, (int, float)) else None
    return {"kind": "scalar", "baseline": b, "candidate": c,
            "delta": c - b if c is not None and b is not None else None}


def compare_bodies(base: dict, cand: dict) -> dict:
    out = {}
    for split in SPLITS:
        bs, cs = base["splits"][split], cand["splits"][split]
        metrics = {m: entry("rate", bs.get(m), cs.get(m)) for m in RATE_METRICS}
        metrics.update({m: entry("scalar", bs.get(m), cs.get(m)) for m in SCALAR_METRICS})
        bg, cg = bs.get("per_generator_clean_fnr") or {}, cs.get("per_generator_clean_fnr") or {}
        generators = [g for g in GENERATOR_ORDER if g in bg or g in cg] + \
                     sorted((set(bg) | set(cg)) - set(GENERATOR_ORDER))
        per_gen = {g: {"clean_fnr": entry("rate", bg.get(g), cg.get(g)),
                       "auroc_vs_jamendo": entry("scalar", (bg.get(g) or {}).get("auroc_vs_jamendo"),
                                                 (cg.get(g) or {}).get("auroc_vs_jamendo"))}
                   for g in generators}
        bt, ct = bs.get("per_transform_fnr") or {}, cs.get("per_transform_fnr") or {}
        per_tr = {v: {"fnr": entry("rate", bt.get(v), ct.get(v))} for v in TRANSFORM_ORDER if v in bt or v in ct}
        out[split] = {"metrics": metrics, "per_generator": per_gen, "per_transform": per_tr}
    return out


# --------------------------------------------------------------------------- gates

def value_of(e: dict, side: str):
    v = e[side]
    return v.get("rate") if e["kind"] == "rate" else v


def lookup(splits: dict, rule: dict) -> dict:
    split = rule.get("split", "all")
    if split not in SPLITS:
        raise InvalidInput(f"rule {rule}: split must be one of {list(SPLITS)}")
    metric = rule.get("metric")
    if "generator" in rule:
        if metric not in GENERATOR_METRICS:
            raise InvalidInput(f"rule {rule}: per-generator metric must be one of {list(GENERATOR_METRICS)}")
        gen = splits[split]["per_generator"].get(rule["generator"])
        if gen is None:
            raise InvalidInput(f"rule {rule}: generator {rule['generator']!r} is in neither report")
        return gen[metric]
    if metric not in splits[split]["metrics"]:
        raise InvalidInput(f"rule {rule}: metric must be one of {list(RATE_METRICS + SCALAR_METRICS)} "
                           f"(or {list(GENERATOR_METRICS)} with a generator)")
    return splits[split]["metrics"][metric]


def rule_label(rule: dict) -> str:
    if rule.get("name"):
        return str(rule["name"])
    parts = [rule["metric"], rule.get("split", "all")] + ([rule["generator"]] if "generator" in rule else [])
    return "/".join(parts)


def evaluate_gates(spec: dict, splits: dict) -> dict:
    rules = spec.get("rules") if isinstance(spec, dict) else None
    if not isinstance(rules, list) or not rules:
        raise InvalidInput("gate spec needs a non-empty 'rules' list")
    results = []
    for rule in rules:
        if not isinstance(rule, dict) or "metric" not in rule:
            raise InvalidInput(f"rule {rule!r} needs a 'metric'")
        unknown = set(rule) - RULE_KEYS
        if unknown:
            raise InvalidInput(f"rule {rule}: unknown keys {sorted(unknown)}")
        comps = [c for c in COMPARATORS if c in rule]
        if not comps:
            raise InvalidInput(f"rule {rule}: needs one of {list(COMPARATORS)}")
        for c in comps:
            if not isinstance(rule[c], (int, float)) or isinstance(rule[c], bool):
                raise InvalidInput(f"rule {rule}: {c} must be a number")
        e = lookup(splits, rule)
        base, cand = value_of(e, "baseline"), value_of(e, "candidate")
        needs_base = any(c in ("max_increase", "min_decrease") for c in comps)
        if cand is None or (needs_base and base is None):
            raise InvalidInput(f"rule {rule_label(rule)}: metric has no rows in the "
                               f"{'candidate' if cand is None else 'baseline'} report")
        checks = []
        for c in comps:
            limit = rule[c]
            if c == "max":
                ok, observed = cand <= limit, cand
            elif c == "min":
                ok, observed = cand >= limit, cand
            elif c == "max_increase":
                ok, observed = cand - base <= limit, cand - base
            else:  # min_decrease
                ok, observed = base - cand >= limit, base - cand
            checks.append({"comparator": c, "limit": limit, "observed": observed, "passed": bool(ok)})
        results.append({"rule": rule_label(rule), "spec": rule, "baseline": base, "candidate": cand,
                        "checks": checks, "passed": all(ch["passed"] for ch in checks)})
    return {"results": results, "passed": all(r["passed"] for r in results)}


# --------------------------------------------------------------------------- rendering

def fmt_rate(b) -> str:
    if not isinstance(b, dict) or b.get("rate") is None:
        return "–"
    lo, hi = b["ci95"] if b.get("ci95") else (None, None)
    ci = f" [{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else ""
    return f"{100 * b['rate']:.1f}{ci} (n={b['n']})"


def fmt_scalar(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def fmt_delta(e: dict) -> str:
    if e["delta"] is None:
        return "–"
    if e["kind"] == "rate":
        ci = e.get("delta_ci95")
        tail = f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else ""
        return f"{100 * e['delta']:+.1f} pp{tail}"
    return f"{e['delta']:+.3f}"


def row(name: str, e: dict) -> str:
    f = fmt_rate if e["kind"] == "rate" else fmt_scalar
    return f"| {name} | {f(e['baseline'])} | {f(e['candidate'])} | {fmt_delta(e)} |"


def to_markdown(doc: dict) -> str:
    b, c = doc["baseline"], doc["candidate"]
    L = ["# MuseBench comparison", "",
         f"Section: `{doc['section']}`. Seen field: `{doc['seen_field']}`. Cohort: "
         + ", ".join(f"{k}={v}" for k, v in doc["cohort"].items()) + ".", "",
         f"- Baseline: `{b['report']}` (threshold {b['threshold']}, {b['decision_rule']}; "
         f"{b['coverage']['scored']} scored of {b['coverage']['eval_rows']})",
         f"- Candidate: `{c['report']}` (threshold {c['threshold']}, {c['decision_rule']}; "
         f"{c['coverage']['scored']} scored of {c['coverage']['eval_rows']})", "",
         "Rates in %, Wilson 95 % interval in brackets; deltas are candidate - baseline in points with "
         "Newcombe's interval (independent-sample, conservative on a shared cohort).", ""]
    for split in SPLITS:
        s = doc["splits"][split]
        L += [f"## {split}", "", "| Metric | baseline | candidate | delta |", "|---|---|---|---|"]
        L += [row(m, s["metrics"][m]) for m in RATE_METRICS + SCALAR_METRICS]
        L += ["", "| Generator (clean FNR) | baseline | candidate | delta |", "|---|---|---|---|"]
        L += [row(g, v["clean_fnr"]) for g, v in s["per_generator"].items()]
        L.append("")
    gates = doc.get("gates")
    if gates:
        L += ["## Gates", "", f"Spec: `{gates['spec']}` (sha256 {gates['spec_sha256'][:12]}). "
              f"Result: **{'PASS' if gates['passed'] else 'FAIL'}**.", "",
              "| Rule | comparator | limit | observed | result |", "|---|---|---|---|---|"]
        for r in gates["results"]:
            for ch in r["checks"]:
                L.append(f"| {r['rule']} | {ch['comparator']} | {ch['limit']} | {ch['observed']:.4f} | "
                         f"{'pass' if ch['passed'] else 'FAIL'} |")
        L.append("")
    return "\n".join(L)


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- main

def build_comparison(baseline_path: Path, candidate_path: Path, section: str = "primary",
                     gates_path: Path | None = None) -> dict:
    base_r, cand_r = load_json(baseline_path, "baseline report"), load_json(candidate_path, "candidate report")
    base_b, cand_b = section_of(base_r, section), section_of(cand_r, section)
    base_c, cand_c = cohort_identity(base_r, base_b), cohort_identity(cand_r, cand_b)
    if base_c != cand_c:
        raise InvalidInput(f"cohort mismatch: baseline {base_c} vs candidate {cand_c}")
    base_s, cand_s = seen_field_of(base_r), seen_field_of(cand_r)
    if base_s != cand_s:
        raise InvalidInput(f"seen-field mismatch: baseline {base_s!r} vs candidate {cand_s!r}; "
                           "build both reports with the same report.py --seen-field")
    doc = {"schema": SCHEMA, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "section": section, "cohort": base_c, "seen_field": base_s,
           "baseline": side_info(baseline_path, base_r, base_b),
           "candidate": side_info(candidate_path, cand_r, cand_b),
           "splits": compare_bodies(base_b, cand_b), "gates": None}
    if gates_path is not None:
        spec = load_json(gates_path, "gate spec")
        doc["gates"] = {"spec": str(gates_path), "spec_sha256": sha256_file(gates_path),
                        **evaluate_gates(spec, doc["splits"])}
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two MuseBench reports on the same cohort and evaluate gates.")
    ap.add_argument("--baseline", type=Path, required=True, help="baseline musebench_report.json")
    ap.add_argument("--candidate", type=Path, required=True, help="candidate musebench_report.json")
    ap.add_argument("--out", type=Path, required=True, help="directory for comparison.json and comparison.md")
    ap.add_argument("--gates", type=Path, default=None, help="gate spec JSON (rules fixed before the run)")
    ap.add_argument("--section", choices=("primary", "strict"), default="primary",
                    help="report section to compare (default: primary)")
    args = ap.parse_args(argv)
    try:
        doc = build_comparison(args.baseline, args.candidate, args.section, args.gates)
    except InvalidInput as exc:
        print(f"compare: {exc}", file=sys.stderr)
        return EXIT_INVALID
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_write(args.out / "comparison.json", json.dumps(doc, indent=1) + "\n")
    atomic_write(args.out / "comparison.md", to_markdown(doc) + "\n")
    gates = doc["gates"]
    if gates is None:
        print(f"compare: no gate spec; wrote {args.out / 'comparison.md'}")
        return EXIT_PASS
    failed = [r["rule"] for r in gates["results"] if not r["passed"]]
    print(f"compare: gates {'PASS' if not failed else 'FAIL'} ({len(gates['results']) - len(failed)}/"
          f"{len(gates['results'])} rules); wrote {args.out / 'comparison.md'}")
    for name in failed:
        print(f"  failed: {name}", file=sys.stderr)
    return EXIT_GATE_FAIL if failed else EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
