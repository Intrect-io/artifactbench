# ArtifactBench: public-score reproduction bundle

Check `BUNDLE_MANIFEST.json` before interpreting results. A bundle named
`artifactbench-results-SMOKE-ONLY` has `smoke_only: true`: it is historical rc1
32-entry/model software wiring, not a corrected v1.2 result or a leaderboard.
The full candidate is named `artifactbench-1.2-rc2-results` and requires all
eight completed rc2 runs, all 2,579 entries per model, and 2,000 bootstrap
replicates. Neither name means it has been uploaded or published.

This archive includes public metadata, fixed-checkpoint scores, explicit model
failures, reference statistical reports, and source code for reconstructing all
nine statistical files (eight model reports and 28 paired model comparisons).
It contains no audio, weights, raw pages, lyrics, prompts, private path maps,
or original failure tracebacks. Saved-score reproduction is not fresh inference.

## Verify and reproduce

Extract your trusted local candidate and change into its top-level directory.
Use Python 3.12 and a dedicated environment **outside** that directory so the
strict inventory does not confuse installed packages with archive payload:

```bash
export PYTHONDONTWRITEBYTECODE=1
python3.12 -m venv ../artifactbench-results-env
../artifactbench-results-env/bin/python -m pip install -r requirements-results.lock
../artifactbench-results-env/bin/python -m artifactbench.v12.package_results verify --bundle .
../artifactbench-results-env/bin/python -m artifactbench.v12.execute_results_notebook \
  --notebook notebooks/v1.2_results_reproduction.ipynb \
  --release release --predictions predictions --output ../results-notebook-execution
```

For the explicitly labelled historical smoke archive, append `--allow-smoke`
to **both** verification and notebook commands. Full mode rejects smoke data.
Use a new output directory for each execution. Avoid writing Python bytecode
inside the archive: set `PYTHONDONTWRITEBYTECODE=1` before running these commands.

All six notebook cells actually execute in the requested environment. They
load the public records, verify exact membership and hashes, recalculate
statistics with NumPy 2.1.3, compare all nine original JSON files byte-for-byte,
and display cohort metrics and paired differences. No GPU or detector package
is needed. Dependency installation may need network access; computation does not.

Alternatively, after setting up the same environment, use the command-line
reproducer with `--release release --predictions predictions`:
`python -m artifactbench.v12.public_results reproduce --output ../recomputed-statistics`.
Append `--allow-smoke` only for smoke data.

## Limits and integrity

Public hashes detect changes relative to this inventory; they do not sign or
authenticate the publisher. Do not execute source from an untrusted archive.
The source membership, inference coverage, failure policy, score direction,
operating thresholds, grouping, and bootstrap method are unchanged in export.
Undefined metrics remain undefined, not zero. All real controls are legacy:
native AI-only cohorts cannot estimate an independent contemporary FPR.
Source/version differences may be associated with codec or transport differences.
Official demonstration clips are descriptive and are not pooled into headline
native results. Per-entry runtime is descriptive; recorded GPU memory is a
cumulative allocator peak, not isolated model memory.

Exact saved-score reproduction does not establish source recording availability,
training independence, full-model portability, causal generator comparisons,
or rights to redistribute third-party content. `LICENSE` applies to supplied
project code, not to audio, model weights, song pages, or upstream assets.
