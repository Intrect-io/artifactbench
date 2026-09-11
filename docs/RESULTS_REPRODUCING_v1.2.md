# Reproducing statistics from public predictions

This is separate from metadata verification and fresh detector inference.
The eight-model rc2 primary evaluation and full saved-score reconstruction
completed on 2026-09-05. `out/v1.2_public_results_rc2_260905` contains 2,579
outcomes per model, including the two retained no-score failures. Its summary
SHA-256 is `9002708c18eff81f93f0e7afab2c00797f24bf0bf64f2e924789a5c5d24c5355`.
Using `.venv-results`, `out/v1.2_public_statistics_reproduced_rc2_260905`
reproduced all nine reference statistical files byte-for-byte, including the
2,000-replicate intervals and all 28 paired model comparisons. Both statistical
summary hashes are `5e7552e88952fb26d1e6001b742f4e98529c5ed8f295cf88209af667d8bdf168`.
This is saved-score reconstruction, not fresh detector inference. The full rc2
archive/notebook portability checks and transport results have now passed.
Historical 32-entry/model smoke remains separate from the full rc2 evidence.

## Export completed runs

The following export was run after all eight rc2 runs and their 2,000-replicate
reports completed:

```bash
python -m artifactbench.v12.public_results export \
  --release out/v1.2_frozen_rc2_260905 --runs out/v1.2_full_rc2_260905 \
  --run-override deepfense=out/v1.2_deepfense_native_full_v2_260905 \
  --statistics out/v1.2_statistics_rc2_260905 --output out/v1.2_public_results_rc2_260905
```

The exporter requires the report to reference these exact run summaries and
release files. It verifies completed row hashes and retains entry IDs, outcomes,
probabilities, fixed operating decisions, and original row hashes. Failed entries
retain their error class, not private error messages or tracebacks. Code and
weight names become unambiguous repository-relative identities, preserving exact
hashes and upstream revisions. No audio, weights, raw pages, credentials, or
local path maps are exported.

Final export and public-result loading now reject the original DeepFense
adapter condition: its completed run remains an audit, not the corrected
comparison. They require `deepfense-native-soxr/2` in the identity and all
successfully prepared rows. Input failures keep their no-score coverage.
The same explicit override must have been used to generate the reference
statistics. Actual condition-check evidence: the original 2,579-row DeepFense
run is rejected; the native version-2 smoke's 32 rows pass the condition check
only, not full-result admission. Historical `--allow-smoke` remains available.
Final `paper.render_results` now requires `--predictions`: it loads validated
full public predictions and requires all nine statistical file hashes to match.
The rendered provenance records the public prediction summary hash. A generic
private report may still be generated as an audit, but it cannot enter this
final rendering path without matching the admitted public predictions. Do not
edit the pinned `report.py` while native primary or aligned transport inference
still depends on its hash.

Wall times are descriptive, not controlled throughput comparisons.
`peak_gpu_bytes` is cumulative PyTorch allocator peak, not isolated per-model
memory. Fakeprint logistic regression runs on CPU even when the requested
runner device is recorded as CUDA.

## Recompute numerical reports

Only Python and NumPy are needed for saved-score recalculation:

```bash
python -m artifactbench.v12.public_results reproduce \
  --release out/v1.2_frozen_rc2_260905 --predictions out/v1.2_public_results_rc2_260905 \
  --output out/my-public-statistics
```

All eight models must have identical fixed membership. Public labels, source
cells, audio hashes, grouping, per-entry RNG seeds, and explicit coverage are
checked. Recalculation uses the same `generate_reports` implementation as the
private-run report. It must reproduce **all nine statistical JSON files**
byte-for-byte: eight model reports plus all 28 paired model comparisons.
An empty or incomplete reference inventory is rejected.

The measured reporting environment uses NumPy 2.1.3. Percentile intervals use
the default [linear quantile convention](https://numpy.org/doc/2.1/reference/generated/numpy.quantile.html).
No quantile method, threshold, grouping, or denominator was changed for export.
Every original historical smoke statistical file and its summary stayed
byte-identical after separating the shared report computation.

## Execute the results notebook

Use an independent CPU-only environment, not a live inference environment:

```bash
python3.12 -m venv .venv-results
.venv-results/bin/python -m pip install -r requirements-results.lock
.venv-results/bin/python -m artifactbench.v12.execute_results_notebook \
  --notebook notebooks/v1.2_results_reproduction.ipynb \
  --release out/v1.2_frozen_rc2_260905 --predictions out/v1.2_public_results_rc2_260905 \
  --output out/my-results-notebook
```

All six cells must actually run in the requested kernel. Source notebooks have
no prefilled outputs. Errors preserve executed cells and stop the run. A success
summary is written only after all nine original statistical files match. The
executor follows the [official NBClient interface](https://nbclient.readthedocs.io/en/latest/client.html).

`--allow-smoke` is an explicit implementation test mode. It cannot produce a
final status or write smoke artifacts under `paper/`. Historical rc1 smoke is
labelled as such and cannot replace the corrected rc2 release or its full results.

`check_results_portability` copies only the ten required source/notebook files,
four public release files, and nine public-prediction files into a fresh
temporary directory. It removes `PYTHONPATH`/`PYTHONHOME` and actually runs the
results notebook there. The historical smoke passed this 23-file copy check and
all six cells with the same nine statistical-file hashes; no original source
or data path was needed. The isolated dependency environment is reused.

```bash
.venv-results/bin/python -m artifactbench.v12.check_results_portability \
  --release out/v1.2_frozen_rc2_260905 --predictions out/v1.2_public_results_rc2_260905 \
  --notebook-python .venv-results/bin/python --output out/my-results-portability
```

This command still needs the completed full rc2 export. A single completed
checkpoint report is an intermediate observation, not an eight-model notebook.

## Build and independently execute the public-score ZIP

After the full export and successful public-score reproduction, package only
the allowlisted source, public metadata/predictions, and reproduced reports:

```bash
python -m artifactbench.v12.package_results build \
  --release out/v1.2_frozen_rc2_260905 --predictions out/v1.2_public_results_rc2_260905 \
  --reproduction out/my-public-statistics --output out/v1.2_results_bundle_rc2_260905
.venv-results/bin/python -m artifactbench.v12.check_results_archive \
  --archive out/v1.2_results_bundle_rc2_260905/artifactbench-1.2-rc2-results.zip \
  --notebook-python .venv-results/bin/python --output out/my-results-zip-portability
```

The archive has 40 payload files plus an exact hash inventory. Its build requires
the successful nine-file reproduction to match the bundled source hashes,
release, and public-prediction summary. Full mode requires the pinned rc2 and all
eight complete runs. Smoke mode is explicit in status, inventory, and archive
name; it cannot be verified as full or written under the manuscript tree.

Extraction rejects duplicate names, path traversal, non-regular files, unlisted
payloads, and oversized contents before writing. These checks follow the
[Python 3.12 ZIP extraction guidance](https://docs.python.org/3.12/library/zipfile.html#zipfile.ZipFile.extractall).
The independent check actually runs both the packaged verifier and six-cell
notebook from the extracted source root, removes `PYTHONPATH`/`PYTHONHOME`, and
checks that all packaged inputs remain unchanged. This reuses the isolated
dependency environment; it does not claim a new installation or fresh inference.
Only run this tool on trusted local candidates: an included hash inventory is
not a publisher signature and cannot authenticate arbitrary executable source.

Measured historical smoke evidence (not full rc2 results):

- `out/v1.2_results_bundle_smoke_260905` and the separate `_repeat_260905` build
  produced identical 618,202-byte archives, SHA-256
  `dd0e680262c97fbc8b4daab12c0e574811f31dff4bb9f2bf507081ede16a763c`.
- `out/v1.2_results_zip_portability_smoke_260905` passed extraction, packaged
  verification, all six notebook cells, and byte-identical reconstruction of all
  nine statistical JSON files. The archive contains the actual historical
  32-entry/model scores, not fabricated benchmark data. No upload was made.

The archive's [standalone README](RESULTS_BUNDLE_README.md) documents extraction,
an external environment/output directory, and bytecode suppression so the
strict payload inventory stays clean.

## LaTeX linkage and limits

`paper/main.tex` and `paper/refs.bib` are the primary manuscript sources.
`paper/render_results.py` generates LaTeX tables and PDF figures from validated
completed reports. The final PDF and arXiv source archive must include those
actual outputs. Metadata verification or historical smoke cannot satisfy this.

Saved-score reproduction does not re-run detectors, establish recording access,
prove training independence, or grant third-party redistribution rights.
