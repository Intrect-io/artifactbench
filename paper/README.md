# ArtifactBench 1.2 benchmark paper

Status: working draft. Corrected data are frozen as v1.2-rc2 (2,579 evaluation entries /
2,576 identified recording groups). ArtifactNet, both SpecTTTra checkpoints,
AST-60s, DeepFense, third-party fakeprint LR and CLAM have completed their original
full runs and 2,000-replicate single-checkpoint
reports; the manuscript still shows the explicitly labelled first ArtifactNet
checkpoint. The full eight-checkpoint comparison,
transport results, and submission checks remain incomplete.
Do not submit this draft or cite its historical audit as a v1.2 leaderboard.
The original DeepFense run is now under an **interpretation hold**: a fixed
nine-file diagnostic shows that changing only the final resampler flips 8/9
decisions, while exact-input replay is bit-exact. The draft includes this
bounded finding, not a corrected population-level performance estimate. See
[the input-path audit](../docs/DEEPFENSE_INPUT_PATH_AUDIT_v1.2.md); a separately
versioned full input-path evaluation is required before the final comparison.
The native-input version 2 has passed 32/32 actual source-smoke inferences and
is now running on all 2,579 entries on CPU (two threads). Version 1's unsupported
WebM/Opus error is preserved; the new explicit AAC/Opus fallback does not change
any of the other 31 waveform/logit/probability results. FST is the final original
GPU run in progress. Neither running experiment is counted as complete.

ONNX publication is also on hold under the predeclared parity rule: one of
the first 205 full comparisons exceeds 1e-3. A same-input CPU/FFT/DFT diagnostic
separates a larger saved-GPU/fresh-CPU difference from an additional STFT-path
contribution. The manuscript reports both, without relaxing the tolerance.
A standard-ONNX-STFT copy is only a one-recording probe: it improves agreement
with the saved GPU score but worsens agreement with the CPU original, so it
has not been adopted. See [the ONNX evidence](../docs/ARTIFACTNET_ONNX_v1.2.md).

The draft is independent of the older ArtifactNet detector manuscript. It cites
that work, but does not copy its earlier sample counts, universal RVQ claims, or
all-model-unseen claims. The Deezer-method adapter uses a third-party `lofcz`
checkpoint; it must not be presented as the official Deezer detector/service.
The [source audit](../docs/PAPER_SOURCE_AUDIT_v1.2.md) separates literature
claims, pinned checkpoint descriptions and locally measured results. It records
the AIME conference citation, original AST architecture versus music fine-tune,
and the exact DeepFense EAT+Nes2Net recipe rather than treating the toolkit as
one detector.
The current transport methods also define same-staging codec increments and
four-view signed interactions. These clarify the existing eight-view experiment;
transport detector results are not available yet. See
[the statistical contract](../docs/STATISTICS_v1.2.md#matched-staging-transport-contrasts).

Generate the verified historical audit table and figure:

```bash
/home/unohee/RTX_ENV/bin/python paper/generate_historical.py
/home/unohee/RTX_ENV/bin/python paper/generate_construction.py
.venv-eval/bin/python -m paper.generate_checkpoint
```

Compile (local compiler is Tectonic 0.17.0; the downloaded archive SHA-256 was
verified against the official GitHub release asset digest):

```bash
../.tools/tectonic-0.17.0/tectonic --keep-logs --keep-intermediates main.tex
```

Run the compile command from `paper/`. The source uses standard LaTeX packages
and BibTeX rather than a conference template. The final arXiv bundle must include
the generated `.bbl`, figures, and tables, but not local data paths, cached HTML,
weights, auxiliary build files, or redistribution-restricted audio.

Remaining submission gates are listed in `../PLAN_v1.2.md`. In particular, a
successfully compiled PDF is not evidence that the benchmark is complete.

The draft's eight-file source ZIP now compiles after fresh extraction, without
the original source directory or new package downloads. All nine rebuilt pages
match the reference PDF's layout text and raster images. This uses Tectonic's
existing cache, not arXiv's TeX Live server. Source inventory, exact commands,
and remaining submission limits are in
[`LATEX_SOURCE_REPRODUCING_v1.2.md`](../docs/LATEX_SOURCE_REPRODUCING_v1.2.md).

`render_results.py` turns hash-bound completed statistical reports into LaTeX
tables, native TPR interval plots, AI/real source-rate heatmaps, and a full
metrics CSV. Final mode requires all eight models, 2,000 bootstrap
replicates, and `--predictions` pointing to the validated public prediction
export. Its nine reference statistics must match the rendered files exactly;
public admission checks include the corrected native DeepFense condition.
An explicit `--allow-smoke` wiring run is watermarked and cannot
write under the manuscript tree. The historical rc1 smoke is a rendering test,
not final data or performance evidence:

```bash
python -m paper.render_results --statistics out/v1.2_statistics_rc2_260905 \
  --predictions out/v1.2_public_results_rc2_260905 \
  --release out/v1.2_frozen_rc2_260905 --output paper/generated/results
```

Run this command from the repository root **after** the full statistical report
and validated public prediction export exist. Rendering uses the installed Matplotlib 3.11.1 API; interval segments
follow [Axes.hlines](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hlines.html)
and heatmaps follow the [official annotated-heatmap example](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html).
No synthetic prediction or example table is substituted when results are missing.
Figure legends are placed outside the data axes following the
[official legend guide](https://matplotlib.org/stable/users/explain/axes/legend_guide.html).
The historical smoke tables also passed a separate LaTeX layout compile outside
the manuscript tree. The current nine-page draft includes methods and an
explicitly labelled first-completed-checkpoint section, not a final leaderboard.
Its new table/numbers come from `out/v1.2_checkpoint_artifactnet_rc2_260905`:
raw-0.5 legacy F1 0.9887, Suno v5.5 TPR 5.0%, Udio 2026 TPR 100%. The existing
ArtifactNet operating stack reaches 7.5% on Suno. All native Suno inputs are
AAC and Udio inputs are MP3, so this contrast is not a version-only causal
estimate. A degenerate all-success bootstrap interval is explicitly qualified.

The user explicitly reconfirmed LaTeX as the manuscript format on 2026-09-05.
Keep `main.tex`/`refs.bib` as primary sources; generated table fragments and PDF
figures accompany them in the eventual arXiv source archive. The draft now
distinguishes metadata, saved-score, and fresh-inference reproduction. See
`../docs/RESULTS_REPRODUCING_v1.2.md` for the prediction export/notebook path.
The public-score ZIP also passed two byte-identical builds and actual six-cell
execution after fresh extraction, reproducing all nine statistical files. This
uses historical smoke scores and the existing isolated dependency environment;
the full rc2 result archive and fresh-inference checks remain separate gates.
