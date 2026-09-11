# Reproducing the LaTeX source archive

Status: local **draft** archive, not a completed benchmark or an arXiv submission.
The ten-page manuscript now includes the completed eight-checkpoint primary
comparison and full public-score statistical reproduction, alongside the
historical audit, input-path diagnostics, and completed eight-checkpoint
transport evaluation. Model-release verification remains incomplete; public
results ZIP and notebook portability checks have passed.

Revision 14 evidence: `out/v1.2_latex_source_draft_v14_260905`, ZIP 32,072 bytes,
SHA-256 `de361af7fbb020d253e8241b97da8330fbfc284019ffd29a3d1cb52194e0736b`.
Fresh extracted-source compilation matched all ten pages in layout text and
raster hashes. Page 8 was visually checked for the transport table.
Five existing underfull warnings remain; no overfull or unresolved references.
The package includes `generated/full_rc2/transport_table.tex`; no arXiv server
compile or upload was performed.

## Contents and build

`paper/source_files.txt` is the explicit source inventory. It currently contains
eleven files: `main.tex`, `refs.bib`, `main.bbl`, seven generated LaTeX fragments,
and one generated PDF figure. Update this list when additional result tables and
figures are incorporated. The packager compares the declared inventory with
the actual compiler dependency trace and rejects missing or unused payloads.

From the repository root:

```bash
python -m paper.package_source \
  --compiler .tools/tectonic-0.17.0/tectonic \
  --output out/my-latex-source-check
```

Python's standard library suffices for this packaging command. It additionally
requires Tectonic 0.17.0 and the `pdfinfo`, `pdftotext` and `pdftoppm` executables.
Do not reuse an output directory. The command records compiler/tool hashes,
source hashes, the original PDF hash and every verification process's actual
stdout/stderr. A failed command leaves evidence but does not write a success
summary. The source archive's filename carries `DRAFT` while the manuscript
contains its draft notice; absence of that notice is not a completion gate.

The ZIP is deterministic: sorted names, fixed timestamp and file permissions.
`main.tex` is at its root, retaining the `generated/` and `figures/` paths.
The archive excludes the compiled main PDF, build logs, intermediate files,
Python scripts, evidence JSON, private mappings, model weights and audio.
The generated `.bbl` is included alongside the `.bib` source. Hash inventories
and process evidence stay outside the source ZIP.

## Independent execution evidence

The command validates names, regular file types, exact inventory and byte hashes
before using a fresh extraction. It clears custom TeX search paths and runs
from the extracted root with Tectonic's `--only-cached --untrusted` options.
No original manuscript search path or newly downloaded package is needed;
the already populated TeX resource cache is reused. This is not a fresh TeX
installation. The [official Tectonic CLI](https://tectonic-typesetting.github.io/book/latest/ref/v1cli.html)
documents these flags and the emitted dependency rules.

The observed `--outdir` dependency trace prepended the output directory even
to inputs read from the manuscript tree. The verification therefore compiles
**without `--outdir`**, inside the extracted root, and checks the resulting
relative dependencies. It does not treat the earlier output-prefixed trace as
proof that nonexistent files were read from that output directory.

Verification includes:

- No unresolved references/citations, TeX errors, or overfull boxes. Existing
  underfull warnings are retained in the summary.
- Figure and compiled-PDF JavaScript state checked by `pdfinfo`.
- Exact original versus rebuilt page count and layout-text SHA-256.
- Actual rasterization of **every page**, with matching PNG hashes at 72 dpi.
- Unchanged original sources/PDF and unchanged packaged inputs, including `.bbl`.

The prior revision-12 draft ZIP is 29,475 bytes, SHA-256
`99fb41240a3b4bc09d53f4813b85549b5bbbac93527bb6f9f3c6ad446018452f`.
Evidence is under `out/v1.2_latex_source_draft_v12_260905`; an independent
`v12_repeat` build has identical archive bytes. This revision retains the final
renderer admission guard: validated full public predictions must bind all eight
model statistical files and the paired-difference file. It retains the implemented
native DeepFense condition, preserved version-1 codec failure and actual
32-source version-2 compatibility checks, and adds the completed CPU full
evaluation: 2578 scores and one input failure, with low AI detection as well as
low false positives. The aligned full transport evaluation is now complete for
all eight checkpoints (406/406 each), and its summary table is included in the
current source inventory.
It retains the observed ONNX full-parity tolerance failure and targeted GPU/CPU/FFT/DFT decomposition.
It retains the bounded DeepFense same-input resampler diagnostic and interpretation hold. See
[the input-path audit](DEEPFENSE_INPUT_PATH_AUDIT_v1.2.md). A separately
versioned full input-path evaluation is now measured; the eight-model comparison,
transport evaluation, and saved-score portability checks are complete, while
model-release checks remain.
It retains the actual 34-recording ONNX source-smoke; the full v2 run is incomplete
with an observed tolerance failure, not an available full-parity success.
All nine pages match
the original manuscript's text and raster images. PDF binary hashes need not
match because build metadata differs; no binary-equality claim is made.
Changed pages 6 through 9 were visually inspected; pages 1–5 are raster-identical
to the prior verified draft. Five underfull warnings
(including long bibliography URLs) remain visible in the evidence; there are
no overfull boxes, undefined references or citations. The main PDF SHA-256 is
`89057b10b97eb844cdbbdb698f3a5930c3a068dee3c5285e36bef5a5a41cc05a`.
The last measured non-slow suites report 306 passed/four skips in the ONNX export environment,
304/four in the original evaluation environment, and 300/eight in RTX_ENV.
DeepFense upstream fixtures skip where the package is absent; ONNX-only checks
skip outside its export environment. The slow real-model
test was measured separately, not run concurrently with the active GPU queue.

## arXiv boundary

The [arXiv TeX instructions](https://info.arxiv.org/help/submit_tex.html)
require paths to work from the submission root and exclude unused/build files.
They support generated `.bbl` files named for the main source, and also `.bib`
processing. Necessary figures must already use a supported format.

As checked on 2026-09-05, arXiv supports TeX Live 2023 and 2025 (default), with
both PDFLaTeX and XeLaTeX options. Tectonic is **not** the arXiv server engine;
this local check is not server-compilation evidence. See the
[current engine/version guidance](https://info.arxiv.org/help/faq/texlive.html).
Actual submission still requires inspecting arXiv's generated PDF and explicit
authorization to upload. No upload or account action is performed here.

Before preparing a final candidate: generate the remaining full result figures
from validated reports,
update the LaTeX narrative and inventory, repeat this entire source check, and
review every final page. Packaging success cannot certify scientific claims,
recording access, training independence or redistribution rights.
