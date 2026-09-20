# ArtifactBench 1.2 release readiness (rc2)

Checked 2026-09-05. This document records what is actually verified; it is not
an upload authorization or a claim of universal generator generalization.

## Verified

- Frozen rc2 manifest: `out/v1.2_frozen_rc2_260905/`
  (`manifest.public.json` SHA-256
  `feab7c4c3d037919fd784dc40e46199470088abde532c5181e34573f20dceefd`).
- Eight fixed-checkpoint primary evaluations: 2,579 attempted per model;
  explicit model/input failures retained in coverage.
- Eight controlled transport evaluations: 406/406 per checkpoint, exit 0;
  transport statistics generated with 2,000 cluster-bootstrap replicates.
- Public results bundle: `out/v1.2_results_bundle_rc2_260905/`, SHA-256
  `2e645fef5c19ab02dc4a547e4ed55f8cd9b499c350bdea0d03b2f6bc729a6b87`.
  Fresh extraction, 40-file verification, and six-cell result notebook passed.
- LaTeX source archive v14: `out/v1.2_latex_source_draft_v14_260905/`, SHA-256
  `de361af7fbb020d253e8241b97da8330fbfc284019ffd29a3d1cb52194e0736b`.
  Ten-page independent compilation passed with no overfull boxes or unresolved
  references; existing underfull warnings are recorded.
- ArtifactNet standard-STFT ONNX qualification: 34/34 attempted, 33 scored
  references comparable, maximum GPU error `0.0005109871`, maximum factory-CPU
  error `0.0002899710`, zero tolerance failures and zero raw-0.5 flips.

## Not yet verified

- Full 2,579-entry ONNX parity for the standard-STFT candidate. Attempts were
  stopped before the first record under host memory pressure; no score is
  inferred from those runs.
- Public redistribution or model-weight licensing. The runtime license does
  not grant rights to third-party audio or weights.
- arXiv server compilation and submission. No account, upload, or publication
  action has been performed.

The current paper is therefore a reproducible rc2 benchmark/methods draft with
validated public saved-score results, not a finished ONNX model release or an
arXiv submission.
