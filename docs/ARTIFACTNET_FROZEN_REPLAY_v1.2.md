# ArtifactNet original-source replay

The live ArtifactNet tree changed after the completed primary evaluation.
Do not revert that tree or relax the transport runner's identity checks.

On 2026-09-05, all 34 original Python files were restored from the primary
run's content-addressed snapshots into
`out/v1.2_artifactnet_frozen_source_260905`. Each file matches the original
identity; all three weight files also still match. The original HPSS fallback
uses PyTorch, not the newly added compiled-kernel alternatives.

An actual Bubblewrap preflight overlaid that directory read-only at
`/home/unohee/dev/ArtifactNet/src`. Inside the namespace, `build_pinned` returned
exactly the original `model_assets`, including paths and source hashes.
No primary prediction or user source was changed. This preflight did not run
model inference and does not establish numerical replay equivalence.

For the next inference, after the other GPU benchmark has stopped, retain the
original RTX_ENV interpreter, offline flags, two threads and 6/8GiB cgroup
limits with swap disabled. The namespace command prefix is:

```bash
bwrap --ro-bind / / --dev-bind /dev /dev --proc /proc --tmpfs /tmp \
  --ro-bind /home/unohee/dev/artifactbench/out/v1.2_artifactnet_frozen_source_260905 /home/unohee/dev/ArtifactNet/src \
  --bind /home/unohee/dev/artifactbench/out/v1.2_artifactnet_transports_frozen_rc2_260905 /home/unohee/dev/artifactbench/out/v1.2_artifactnet_transports_frozen_rc2_260905 \
  -- /home/unohee/RTX_ENV/bin/python -B -m artifactbench.v12.run_transports \
  --model artifactnet --device cuda \
  --snapshots out/v1.2_baseline_assets_v2_260905/snapshots.json \
  --release out/v1.2_frozen_rc2_260905 --prepared out/v1.2_transports_rc2_260905 \
  --reference-run out/v1.2_full_rc2_260905/artifactnet \
  --output out/v1.2_artifactnet_transports_frozen_rc2_260905
```

The output directory has been created empty. The prefix alone does not impose
cgroup limits or set the required environment; launch it through the measured
user-systemd workflow. Do not bind the live source writable. `/dev` is required:
without it, the actual Torch import failed opening `/dev/urandom`. A plain
`unshare --user` failed on this host, but Bubblewrap's actual preflight passed.
Any further cache-write failure must be diagnosed specifically, not resolved by
making the source writable. Preserve all failed runs and compare original-view
scores after execution before interpreting codec differences.

The first full launch failed before inference with CUDA error 304. Adding
`--proc /proc` passed an actual `torch.cuda.init()` probe and device-name read;
the corrected full launch uses that option. A successful source-import preflight
alone does not establish GPU accessibility inside the namespace.

## Completed replay evidence (2026-09-05 20:47)

The corrected unit completed 406/406 scores with exit0, peak RAM5.6G and no
swap. `out/v1.2_artifactnet_transports_frozen_rc2_260905/summary.json` SHA-256:
`ab1c1b8f54247d0bad025b69bf36c4f0afa5ba724fb0db60100f04cbdb35bf01`.
All row hashes and fixed membership passed the 2,000-bootstrap statistics
command. All 53 original-view scores matched the primary run exactly.
Statistics: `out/v1.2_artifactnet_transport_statistics_260905/statistics.json`,
SHA-256 `e0c997fa4e9f9d7ca5042d8210f9ed127fecfe8304c34fa84c4b00e636b35378`.
This proves this transport replay, not full ONNX parity or changed-code equivalence.
