"""Helpers for reproducibly materializing model assets from Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path


def materialize_hf_onnx_bundle(
    repo_id: str,
    filename: str,
    revision: str,
    cache_root: str | Path | None = None,
) -> Path:
    """Download an ONNX file and adjacent external-data files into one directory.

    Loading an ONNX file directly from the Hub blob cache can fail when the graph
    references ``<filename>.data``: each cached object has a different resolved
    blob path. ``local_dir`` creates a normal, colocated bundle that ONNX Runtime
    can resolve safely.
    """
    from huggingface_hub import hf_hub_download

    root = Path(cache_root or Path.home() / ".cache" / "artifactbench" / "models")
    target = root / repo_id.replace("/", "--") / revision
    target.mkdir(parents=True, exist_ok=True)

    model_path = Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=target,
    ))
    external_name = filename + ".data"
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=external_name,
            revision=revision,
            local_dir=target,
        )
    except Exception as exc:
        # A graph without external data is valid. Only suppress the Hub's
        # explicit missing-file response; transport and authentication failures
        # must remain visible.
        if exc.__class__.__name__ not in {"EntryNotFoundError", "RemoteEntryNotFoundError"}:
            raise
    return model_path
