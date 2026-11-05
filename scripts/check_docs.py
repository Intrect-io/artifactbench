#!/usr/bin/env python3
# Created: 2026-11-03
# Purpose: Verify documentation: every relative Markdown link resolves, and every CLI
#          flag shown in a fenced command is accepted by the program it is passed to.
# Dependencies: standard library only (documented programs are run with --help).
# Test Status: tests/test_check_docs.py
"""Documentation checker.

Link check: relative ``[text](target)`` links in README.md, NOTICE.md, RESULTS_8WAY.md, docs/*.md and
paper/README.md must point at an existing file or directory (anchors and URLs skipped).

Flag check: in fenced code blocks, commands of the form ``python -m pkg.mod ...`` or
``python path/to/script.py ...`` are run with ``--help`` and every documented
``--flag`` must appear in that program's help output.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_GLOBS = ("README.md", "NOTICE.md", "RESULTS_8WAY.md", "docs/*.md", "paper/README.md")

LINK_RE = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE_RE = re.compile(r"^(\s*)(```|~~~)")
FLAG_RE = re.compile(r"(?<![\w-])--[A-Za-z][\w-]*")
# Commands that are not project programs.
IGNORED_MODULES = {"pip", "venv", "pytest", "http.server"}


def doc_files(root: Path = ROOT) -> list[Path]:
    files: list[Path] = []
    for pattern in DOC_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    return files


def _strip_fences(text: str) -> list[tuple[int, str, bool]]:
    """Return (lineno, line, in_fence) for every line."""
    out, fence = [], None
    for n, line in enumerate(text.splitlines(), 1):
        m = FENCE_RE.match(line)
        if m:
            if fence is None:
                fence = m.group(2)
            elif m.group(2) == fence:
                fence = None
            out.append((n, line, True))
            continue
        out.append((n, line, fence is not None))
    return out


def check_links(path: Path, root: Path = ROOT) -> list[str]:
    errors = []
    for n, line, in_fence in _strip_fences(path.read_text(encoding="utf-8")):
        if in_fence:
            continue
        for target in LINK_RE.findall(line):
            if re.match(r"^([a-z][a-z0-9+.-]*:|#|//)", target, re.I):
                continue
            rel = target.split("#", 1)[0].split("?", 1)[0]
            if not rel:
                continue
            resolved = (root / rel.lstrip("/")) if rel.startswith("/") else (path.parent / rel)
            if not resolved.exists():
                errors.append(f"{path.relative_to(root)}:{n}: broken link -> {target}")
    return errors


def extract_commands(text: str) -> list[tuple[int, list[str]]]:
    """Commands from fenced blocks, with backslash continuations joined."""
    commands, pending, start = [], "", 0
    for n, line, in_fence in _strip_fences(text):
        if not in_fence or FENCE_RE.match(line):
            pending = ""
            continue
        stripped = line.strip().removeprefix("$ ")
        if not pending:
            start = n
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        full = (pending + stripped).strip()
        pending = ""
        if re.match(r"^python3?\s", full):
            commands.append((start, full.split()))
    return commands


def program_of(tokens: list[str]) -> tuple[str, ...] | None:
    """('-m', module[, subcommand]) or (script_path[, subcommand]), else None.

    A bare word right after the program is treated as a subcommand (argparse
    subparsers); its --help lists that subcommand's flags.
    """
    args = tokens[1:]
    if len(args) >= 2 and args[0] == "-m":
        if args[1] in IGNORED_MODULES:
            return None
        program, rest = ("-m", args[1]), args[2:]
    elif args and args[0].endswith(".py"):
        program, rest = (args[0],), args[1:]
    else:
        return None
    if rest and re.fullmatch(r"[a-z][a-z0-9_-]*", rest[0]):
        program += (rest[0],)
    return program


def help_flags(program: tuple[str, ...], root: Path = ROOT) -> tuple[set[str] | None, str]:
    cmd = [sys.executable, *program, "--help"]
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None, "timed out running --help"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return None, f"--help exited {proc.returncode}: {tail[0]}"
    return set(FLAG_RE.findall(proc.stdout + proc.stderr)), ""


def check_flags(path: Path, root: Path = ROOT, cache: dict | None = None) -> list[str]:
    cache = {} if cache is None else cache
    errors = []
    for n, tokens in extract_commands(path.read_text(encoding="utf-8")):
        program = program_of(tokens)
        if program is None:
            continue
        flags = {f.split("=", 1)[0] for f in tokens if FLAG_RE.fullmatch(f.split("=", 1)[0])}
        where = f"{path.relative_to(root)}:{n}"
        if program not in cache:
            cache[program] = help_flags(program, root)
        accepted, problem = cache[program]
        label = " ".join(program)
        if accepted is None:
            errors.append(f"{where}: cannot verify `{label}`: {problem}")
            continue
        for flag in sorted(flags - accepted):
            errors.append(f"{where}: `{label}` does not accept {flag}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-flags", action="store_true", help="only check links")
    args = parser.parse_args(argv)
    files = doc_files()
    errors: list[str] = []
    cache: dict = {}
    for path in files:
        errors += check_links(path)
        if not args.skip_flags:
            errors += check_flags(path, cache=cache)
    for e in errors:
        print(e)
    print(f"checked {len(files)} files: {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
