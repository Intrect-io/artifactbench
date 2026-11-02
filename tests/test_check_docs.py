# Created: 2026-11-03
# Purpose: Cover scripts/check_docs.py (link resolution and documented CLI flags).
# Dependencies: pytest
# Test Status: passing
import textwrap

from scripts import check_docs as cd


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_broken_relative_link_is_reported(tmp_path):
    write(tmp_path, "docs/OK.md", "ok\n")
    doc = write(tmp_path, "docs/A.md", """\
        [good](OK.md) [anchor](OK.md#x) [url](https://example.com/x) [self](#top)
        [bad](MISSING.md) [dir](../docs)
        """)
    errors = cd.check_links(doc, root=tmp_path)
    assert len(errors) == 1 and "MISSING.md" in errors[0] and "A.md:2" in errors[0]


def test_links_inside_code_fences_are_ignored(tmp_path):
    doc = write(tmp_path, "README.md", "```\n[x](nope.md)\n```\n")
    assert cd.check_links(doc, root=tmp_path) == []


def test_root_relative_and_image_links(tmp_path):
    write(tmp_path, "a/b.md", "x\n")
    doc = write(tmp_path, "docs/A.md", "[r](/a/b.md) ![img](missing.png)\n")
    assert cd.check_links(doc, root=tmp_path) == []  # images are not checked


def test_extract_commands_joins_continuations_and_skips_prose():
    text = "prose python -m x --no\n```bash\npython -m pkg.mod run \\\n  --a 1 \\\n  --b\nls\n```\n"
    assert cd.extract_commands(text) == [(3, ["python", "-m", "pkg.mod", "run", "--a", "1", "--b"])]


def test_program_of():
    assert cd.program_of(["python", "-m", "pip", "install"]) is None
    assert cd.program_of(["python", "-m", "pkg.mod", "--x"]) == ("-m", "pkg.mod")
    assert cd.program_of(["python", "-m", "pkg.mod", "build", "--x"]) == ("-m", "pkg.mod", "build")
    assert cd.program_of(["python", "scripts/tool.py", "--x"]) == ("scripts/tool.py",)
    assert cd.program_of(["python", "-c", "print()"]) is None


def test_flags_accepted_and_rejected(tmp_path):
    write(tmp_path, "tool.py", """\
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--known")
        p.parse_args()
        """)
    good = write(tmp_path, "good.md", "```bash\npython tool.py --known 1\n```\n")
    bad = write(tmp_path, "bad.md", "```bash\npython tool.py --known 1 \\\n  --unknown 2\n```\n")
    assert cd.check_flags(good, root=tmp_path) == []
    errors = cd.check_flags(bad, root=tmp_path)
    assert len(errors) == 1 and "--unknown" in errors[0]


def test_subcommand_flags_are_checked_against_the_subcommand(tmp_path):
    write(tmp_path, "tool.py", """\
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd")
        sub.add_parser("build").add_argument("--out")
        sub.add_parser("verify").add_argument("--bundle")
        p.parse_args()
        """)
    doc = write(tmp_path, "d.md", "```bash\npython tool.py build --out x\npython tool.py build --bundle y\n```\n")
    errors = cd.check_flags(doc, root=tmp_path)
    assert len(errors) == 1 and "--bundle" in errors[0] and "d.md:3" in errors[0]


def test_unrunnable_program_is_reported_not_ignored(tmp_path):
    doc = write(tmp_path, "d.md", "```bash\npython missing_tool.py --x\n```\n")
    errors = cd.check_flags(doc, root=tmp_path)
    assert len(errors) == 1 and "cannot verify" in errors[0]


def test_repository_documentation_links_resolve():
    errors = [e for p in cd.doc_files() for e in cd.check_links(p)]
    assert errors == []
