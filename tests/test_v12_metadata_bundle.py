"""번들 경계의 unit fixture; 데이터셋이나 추론 결과를 모사하지 않는다."""
import json
from pathlib import Path
import sys

import pytest

from artifactbench.v12.package_metadata import public_payload
from artifactbench.v12.verify_bundle import safe_member


@pytest.mark.parametrize('name', ('../x', '/absolute', 'a\\b', 'a//b', '.', 'a/../b', 'C:/private', 'a\0b'))
def test_unsafe_archive_member_rejected(name):
    with pytest.raises(ValueError, match='Unsafe'):
        safe_member(name)


def test_public_media_url_is_not_a_local_path(tmp_path):
    source = tmp_path/'unit.json'
    source.write_text(json.dumps({'source_url': 'https://example.org/media/audio.mp3'}))
    assert public_payload(source) == source.read_bytes()
    source.write_text(json.dumps({'source_url': '/media/private/audio.mp3'}))
    with pytest.raises(ValueError, match='local path'):
        public_payload(source)


def test_notebook_outputs_and_raw_content_rejected(tmp_path):
    source = tmp_path/'unit.ipynb'
    source.write_text(json.dumps({'cells': [{'cell_type': 'code', 'execution_count': 1, 'outputs': []}]}))
    with pytest.raises(ValueError, match='unexecuted'):
        public_payload(source)
    source.write_text(json.dumps({'lyrics': 'unit privacy marker'}))
    with pytest.raises(ValueError, match='Private/raw-content'):
        public_payload(source)


def test_docs_distinguish_urls_from_local_paths(tmp_path):
    source = tmp_path/'unit.md'
    source.write_text('[Public](https://example.org/media/audio.mp3)')
    assert public_payload(source) == source.read_bytes()
    source.write_text('Source is /home/private/audio.mp3')
    with pytest.raises(ValueError, match='Machine-local path'):
        public_payload(source)


def test_same_base_python_does_not_prove_same_notebook_environment(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/'notebooks/v1.2_metadata_quickstart.ipynb').read_text())
    configuration = next(c['source'] for c in notebook['cells'] if c['id'] == 'configuration')
    first, second = tmp_path/'first-python', tmp_path/'second-python'
    first.symlink_to(sys.executable)
    second.symlink_to(sys.executable)
    assert first.resolve() == second.resolve()
    monkeypatch.chdir(root)
    monkeypatch.setenv('ARTIFACTBENCH_NOTEBOOK_PYTHON', str(first))
    monkeypatch.setattr(sys, 'executable', str(second))
    with pytest.raises(AssertionError, match='kernel executable'):
        exec(configuration, {})


def test_notebook_checks_environment_prefix_too(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/'notebooks/v1.2_metadata_quickstart.ipynb').read_text())
    configuration = next(c['source'] for c in notebook['cells'] if c['id'] == 'configuration')
    monkeypatch.chdir(root)
    monkeypatch.setenv('ARTIFACTBENCH_NOTEBOOK_PYTHON', sys.executable)
    monkeypatch.setenv('ARTIFACTBENCH_NOTEBOOK_PREFIX', str(root/'unit-fixture-wrong-prefix'))
    with pytest.raises(AssertionError, match='kernel environment'):
        exec(configuration, {})
