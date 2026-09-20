import json

import pytest

from artifactbench.v12.common import digest, write_once
from artifactbench.v12.report import MODELS
from paper.render_results import formatted, load_report, tex_escape, verify_prediction_binding


def rendering_fixture(tmp_path):
    """로더의 출처 검증만 검사하는 unit fixture; 모델/통계 실행 결과가 아니다."""
    release, report = tmp_path/'release', tmp_path/'report'
    write_once(release/'manifest.public.json', {'unit_fixture': True})
    write_once(release/'release.json', {'evaluation_entries': 2, 'public_file_sha256': {
        'manifest.public.json': digest(release/'manifest.public.json')}})
    write_once(report/'inputs.json', {'input_hashes': {'manifest.public.json': digest(release/'manifest.public.json')}})
    for name in MODELS:
        write_once(report/(name+'.json'), {'model': name, 'coverage': {'scored': 2}})
    write_once(report/'paired_differences.json', {})
    write_once(report/'summary.json', {'status': 'SMOKE REPORT WIRING ONLY', 'replicates': 20,
        'models': list(MODELS), 'attempted_per_model': 2,
        'report_sha256': {name: digest(report/(name+'.json')) for name in MODELS},
        'paired_differences_sha256': digest(report/'paired_differences.json')})
    return release, report


def test_smoke_cannot_be_rendered_as_final(tmp_path):
    release, report = rendering_fixture(tmp_path)
    with pytest.raises(ValueError, match='2000-replicate'):
        load_report(report, release)
    loaded, _, _ = load_report(report, release, allow_smoke=True)
    assert set(loaded) == set(MODELS)


def test_changed_statistics_are_rejected(tmp_path):
    release, report = rendering_fixture(tmp_path)
    (report/'artifactnet.json').write_text(json.dumps({'model': 'artifactnet', 'coverage': {'scored': 1}}))
    with pytest.raises(ValueError, match='hash changed'):
        load_report(report, release, allow_smoke=True)


def test_undefined_metric_is_not_shown_as_zero_and_labels_are_escaped():
    assert formatted({'estimate': None}) == '--'
    assert formatted({'estimate': .25, 'lower': .1, 'upper': .5}, True) == '25.0 [10.0, 50.0]'
    assert tex_escape('source_name & 50%') == r'source\_name \& 50\%'


def test_final_render_requires_public_prediction_admission(tmp_path):
    release, report = rendering_fixture(tmp_path)
    path = report/'summary.json'
    summary = json.loads(path.read_text())
    summary.update(status='statistical report complete', replicates=2000)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='requires --predictions'):
        load_report(report, release)


def test_public_prediction_binding_requires_exact_all_model_and_paired_statistics(tmp_path, monkeypatch):
    release, report = rendering_fixture(tmp_path)
    files = {name+'.json': digest(report/(name+'.json')) for name in MODELS}
    files['paired_differences.json'] = digest(report/'paired_differences.json')
    def checked(predictions, checked_release):
        assert predictions == tmp_path/'predictions' and checked_release == release
        return {}, {}, {}, {'reference_statistics': {'file_sha256': files}}
    monkeypatch.setattr('paper.render_results.load_public_results', checked)
    verify_prediction_binding(report, release, tmp_path/'predictions')
    files['deepfense.json'] = '0'*64
    with pytest.raises(ValueError, match='differ from validated public predictions'):
        verify_prediction_binding(report, release, tmp_path/'predictions')
    files.pop('deepfense.json')
    with pytest.raises(ValueError, match='all nine'):
        verify_prediction_binding(report, release, tmp_path/'predictions')


def test_failed_public_admission_cannot_be_bypassed_by_rendering(tmp_path, monkeypatch):
    def rejected(*args):
        raise ValueError('unit original adapter audit rejected')
    monkeypatch.setattr('paper.render_results.load_public_results', rejected)
    with pytest.raises(ValueError, match='original adapter audit rejected'):
        verify_prediction_binding(tmp_path, tmp_path, tmp_path)
