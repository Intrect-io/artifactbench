"""완료된 첫 rc2 checkpoint만 명시적으로 표시한 draft LaTeX 표/숫자를 만든다."""
import json
from pathlib import Path

from artifactbench.v12.common import digest, write_once
from artifactbench.v12.report import load_completed
from artifactbench.v12.verify_release import verify
from paper.render_results import formatted, table


def main():
    root = Path(__file__).resolve().parents[1]
    release = root/'out/v1.2_frozen_rc2_260905'
    checkpoint = root/'out/v1.2_checkpoint_artifactnet_rc2_260905'
    run = root/'out/v1.2_full_rc2_260905/artifactnet'
    verification = verify(release)
    entries = json.loads((release/'manifest.public.json').read_text())['bench']
    rows, identity = load_completed(run, {r['id']: r for r in entries})
    result = json.loads((checkpoint/'checkpoint.json').read_text())
    summary = json.loads((checkpoint/'summary.json').read_text())
    inputs = json.loads((checkpoint/'inputs.json').read_text())
    if (result['model'] != 'artifactnet' or identity['model'] != 'artifactnet' or len(rows) != len(entries)
            or summary['checkpoint_sha256'] != digest(checkpoint/'checkpoint.json')
            or inputs['manifest_sha256'] != verification['manifest_sha256']
            or inputs['run_summary_sha256'] != digest(run/'summary.json')
            or inputs['replicates'] != 2000):
        raise ValueError('Completed checkpoint evidence does not match the frozen release')
    suno, udio = 'suno_v5.5_native_260905', 'udio_2026_version_unknown_native_260905'
    for source, codec in ((suno, 'aac'), (udio, 'mp3')):
        if {r['audio_codec'] for r in entries if r['source'] == source} != {codec}:
            raise ValueError('Draft transport-confounding statement no longer matches the data')
    values = {
        'CheckpointLegacyFone': f"{result['raw_05']['legacy']['metrics']['F1']['estimate']:.4f}",
        'CheckpointMixedFone': f"{result['raw_05']['non_demo']['metrics']['F1']['estimate']:.4f}",
        'CheckpointSunoRawTpr': f"{100*result['raw_05'][suno]['metrics']['TPR']['estimate']:.1f}",
        'CheckpointSunoRawLow': f"{100*result['raw_05'][suno]['metrics']['TPR']['lower']:.1f}",
        'CheckpointSunoRawHigh': f"{100*result['raw_05'][suno]['metrics']['TPR']['upper']:.1f}",
        'CheckpointSunoStackTpr': f"{100*result['operating_stacks'][suno]['cnn_0225_rescue_002']['metrics']['TPR']['estimate']:.1f}",
        'CheckpointLyriaTP': f"{result['official_demos']['lyria_official_demo_260905']['raw_05']['TP']:.0f}",
        'CheckpointStableTP': f"{result['official_demos']['stable_audio_official_demo_260905']['raw_05']['TP']:.0f}",
    }
    output = root/'paper/generated'
    (output/'checkpoint_numbers.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k, v in values.items()))
    data = []
    for source, label in (('legacy', 'Legacy'), (suno, 'Suno v5.5 (AAC)'), (udio, 'Udio 2026 (MP3)')):
        m = result['raw_05'][source]['metrics']
        data.append([label, f"{m['scored']['estimate']:.0f}/{m['attempted']['estimate']:.0f}",
                     f"{m['TP']['estimate']:.0f}/{m['TP']['estimate']+m['FN']['estimate']:.0f}",
                     formatted(m['TPR'], True), '--' if m['F1']['estimate'] is None else f"{m['F1']['estimate']:.4f}"])
    table(output/'checkpoint_table.tex', ['Cohort', 'Scored/all', 'TP/scored AI', r'TPR (\%) [95\% CI]', 'F1'], data, smoke=False)
    write_once(output/'checkpoint_evidence.json', {'scope': 'First completed ArtifactNet only; seven other checkpoints still pending',
        'manifest_sha256': verification['manifest_sha256'], 'checkpoint_sha256': digest(checkpoint/'checkpoint.json'),
        'run_summary_sha256': digest(run/'summary.json'), 'generator_sha256': digest(__file__),
        'raw_point': .5, 'suno_codec': 'aac', 'udio_codec': 'mp3'})


if __name__ == '__main__':
    main()
