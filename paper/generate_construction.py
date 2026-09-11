"""동결된 데이터의 수치만 논문 매크로에 전달한다. 추론 결과와 혼합하지 않는다."""
import hashlib
import json
from pathlib import Path


def main():
    paper = Path(__file__).resolve().parent
    release = paper.parent / 'out/v1.2_frozen_rc2_260905'
    summary = json.loads((release/'release.json').read_text())
    manifest = json.loads((release/'manifest.public.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(release/'manifest.public.json') == summary['public_file_sha256']['manifest.public.json']
    assert len(manifest['bench']) == summary['evaluation_entries']
    counts = summary['source_counts']
    values = {'BenchEntries': summary['evaluation_entries'],
              'LegacyEntries': summary['partitions']['legacy'],
              'SunoEntries': counts['suno_v5.5_native_260905'],
              'UdioEntries': counts['udio_2026_version_unknown_native_260905'],
              'LyriaEntries': counts['lyria_official_demo_260905'],
              'StableAudioEntries': counts['stable_audio_official_demo_260905'],
              'RecordingGroups': summary['identified_recording_groups'],
              'DependenceClusters': summary['dependence_clusters'],
              'NativePairs': summary['native_transport_pairs']}
    lines = [f'\\newcommand{{\\{key}}}{{{value:,}}}' for key, value in values.items()]
    (paper/'generated/construction_numbers.tex').write_text('\n'.join(lines)+'\n')
    evidence = {'scope': 'frozen construction counts, not detector performance', 'values': values,
                'input_sha256': {str(p.relative_to(paper.parent)): digest(p) for p in
                    (release/'release.json', release/'manifest.public.json', Path(__file__))}}
    (paper/'generated/construction_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(values))


if __name__ == '__main__':
    main()
