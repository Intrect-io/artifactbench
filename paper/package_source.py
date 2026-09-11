"""원고에 필요한 파일만 ZIP으로 묶고 새 추출본 전체를 실제 컴파일한다."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile

from artifactbench.v12.common import digest, write_once
from artifactbench.v12.package_metadata import public_payload
from artifactbench.v12.verify_bundle import safe_member
from artifactbench.v12.verify_release import check_public


def source_name(name):
    path = safe_member(name)
    if (not re.fullmatch(r'[A-Za-z0-9_./-]+', name) or any(p.startswith('.') for p in path.parts)
            or path.suffix not in ('.tex', '.bib', '.bbl', '.pdf') or name == 'main.pdf'):
        raise ValueError('Unsupported source archive name: '+name)
    if path.suffix == '.pdf' and path.parts[0] not in ('figures', 'generated'):
        raise ValueError('Only manuscript figures may be PDF payloads')
    return name


def source_names(list_path):
    names = [source_name(line) for line in list_path.read_text().splitlines() if line]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate source inventory entry')
    if not {'main.tex', 'main.bbl', 'refs.bib'}.issubset(names):
        raise ValueError('Main source and both bibliography sources are required')
    return sorted(names)


def dependency_names(text):
    """고정 Tectonic CLI의 단일 rule 형식만 지원한다. 일반 Makefile parser가 아니다."""
    text = text.replace('\\\n', ' ')
    if text.count(' : ') != 1:
        raise ValueError('Unsupported compiler dependency format')
    names = text.split(' : ', 1)[1].split()
    return {source_name(name) for name in names}


def check_log(text):
    if re.search(r'^!|Undefined control sequence|undefined references|Citation .* undefined|'
                 r'Reference .* undefined|Overfull \\[hv]box|Rerun to get cross-references right', text, re.MULTILINE):
        raise ValueError('Unresolved TeX error/reference or overflowing layout')
    return [line for line in text.splitlines() if 'Underfull ' in line]


def extract_source_archive(archive, destination, hashes):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Source extraction destination must be empty')
    for name in hashes:
        source_name(name)
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        if len({m.filename for m in members}) != len(members):
            raise ValueError('Duplicate source ZIP member')
        for member in members:
            source_name(member.filename)
            if not stat.S_ISREG(member.external_attr >> 16):
                raise ValueError('Non-regular source ZIP member')
        if {m.filename for m in members} != set(hashes):
            raise ValueError('Source ZIP inventory mismatch')
        if sum(m.file_size for m in members) > 64*1024*1024:
            raise ValueError('Source ZIP exceeds declared 64 MiB limit')
        stream.extractall(destination)
    if any(digest(destination/name) != checksum for name, checksum in hashes.items()):
        raise ValueError('Extracted source hash mismatch')


def invoke(command, cwd, evidence, env):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    write_once(evidence, {'command': command, 'returncode': result.returncode,
                          'stdout': result.stdout, 'stderr': result.stderr})
    if result.returncode:
        raise RuntimeError('Manuscript verification command failed; see '+evidence.name)
    return result.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paper', type=Path, default=Path('paper'))
    ap.add_argument('--compiler', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    paper, output, compiler = args.paper.resolve(), args.output.resolve(), args.compiler.resolve()
    names = source_names(paper/'source_files.txt')
    if output.is_relative_to(paper):
        raise ValueError('Packaging evidence must be outside the manuscript tree')
    if not compiler.is_file():
        raise ValueError('Compiler is missing')
    binaries = {}
    for name in ('pdfinfo', 'pdftotext', 'pdftoppm'):
        binary = shutil.which(name)
        if binary is None:
            raise ValueError('Required PDF verification tool missing: '+name)
        binaries[name] = binary
    inputs = {}
    for name in names:
        path = paper/name
        if path.resolve() != path or not path.is_file():
            raise ValueError('Source file missing or symbolic link: '+name)
        if path.suffix != '.pdf':
            public_payload(path)
        elif not path.read_bytes().startswith(b'%PDF-'):
            raise ValueError('Figure is not a PDF: '+name)
        inputs[name] = digest(path)
    reference = paper/'main.pdf'
    reference_hash = digest(reference)
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # 기존 원고 경로나 사용자 TeX search path로 빠지는 것을 막는다. 표준 캐시는 재사용한다.
    for key in ('TEXINPUTS', 'BIBINPUTS', 'BSTINPUTS', 'TEXMFHOME', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    version = invoke([str(compiler), '--version'], paper, output/'compiler.json', env).strip()
    if version != 'Tectonic 0.17.0':
        raise ValueError('This verified packaging path requires Tectonic 0.17.0')
    for index, name in enumerate(n for n in names if n.endswith('.pdf')):
        info = invoke([binaries['pdfinfo'], str(paper/name)], paper, output/f'figure_{index}_pdfinfo.json', env)
        check_public(info)
        if not re.search(r'^JavaScript:\s+no$', info, re.MULTILINE):
            raise ValueError('Figure PDF JavaScript state is not verified absent')
    draft = 'Draft status.' in (paper/'main.tex').read_text()
    archive = output/('artifactbench-latex-source-DRAFT.zip' if draft else 'artifactbench-latex-source.zip')
    write_once(output/'inputs.json', {'scope': 'Source packaging and local compiler verification only; not submission approval',
        'source_sha256': inputs, 'source_list_sha256': digest(paper/'source_files.txt'),
        'reference_pdf_sha256': reference_hash, 'compiler_sha256': digest(compiler),
        'compiler_version': version, 'tool_sha256': digest(__file__),
        'draft_marker_present': draft, 'verification_tools': {name: digest(path) for name, path in binaries.items()}})
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for name in names:
            member = zipfile.ZipInfo(name, (2026, 9, 5, 0, 0, 0))
            member.create_system = 3
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            stream.writestr(member, (paper/name).read_bytes(), compresslevel=9)
    extracted = Path(tempfile.mkdtemp(prefix='artifactbench-latex-'))
    extract_source_archive(archive, extracted, inputs)
    write_once(output/'extraction.json', {'directory': str(extracted), 'archive_sha256': digest(archive)})
    invoke([str(compiler), '--only-cached', '--untrusted', '--keep-logs', '--keep-intermediates',
            '--makefile-rules', 'dependencies.d', 'main.tex'], extracted, output/'compile.json', env)
    # outdir를 주면 dependency 경로에 outdir가 붙는다. 추출 루트에서 outdir 없이 실행한다.
    dependencies = dependency_names((extracted/'dependencies.d').read_text()) | {'main.bbl'}
    if dependencies != set(names):
        raise ValueError('Declared source inventory differs from actual compilation inputs')
    warnings = check_log((extracted/'main.log').read_text())
    if any(digest(extracted/name) != checksum for name, checksum in inputs.items()):
        raise ValueError('Compilation changed a packaged input, including the bibliography')
    metadata, text_hashes, raster_hashes = {}, {}, {}
    for label, pdf in (('reference', reference), ('extracted', extracted/'main.pdf')):
        info = invoke([binaries['pdfinfo'], str(pdf)], extracted, output/(label+'_pdfinfo.json'), env)
        if not re.search(r'^JavaScript:\s+no$', info, re.MULTILINE):
            raise ValueError('PDF JavaScript state is not verified absent')
        metadata[label] = {'pages': int(re.search(r'^Pages:\s+(\d+)$', info, re.MULTILINE)[1])}
        text = invoke([binaries['pdftotext'], '-layout', str(pdf), '-'], extracted, output/(label+'_text.json'), env)
        text_hashes[label] = hashlib.sha256(text.encode()).hexdigest()
        rendered = output/(label+'_pages')
        rendered.mkdir()
        invoke([binaries['pdftoppm'], '-r', '72', '-png', str(pdf), str(rendered/'page')],
               extracted, output/(label+'_render.json'), env)
        raster_hashes[label] = {path.name: digest(path) for path in sorted(rendered.glob('page-*.png'))}
        if len(raster_hashes[label]) != metadata[label]['pages']:
            raise ValueError('Not every PDF page was rasterized')
    if (metadata['reference'] != metadata['extracted'] or text_hashes['reference'] != text_hashes['extracted']
            or raster_hashes['reference'] != raster_hashes['extracted']):
        raise ValueError('Extracted-source PDF differs in text, page count, or rendered pages')
    if any(digest(paper/name) != checksum for name, checksum in inputs.items()) or digest(reference) != reference_hash:
        raise ValueError('Original manuscript changed during source packaging')
    shutil.copyfile(extracted/'main.pdf', output/'recompiled.pdf')
    summary = {'status': 'LaTeX source archive and independent compilation verified',
        'scope': 'Local Tectonic 0.17.0 with existing cached TeX resources, not arXiv server compilation or scientific completion',
        'draft_marker_present': draft, 'archive': archive.name, 'archive_sha256': digest(archive),
        'archive_bytes': archive.stat().st_size, 'source_files': names, 'pages': metadata['extracted']['pages'],
        'text_sha256': text_hashes['extracted'], 'page_image_sha256': raster_hashes['extracted'],
        'recompiled_pdf_sha256': digest(output/'recompiled.pdf'), 'underfull_warnings': warnings,
        'inference_or_publication_claim': False}
    write_once(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
