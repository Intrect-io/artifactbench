"""논문 소스 묶음의 이름·dependency·로그·추출 경계를 검사한다."""
import hashlib
import stat
import zipfile

import pytest

from paper.package_source import check_log, dependency_names, extract_source_archive, source_name, source_names


@pytest.mark.parametrize('name', ('../private.tex', '/absolute.tex', '.hidden.tex', 'a/.hidden.tex',
                                  'main.pdf', 'main.log', 'audio.wav', 'a file.tex', 'source.py'))
def test_unsupported_source_names_rejected(name):
    with pytest.raises(ValueError):
        source_name(name)


def test_source_inventory_requires_bibliography_and_no_duplicates(tmp_path):
    listing = tmp_path/'source_files.txt'
    listing.write_text('main.tex\nrefs.bib\n')
    with pytest.raises(ValueError, match='bibliography'):
        source_names(listing)
    listing.write_text('main.tex\nrefs.bib\nmain.bbl\nmain.tex\n')
    with pytest.raises(ValueError, match='Duplicate'):
        source_names(listing)


def test_compiler_dependencies_are_read_without_outdir_rewriting():
    assert dependency_names('main.pdf main.aux : main.tex \\\n  generated/table.tex \\\n  refs.bib\n') == {
        'main.tex', 'generated/table.tex', 'refs.bib'}
    with pytest.raises(ValueError):
        dependency_names('main.pdf : main.tex \\\n  ../out/generated/table.tex\n')
    with pytest.raises(ValueError, match='format'):
        dependency_names('not a dependency rule')


@pytest.mark.parametrize('message', ('! Undefined control sequence.', 'LaTeX Warning: There were undefined references.',
    "Package natbib Warning: Citation `missing' on page 1 undefined", 'Overfull \\hbox (12.0pt too wide)',
    'LaTeX Warning: Reference `missing\' on page 1 undefined', 'Rerun to get cross-references right.'))
def test_unresolved_or_overflowing_compilation_is_not_accepted(message):
    with pytest.raises(ValueError, match='Unresolved'):
        check_log(message)


def test_underfull_warning_is_preserved_not_hidden():
    warning = 'Underfull \\hbox (badness 1200) in paragraph'
    assert check_log('normal line\n'+warning) == [warning]


def unit_zip(path, members):
    with zipfile.ZipFile(path, 'x') as archive:
        for name, mode in members:
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = mode << 16
            archive.writestr(member, b'unit source fixture')


def test_source_zip_extracts_exact_bytes(tmp_path):
    archive = tmp_path/'unit.zip'
    unit_zip(archive, [('main.tex', stat.S_IFREG | 0o644)])
    hashes = {'main.tex': hashlib.sha256(b'unit source fixture').hexdigest()}
    extract_source_archive(archive, tmp_path/'extracted', hashes)
    assert (tmp_path/'extracted/main.tex').read_bytes() == b'unit source fixture'


def test_source_zip_rejects_duplicate_names(tmp_path):
    archive = tmp_path/'unit.zip'
    with pytest.warns(UserWarning, match='Duplicate'):
        unit_zip(archive, [('main.tex', stat.S_IFREG | 0o644)]*2)
    with pytest.raises(ValueError, match='Duplicate'):
        extract_source_archive(archive, tmp_path/'extracted', {'main.tex': '0'*64})


def test_source_zip_rejects_symbolic_link_before_extraction(tmp_path):
    archive = tmp_path/'unit.zip'
    unit_zip(archive, [('main.tex', stat.S_IFLNK | 0o777)])
    with pytest.raises(ValueError, match='Non-regular'):
        extract_source_archive(archive, tmp_path/'extracted', {'main.tex': '0'*64})
    assert not (tmp_path/'extracted').exists()


def test_source_zip_rejects_extra_file_before_extraction(tmp_path):
    archive = tmp_path/'unit.zip'
    unit_zip(archive, [('main.tex', stat.S_IFREG | 0o644), ('extra.tex', stat.S_IFREG | 0o644)])
    with pytest.raises(ValueError, match='inventory'):
        extract_source_archive(archive, tmp_path/'extracted', {'main.tex': '0'*64})
    assert not (tmp_path/'extracted').exists()


def test_source_zip_rejects_changed_file(tmp_path):
    archive = tmp_path/'unit.zip'
    unit_zip(archive, [('main.tex', stat.S_IFREG | 0o644)])
    with pytest.raises(ValueError, match='hash mismatch'):
        extract_source_archive(archive, tmp_path/'extracted', {'main.tex': '0'*64})


def test_source_zip_refuses_existing_directory(tmp_path):
    (tmp_path/'keep.tex').write_text('preserve')
    with pytest.raises(ValueError, match='must be empty'):
        extract_source_archive(tmp_path/'absent.zip', tmp_path, {'main.tex': '0'*64})
    assert (tmp_path/'keep.tex').read_text() == 'preserve'
