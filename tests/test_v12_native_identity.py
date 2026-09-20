import pytest

from artifactbench.v12.correct_native_identity import choose_udio_variant, url_key
from artifactbench.v12.prepare_selection import select_native


def variant(index, container='mp3'):
    return {'source_url': f'https://example.org/samples/parent/{index}/song.{container}',
            'variant_id': str(index), 'container': container}


def test_numbered_sibling_is_not_a_transport_even_with_same_parent_id():
    a, b = variant(1), variant(2)
    matched, rejected = choose_udio_variant([a, b], {'song_path': b['source_url']})
    assert matched == [(b, ['song_path'])] and rejected == [a]


def test_missing_matching_url_does_not_fall_back_to_sibling():
    with pytest.raises(ValueError, match='No complete'):
        choose_udio_variant([variant(1)], {'song_path': variant(2)['source_url']})


def test_original_path_fallback_is_explicit_and_current_path_is_preferred():
    original, current = variant(1), variant(2)
    metadata = {'song_path': current['source_url'], 'lineage': {'original_song_path': original['source_url']}}
    assert choose_udio_variant([original], metadata)[0] == [(original, ['original_song_path'])]
    assert choose_udio_variant([original, current], metadata)[0][0] == (current, ['song_path'])


def test_query_and_path_components_are_not_discarded():
    assert url_key('https://example.org/a?trim=0:10') != url_key('https://example.org/a?trim=0:20')
    assert url_key('https://example.org/trims/1/song.mp3') != url_key('https://example.org/1/song.mp3')
    assert url_key('https://user:password@example.org/a') is None


def test_selection_uses_declared_file_not_first_variant_hash():
    a, b = variant(1), variant(2)
    candidate = {'track_id': 'udio:one', 'provider': 'udio', 'variants': [a, b]}
    metadata = {'udio:one': {'outcome': 'unknown_version', 'created_at': '2026-08-01',
        'creator_id': 'creator', 'artist': 'Artist', 'song_path': b['source_url'],
        'source_url': 'https://example.org/song/one', 'page_sha256': 'page-sha',
        'retrieved_at': '2026-09-05', 'label_scope': 'platform output'}}
    selected, _ = select_native([candidate], metadata, set())
    assert selected[0]['source_url'] == b['source_url']
    assert selected[0]['additional_native_variants'] == []
