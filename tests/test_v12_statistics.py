import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from artifactbench.v12.statistics import bootstrap_weights, metric_arrays, paired_difference, weighted_auc


def entries():
    return [{'label': 'ai' if i < 4 else 'real', 'source': 'x' if i % 2 else 'y',
             'dependence_cluster': 'same' if i in (0, 1) else str(i)} for i in range(8)]


def test_cross_source_cluster_never_splits_and_resampling_is_deterministic():
    weights, strata = bootstrap_weights(entries(), replicates=40)
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    assert any(len(s['cells']) == 2 for s in strata)
    np.testing.assert_array_equal(weights, bootstrap_weights(entries(), replicates=40)[0])
    assert np.unique(weights[:, 0]).size > 1


def test_exact_weighted_auc_ties_agrees_with_sklearn():
    labels = np.array([True, False, True, False, True, False])
    scores = np.array([.5, .5, .9, .1, .2, .9])
    weights = np.array([[1, 1, 1, 1, 1, 1], [2, 3, 0, 1, 2, 1]])
    actual = weighted_auc(labels, scores, weights)
    for index, w in enumerate(weights):
        assert actual[index] == pytest.approx(roc_auc_score(labels, scores, sample_weight=w))


def test_failed_row_remains_in_attempt_denominator_and_no_fake_auroc():
    e = [{'label': 'ai', 'source': 'x'}, {'label': 'real', 'source': 'y'}, {'label': 'ai', 'source': 'x'}]
    p = [{'outcome': 'scored', 'prob': .9}, {'outcome': 'scored', 'prob': .1}, {'outcome': 'error'}]
    result = metric_arrays(e, p, np.ones((1, 3)))
    assert result['F1'][0] == result['AUROC'][0] == 1
    assert result['coverage'][0] == result['all_attempt_correctness'][0] == 2/3


def test_identical_models_have_zero_paired_intervals_on_common_coverage():
    e = entries()
    p = [{'outcome': 'scored', 'prob': .8 if row['label'] == 'ai' else .2} for row in e]
    p[1] = {'outcome': 'error'}
    result = paired_difference(e, p, p, replicates=40)
    assert result['common_successes'] == 7
    for metric in result['differences'].values():
        assert metric['estimate'] == metric['lower'] == metric['upper'] == 0


def test_single_class_binary_auc_and_f1_are_undefined():
    e = [{'label': 'ai', 'source': 'x'}]
    result = metric_arrays(e, [{'outcome': 'scored', 'prob': .8}], np.ones((1, 1)))
    assert np.isnan(result['AUROC'][0]) and np.isnan(result['F1'][0])
