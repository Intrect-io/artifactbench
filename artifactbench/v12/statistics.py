"""소스 간 공유 집단을 보존하는 결정론적 bootstrap과 실패 인지 지표."""
from collections import defaultdict

import numpy as np


def bootstrap_weights(entries, replicates=2000, seed=260905):
    """집단이 가로지르는 source/label cell은 함께 층화한다. 점수는 입력받지 않는다."""
    clusters = defaultdict(list)
    for index, entry in enumerate(entries):
        clusters[entry['dependence_cluster']].append(index)
    parents = {(r['label'], r['source']): (r['label'], r['source']) for r in entries}
    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    for indices in clusters.values():
        cells = [(entries[i]['label'], entries[i]['source']) for i in indices]
        for cell in cells[1:]:
            a, b = root(cells[0]), root(cell)
            parents[max(a, b)] = min(a, b)
    strata = defaultdict(list)
    for cluster, indices in sorted(clusters.items()):
        first = entries[indices[0]]
        strata[root((first['label'], first['source']))].append(cluster)
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(entries)), dtype=np.int32)
    description = []
    for stratum, members in sorted(strata.items()):
        n = len(members)
        sampled = rng.integers(0, n, size=(replicates, n))
        counts = np.zeros((replicates, n), dtype=np.int32)
        np.add.at(counts, (np.arange(replicates)[:, None], sampled), 1)
        for index, cluster in enumerate(members):
            weights[:, clusters[cluster]] = counts[:, index, None]
        description.append({'anchor_cell': list(stratum), 'clusters': members,
                            'cells': [list(c) for c in sorted(parents) if root(c) == stratum]})
    return weights, description


def _ratio(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=np.asarray(b) != 0)


def weighted_auc(labels, scores, weights):
    """동점에 1/2 가중치를 주는 정확한 weighted AUROC; 점수 정렬은 한 번만 한다."""
    if not len(labels):
        return np.full(len(weights), np.nan)
    order = np.argsort(scores, kind='stable')
    labels, scores, weights = labels[order], scores[order], weights[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    positive = np.add.reduceat(weights * labels, starts, axis=1)
    negative = np.add.reduceat(weights * (~labels), starts, axis=1)
    before = np.cumsum(negative, axis=1) - negative
    numerator = np.sum(positive * (before + .5*negative), axis=1)
    return _ratio(numerator, np.sum(positive, axis=1)*np.sum(negative, axis=1))


def metric_arrays(entries, predictions, weights):
    weights = np.asarray(weights, dtype=np.float64)
    labels = np.array([r['label'] == 'ai' for r in entries])
    successful = np.array([r['outcome'] == 'scored' for r in predictions])
    # NaN은 실패의 내부 표식이며 모델 출력/가짜 판정으로 저장하지 않는다.
    scores = np.array([r['prob'] if r['outcome'] == 'scored' else np.nan for r in predictions])
    if np.any(~np.isfinite(scores[successful])) or np.any((scores[successful] < 0) | (scores[successful] > 1)):
        raise ValueError('Invalid successful prediction')
    positive = scores >= .5
    tp = weights @ (successful & labels & positive)
    fn = weights @ (successful & labels & ~positive)
    fp = weights @ (successful & ~labels & positive)
    tn = weights @ (successful & ~labels & ~positive)
    both = (tp+fn > 0) & (fp+tn > 0)
    tpr, fpr = _ratio(tp, tp+fn), _ratio(fp, fp+tn)
    total, scored = weights.sum(axis=1), weights @ successful
    values = {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn, 'scored': scored, 'attempted': total,
        'coverage': _ratio(scored, total), 'precision': _ratio(tp, tp+fp), 'TPR': tpr,
        'FPR': fpr, 'specificity': 1-fpr, 'F1': np.where(both, _ratio(2*tp, 2*tp+fp+fn), np.nan),
        'balanced_accuracy': np.where(both, .5*(tpr+1-fpr), np.nan),
        'all_attempt_correctness': _ratio(tp+tn, total),
        'AUROC': weighted_auc(labels[successful], scores[successful], weights[:, successful])}
    source_values = defaultdict(list)
    for source in sorted({r['source'] for r in entries}):
        mask = np.array([r['source'] == source for r in entries])
        ai, real = mask & labels, mask & ~labels
        if ai.any():
            source_values['macro_TPR'].append(_ratio(weights @ (ai & successful & positive), weights @ (ai & successful)))
        if real.any():
            source_values['macro_FPR'].append(_ratio(weights @ (real & successful & positive), weights @ (real & successful)))
    for name in ('macro_TPR', 'macro_FPR'):
        # 실패 때문에 완전히 측정 불가능한 cell을 조용히 평균에서 삭제하지 않는다.
        values[name] = np.mean(source_values[name], axis=0) if source_values[name] else np.full(len(weights), np.nan)
    return values


def interval(values, point):
    finite = values[np.isfinite(values)]
    return {'estimate': float(point) if np.isfinite(point) else None,
            'lower': float(np.quantile(finite, .025)) if len(finite) else None,
            'upper': float(np.quantile(finite, .975)) if len(finite) else None,
            'finite_replicates': len(finite), 'replicates': len(values)}


def summarize_bootstrap(entries, predictions, replicates=2000, seed=260905):
    if len(entries) != len(predictions) or not entries:
        raise ValueError('Nonempty aligned entries/predictions are required')
    weights, strata = bootstrap_weights(entries, replicates, seed)
    point = metric_arrays(entries, predictions, np.ones((1, len(entries))))
    sampled = metric_arrays(entries, predictions, weights)
    return {'entries': len(entries), 'clusters': len({r['dependence_cluster'] for r in entries}),
            'metrics': {key: interval(sampled[key], value[0]) for key, value in point.items()}, 'strata': strata}


def paired_difference(entries, first, second, replicates=2000, seed=260905):
    if len(entries) != len(first) or len(entries) != len(second):
        raise ValueError('Paired inputs are not aligned')
    common = [i for i, (a, b) in enumerate(zip(first, second)) if a['outcome'] == b['outcome'] == 'scored']
    if not common:
        return {'attempted_common_frame': len(entries), 'common_successes': 0, 'differences': {}}
    selected, a, b = ([rows[i] for i in common] for rows in (entries, first, second))
    weights, strata = bootstrap_weights(selected, replicates, seed)
    points = [metric_arrays(selected, rows, np.ones((1, len(selected)))) for rows in (a, b)]
    sampled = [metric_arrays(selected, rows, weights) for rows in (a, b)]
    names = ('F1', 'AUROC', 'TPR', 'FPR', 'balanced_accuracy', 'macro_TPR', 'macro_FPR')
    return {'attempted_common_frame': len(entries), 'common_successes': len(common),
            'direction': 'first minus second on identical successful recordings and bootstrap weights',
            'differences': {key: interval(sampled[0][key]-sampled[1][key], points[0][key][0]-points[1][key][0]) for key in names},
            'strata': strata}
