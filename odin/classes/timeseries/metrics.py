import math

import numpy as np
import pandas as pd


def get_confusion_matrix(y_true, y_score, threshold):
    data = pd.DataFrame(data={'y_true': y_true,
                              'y_score': y_score})

    data['y_pred'] = np.where(data['y_score'] >= threshold, 1, 0)
    tp = len(data.loc[(data['y_true'] == 1) & (data['y_pred'] == 1)].index)
    fp = len(data.loc[(data['y_true'] == 0) & (data['y_pred'] == 1)].index)
    fn = len(data.loc[(data['y_true'] == 1) & (data['y_pred'] == 0)].index)
    tn = len(data.loc[(data['y_true'] == 0) & (data['y_pred'] == 0)].index)

    cm = np.empty((2, 2))
    cm[0][0] = tn
    cm[0][1] = fp
    cm[1][0] = fn
    cm[1][1] = tp

    return cm


def accuracy(y_true, y_score, threshold):
    tn, fp, fn, tp = get_confusion_matrix(y_true, y_score, threshold).ravel()

    return (tp+tn)/(tp+fp+fn+tn)


def precision_score(y_true, y_score, threshold):
    _, fp, _, tp = get_confusion_matrix(y_true, y_score, threshold).ravel()

    if tp == 0 and fp == 0:
        return 0

    return tp/(tp+fp)


def recall_score(y_true, y_score, threshold):
    _, _, fn, tp = get_confusion_matrix(y_true, y_score, threshold).ravel()

    if tp == 0 and fn == 0:
        return 0

    return tp/(tp+fn)


def f1_score(y_true, y_score, threshold):
    precision = precision_score(y_true, y_score, threshold)
    recall = recall_score(y_true, y_score, threshold)

    if precision == 0 and recall == 0:
        return 0

    return 2*precision*recall/(precision+recall)


def false_alarm_rate(y_true, y_score, threshold):
    tn, fp, _, _ = get_confusion_matrix(y_true, y_score, threshold).ravel()

    if fp == 0 and tn == 0:
        return 0

    return fp/(fp+tn)


def miss_alarm_rate(y_true, y_score, threshold):
    _, _, fn, tp = get_confusion_matrix(y_true, y_score, threshold).ravel()

    if fn == 0 and tp == 0:
        return 0

    return fn/(fn+tp)


def nab_score(anomaly_windows, proposals, y_score, threshold, A_tp, A_fp, A_fn):
    all_sigma = []
    proposals['label'] = np.where(y_score >= threshold, 1, 0)

    for i, (start, end) in enumerate(anomaly_windows):
        max_d = len(proposals.loc[(proposals.index >= start) & (proposals.index <= end)].index)

        # evaluate TP and FN
        prediction = proposals.loc[(proposals.index >= start) & (proposals.index <= end) & proposals['label'] == 1].head(1)
        if prediction.empty:
            all_sigma.append(-A_fn)
        else:
            d = len(proposals.loc[(proposals.index >= prediction.index[0]) & (proposals.index <= end)].index)-1
            y = (d/max_d)*(-math.e)
            sigma = (2/(1+math.e**(5*y)))-1
            all_sigma.append(A_tp*sigma)

        if i == 0:
            fps = proposals.loc[(proposals.index < start) & (proposals['label'] == 1)]
            all_sigma.append(-len(fps.index)*A_fp)

        if i+1 == len(anomaly_windows):
            fps = proposals.loc[(proposals.index > end) & (proposals['label'] == 1)]
        else:
            fps = proposals.loc[(proposals.index > end) & (proposals.index < anomaly_windows[i + 1][0]) & (proposals['label'] == 1)]

        for index, fp in fps.iterrows():
            d = len(proposals.loc[(proposals.index >= end) & (proposals.index <= index)].index)-1

            if d >= max_d:
                all_sigma.append(-1*A_fp)
            else:
                y = (d/max_d)*math.e
                sigma = (2/(1+math.e**(5*y)))-1
                all_sigma.append(sigma*A_fp)
    raw_score = sum(all_sigma)

    null = -len(anomaly_windows)*A_fn
    NAB_score = (raw_score-null)/(len(anomaly_windows)-null)
    return NAB_score


def mean_absolute_error(y_true, y_score):
    return np.mean(np.abs(np.array(y_true)-np.array(y_score)))


def mean_squared_error(y_true, y_score):
    return np.mean((np.array(y_true)-np.array(y_score))**2)


def root_mean_squared_error(y_true, y_score):
    return np.sqrt(mean_squared_error(y_true, y_score))


def mean_absolute_percentage_error(y_true, y_score):
    return np.sum(np.abs(np.divide(np.array(y_true) - np.array(y_score), np.array(y_true))))/len(y_true)


def precision_recall_curve_values(y_true, y_score):
    data = pd.DataFrame({'y_true': y_true,
                         'y_score': y_score})
    data = data.sort_values(by='y_score', ascending=False)
    data['tp'] = np.where(data['y_true'] == 1, 1, 0)
    data['fp'] = np.where(data['y_true'] == 0, 1, 0)
    tp = np.cumsum(data['tp'].values)
    fp = np.cumsum(data['fp'].values)
    n_anns = len(data.loc[data['y_true'] == 1].index)

    precision = np.divide(tp, np.add(tp, fp))
    recall = np.divide(tp, n_anns)

    thresholds = np.unique(data['y_score'])
    confidence = data['y_score'].values
    rel_indexes = []
    for t in thresholds:
        indexes = np.where(confidence == t)[0]
        for i in indexes:
            precision[i] = precision[indexes[-1]]
            recall[i] = recall[indexes[-1]]
        rel_indexes.append(indexes[0])

    precision = precision[np.sort(rel_indexes)]
    recall = recall[np.sort(rel_indexes)]

    one = np.ones(1)
    zero = np.zeros(1)

    recall = np.concatenate([zero, recall])
    precision = np.concatenate([one, precision])

    if recall[-1] != 1:
        p_value = np.zeros(1)
        p_value[0] = tp[-1] / len(tp)
        recall = np.concatenate([recall, one])
        precision = np.concatenate([precision, p_value])
    recall = np.flip(recall)
    precision = np.flip(precision)

    return precision, recall
