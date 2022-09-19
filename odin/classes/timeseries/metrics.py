import math
import numpy as np
import pandas as pd
from odin.classes.timeseries.anomaly_matching_strategies import AnomalyMatchingStrategyPointToPoint


def matthews_correlation_coefficient(y_true, y_score, threshold, evaluation_type=AnomalyMatchingStrategyPointToPoint(), inverse_threshold=False, min_consecutive_samples=1):
    tn, fp, fn, tp = evaluation_type.get_confusion_matrix(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples).ravel()
    return (tp*tn - fp*fn) / np.sqrt((tp+fn)*(fp+tn)*(fn+tn)*(tp+fp))


def accuracy(y_true, y_score, threshold, evaluation_type=AnomalyMatchingStrategyPointToPoint(), inverse_threshold=False, min_consecutive_samples=1):
    tn, fp, fn, tp = evaluation_type.get_confusion_matrix(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples).ravel()

    return (tp+tn)/(tp+fp+fn+tn)


def precision_score(y_true, y_score, threshold, evaluation_type=AnomalyMatchingStrategyPointToPoint(), inverse_threshold=False, min_consecutive_samples=1):
    _, fp, _, tp = evaluation_type.get_confusion_matrix(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples).ravel()

    if tp == 0 and fp == 0:
        return 0

    return tp/(tp+fp)

def recall_score(y_true, y_score, threshold, evaluation_type=AnomalyMatchingStrategyPointToPoint(), inverse_threshold=False, min_consecutive_samples=1):
    _, _, fn, tp = evaluation_type.get_confusion_matrix(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples).ravel()

    if tp == 0 and fn == 0:
        return 0

    return tp/(tp+fn)


def f1_score(y_true, y_score, threshold, evaluation_type=AnomalyMatchingStrategyPointToPoint(), inverse_threshold=False, min_consecutive_samples=1):
    precision = precision_score(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)
    recall = recall_score(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)

    if precision == 0 and recall == 0:
        return 0

    return 2*precision*recall/(precision+recall)


def f_beta_score(y_true, y_score, threshold, beta=0.1, evaluation_type=AnomalyMatchingStrategyPointToPoint(), inverse_threshold=False, min_consecutive_samples=1):
    precision = precision_score(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)
    recall = recall_score(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)

    if precision == 0 and recall == 0:
        return 0
    
    beta_squared = beta**2
    return (1+beta_squared)* precision*recall/((beta_squared*precision)+recall)



def false_alarm_rate(y_true, y_score, threshold, evaluation_type=AnomalyMatchingStrategyPointToPoint(), inverse_threshold=False, min_consecutive_samples=1):
    tn, fp, _, _ = evaluation_type.get_confusion_matrix(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples).ravel()

    if fp == 0 and tn == 0:
        return 0

    return fp/(fp+tn)


def miss_alarm_rate(y_true, y_score, threshold, evaluation_type=AnomalyMatchingStrategyPointToPoint(), inverse_threshold=False, min_consecutive_samples=1):
    _, _, fn, tp = evaluation_type.get_confusion_matrix(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples).ravel()

    if fn == 0 and tp == 0:
        return 0

    return fn/(fn+tp)


def nab_score(anomaly_windows, proposals, y_score, threshold, A_tp, A_fp, A_fn, inverse_threshold=False, min_consecutive_samples=1):
    all_sigma = []
    # proposals['label'] = np.where(y_score >= threshold, 1, 0)

    proposals['label'] = AnomalyMatchingStrategyPointToPoint()._get_y_pred(y_score, threshold, inverse_threshold, min_consecutive_samples)

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
    return 100 * np.mean(np.abs(np.divide(np.array(y_true) - np.array(y_score), np.array(y_true))))


def mean_absolute_ranged_relative_error(y_true, y_score):
    return 100 * np.mean(np.abs(np.divide(np.array(y_true) - np.array(y_score), np.max(np.array(y_true)) - np.min(np.array(y_true)))))


def overall_percentage_error(y_true, y_score):
    y_true_sum = np.sum(np.array(y_true))
    return 100 * np.abs((y_true_sum - np.sum(np.array(y_score))) / y_true_sum)


def coefficient_of_variation(y_true, y_score):
    return 100*root_mean_squared_error(y_true, y_score)/np.mean(y_true)

# r^2
def coefficient_of_determination(y_true, y_score):
    np_ytrue = np.array(y_true)
    np_yscore = np.array(y_score)
    mean_ytrue = np.mean(np_ytrue)
    ss_res = np.sum((np_ytrue-np_yscore)**2)
    ss_tot = np.sum((np_ytrue-mean_ytrue)**2)
    return 1 - (ss_res/ss_tot)


def root_mean_squared_log_error(y_true, y_score):
    return np.sqrt(np.mean((np.log(np.array(y_true)+1) - np.log(np.array(y_score)+1))**2))


def symmetric_mean_absolute_percentage_error(y_true, y_score):
    np_ytrue = np.array(y_true)
    np_yscore = np.array(y_score)
    diff = np.abs(np_ytrue - np_yscore)
    abs_sum = np.abs(np_ytrue) + np.abs(np_yscore)
    return 200 * np.mean(np.divide(diff, abs_sum))


# TODO , evaluation_type=TPEvaluation.POINT_POINT, inverse_threshold=False, min_consecutive_samples=1
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
