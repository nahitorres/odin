from odin.classes.timeseries.anomaly_matching_strategies.matching_strategy_interface import AnomalyMatchingStrategyInterface
import numpy as np

class AnomalyMatchingStrategyPointToPoint(AnomalyMatchingStrategyInterface):

    def get_confusion_matrix(self, y_true, y_score, threshold, inverse_threshold=False, min_consecutive_samples=1):
        data = self._get_data(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples)

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