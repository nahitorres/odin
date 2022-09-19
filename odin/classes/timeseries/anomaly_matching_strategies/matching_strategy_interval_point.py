from odin.classes.timeseries.anomaly_matching_strategies.matching_strategy_interface import AnomalyMatchingStrategyInterface
import numpy as np

class AnomalyMatchingStrategyIntervalToPoint(AnomalyMatchingStrategyInterface):

    def get_confusion_matrix(self, y_true, y_score, threshold, inverse_threshold=False, min_consecutive_samples=1):
        data = self._get_data(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples)

        y_pred_positions = np.where(data['y_pred'] == 1)[0]

        intervals = data['y_true'].values - data['y_true'].shift(1).values
        intervals[0] = data['y_true'].values[0]
        start_intervals = np.where(intervals == 1)[0]
        end_intervals = np.where(intervals == -1)[0] -1
        
        tp = 0
        fp = len(y_pred_positions)
        for start, end in zip(start_intervals, end_intervals):
            all_tp = len(np.where((y_pred_positions >= start) & (y_pred_positions <= end))[0])
            if all_tp > 0:
                tp += 1
            fp -= all_tp
        
        fn = len(start_intervals) - tp
        tn = len(data.loc[(data['y_true'] == 0) & (data['y_pred'] == 0)].index)

        cm = np.empty((2, 2))
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp

        return cm