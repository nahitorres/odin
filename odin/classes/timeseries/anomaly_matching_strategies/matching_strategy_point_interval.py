from odin.classes.timeseries.anomaly_matching_strategies.matching_strategy_interface import AnomalyMatchingStrategyInterface
import numpy as np

class AnomalyMatchingStrategyPointToInterval(AnomalyMatchingStrategyInterface):

    def get_confusion_matrix(self, y_true, y_score, threshold, inverse_threshold=False, min_consecutive_samples=1):
        data = self._get_data(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples)

        gt_positions = np.where(data['y_true'] == 1)[0]

        intervals = data['y_pred'].values - data['y_pred'].shift(1).values
        intervals[0] = data['y_pred'].values[0]
        start_intervals = np.where(intervals == 1)[0]
        end_intervals = np.where(intervals == -1)[0] -1
        
        tp = 0
        fp = 0
        for start, end in zip(start_intervals, end_intervals):
            points_contained = 0
            for gt_p in gt_positions:
                if gt_p < start or gt_p > end:
                    continue
                points_contained += 1
            if points_contained > 0:
                tp += points_contained
            else:
                fp += 1
        
        fn = len(gt_positions) - tp
        tn = len(data.loc[(data['y_true'] == 0) & (data['y_pred'] == 0)].index)

        cm = np.empty((2, 2))
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp

        return cm