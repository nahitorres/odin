import numpy as np
from numbers import Number
from odin.classes.timeseries.anomaly_matching_strategies.matching_strategy_interface import AnomalyMatchingStrategyInterface

from odin.classes.strings import *
from odin.utils.env import get_root_logger

class AnomalyMatchingStrategyIntervalToInterval(AnomalyMatchingStrategyInterface):

    def __init__(self, iou_threshold=0.5):
        if not isinstance(iou_threshold, Number):
            raise TypeError(err_type.format("iou_threshold"))
        elif not (0 < iou_threshold <= 1):
            raise ValueError(err_value.format("iou_threshold", "0 < iou_threshold <= 1"))

        self._iou_threshold = iou_threshold
        super().__init__()
        
    def get_iou_threshold(self):
        """Gets the iou threshold.
        
        Returns
        -------
        iou_threshold : float
            The threshold to be used with iou.
        """
        return self._iou_threshold
    
    def set_iou_threshold(self, iou_threshold):
        """
        Set the Intersection Over Union threshold

        Parameters
        ----------
        iou_threshold: float
            Intersection Over Union threshold
        """
        if not isinstance(iou_threshold, Number):
            get_root_logger().error(err_type.format("iou_threshold"))
            return -1
        if not (0 < iou_threshold <= 1):
            get_root_logger().error(err_value.format("iou_threshold", "0 < iou_threshold <= 1"))
            return -1

        self._iou_threshold = iou_threshold

    def get_confusion_matrix(self, y_true, y_score, threshold, inverse_threshold=False, min_consecutive_samples=1):
        """
        If iou(gt_interval, predicted_interval) >= iou_threshold --> TP
        """
        data = self._get_data(y_true, y_score, threshold, inverse_threshold, min_consecutive_samples)

        gt_intervals = data['y_true'].values - data['y_true'].shift(1).values
        gt_intervals[0] = data['y_true'].values[0]

        gt_start_intervals = np.where(gt_intervals == 1)[0]
        gt_end_intervals = np.where(gt_intervals == -1)[0] -1

        y_pred_intervals = data['y_pred'].values - data['y_pred'].shift(1).values
        y_pred_intervals[0] = data['y_pred'].values[0]
        y_pred_start_intervals = np.where(y_pred_intervals == 1)[0]
        y_pred_end_intervals = np.where(y_pred_intervals == -1)[0] -1


        tp = 0
        fp = 0

        for start, end in zip(y_pred_start_intervals, y_pred_end_intervals):
            start_ix = np.where((gt_start_intervals <= end) & ((gt_end_intervals >= start)))[0]
            counter = 0
            predicted_interval = [v for v in range(start, end+1)]
            for i in start_ix:
                gt_interval = [v for v in range(gt_start_intervals[i], gt_end_intervals[i]+1)]
                inter = len(list(set(gt_interval) & set(predicted_interval)))
                iou_v = inter/(len(predicted_interval) + len(gt_interval) - inter)
                if iou_v >= self._iou_threshold:
                    tp += 1
                    counter += 1
                break
            
            if counter == 0:
                fp += 1

        fn = len(gt_start_intervals) - tp
        tn = len(data.loc[(data['y_true'] == 0) & (data['y_pred'] == 0)].index)


        cm = np.empty((2, 2))
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp

        return cm