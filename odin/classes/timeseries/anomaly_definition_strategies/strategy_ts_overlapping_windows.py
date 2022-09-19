from enum import Enum

import numpy as np
import pandas as pd
from odin.classes import Metrics

from odin.classes.strings import err_type, err_value
from odin.classes.timeseries.anomaly_definition_strategies import AnomalyDefinitionStrategyTSInterface


class OverlappingTruncation(Enum):
    NEITHER = 0,
    HEAD = 1,
    TAIL = 2,
    BOTH = 3


class AnomalyDefinitionStrategyTSOverlappingWindows(AnomalyDefinitionStrategyTSInterface):
    """Evaluates the errors of overlapping windows.
    
    Parameters
    ----------
    window_size : int
        It is the window size of the model.
        
    truncation_flag : OverlappingTruncation
        It is the way in which truncation must be performed.
    """
    __available_error_methods = [Metrics.MAE, Metrics.MSE]
    
    def __init__(self, window_size: int,
                 truncation_flag: OverlappingTruncation = OverlappingTruncation.BOTH,
                 window_error_method: Metrics = Metrics.MAE):
        if not isinstance(window_size, int):
            raise TypeError(err_type.format('window_size'))
        if not isinstance(truncation_flag, OverlappingTruncation):
            raise TypeError(err_type.format('truncation_flag'))
        if not isinstance(window_error_method, Metrics):
            raise TypeError(err_type.format('window_error_method'))
        
        if window_error_method not in self.__available_error_methods:
            raise ValueError(err_value.format('window_error_method', self.__available_error_methods))
        
        # TODO: implement also other truncation methods
        if truncation_flag != OverlappingTruncation.BOTH:
            raise NotImplementedError("Only BOTH is currently implemented")
        
        if window_size < 1:
            raise ValueError(err_value.format('window_size', 'Values in [1, N]'))
        
        super().__init__()
        
        self.window_size = window_size
        self.truncation_flag = truncation_flag
        self.window_error_method = window_error_method
        
    # TODO: implement also truncation methods different from BOTH
    def get_anomaly_scores(self, observations: pd.DataFrame,
                           proposals: pd.DataFrame) -> np.ndarray:
        
        c_name = observations.columns.difference(['anomaly', 'anomaly_window'])[0]
        gt_values = observations[c_name].values
        props_values = proposals[c_name].tolist()
        
        # create gt sliding window
        gt_windows = np.array([gt_values[i:i+self.window_size] for i in range(0, len(gt_values)-self.window_size+1)]).reshape(-1, self.window_size)

        y = props_values.copy()
        
        # padding all the lists with window_size length
        for i in range(0, self.window_size -1):
            tmp = np.zeros(self.window_size -1 -i).tolist()
            y[i] = tmp + y[i]
        for i, v in enumerate(range(len(y)-self.window_size+1, len(y))):
            tmp = np.zeros(i+1).tolist()
            y[v] = y[v] + tmp
        
        # now y can be converted to np.array with shape (len(gt_windows), window_size)
        y = np.array(y)

        
        # create predicted overlap windows 
        overlap_windows = []
        for i in range(len(y)-self.window_size+1):
            win = [y[v][self.window_size -1 - k] for k, v in enumerate(range(i, i+self.window_size))]
            overlap_windows.append(win)
        overlap_windows = np.array(overlap_windows)

        # calculate the error for each window
        if self.window_error_method == Metrics.MAE:
            win_errors = np.mean(np.abs(gt_windows - overlap_windows), axis=1)
        else:
            win_errors = np.mean((gt_windows - overlap_windows)**2, axis=1)
        
        errors = []
        for i in range(0, len(win_errors)-self.window_size+1):
            errors.append(win_errors[i:i+self.window_size])

        return np.array(errors)
    
    def is_proposals_file_valid(self, proposals: pd.DataFrame) -> bool:
        c_name = proposals.columns[0]
        return type(proposals[c_name].tolist()[0]) == list
    
        