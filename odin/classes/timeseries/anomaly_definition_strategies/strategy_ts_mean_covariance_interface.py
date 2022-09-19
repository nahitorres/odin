import abc
import warnings

import numpy as np
import pandas as pd

from odin.classes.timeseries.anomaly_definition_strategies import AnomalyDefinitionStrategyTSInterface


class AnomalyDefinitionStrategyTSMeanCovarianceInterface(AnomalyDefinitionStrategyTSInterface, metaclass=abc.ABCMeta):
    """Abstract class implementing evaluators using mean and covariance.
    
    Parameters
    ----------
    mean_vec : array-like
        Object storing the mean vector.
    
    covariance_mat : array-like
        Object storing the covariance matrix.
    """
    def __init__(self, mean_vec,
                 covariance_mat):
        mean_vec = np.array(mean_vec)
        covariance_mat = np.array(covariance_mat)
        
        if mean_vec.ndim != 1:
            raise ValueError("Vectors have only 1 dimension")
        
        if covariance_mat.ndim != 2:
            raise ValueError("Covariance matrix have exactly 2 dimensions")
        
        if mean_vec.shape[0] != covariance_mat.shape[0] or mean_vec.shape[0] != covariance_mat.shape[1]:
            raise ValueError(
                "Covariance matrix must be NxN where N is mean shape")
        
        if mean_vec.shape[0] == 1:
            warnings.warn("You are using {} with only 1 predicted value, generally it is used with X>1.".format(self.__class__.__name__), UserWarning)
        
        super().__init__()
        
        self.mean_vec = mean_vec
        self.covariance_mat = covariance_mat
        
    def is_proposals_file_valid(self, proposals: pd.DataFrame) -> bool:
        c_name = proposals.columns[0]
        return type(proposals[c_name].tolist()[0]) == list
