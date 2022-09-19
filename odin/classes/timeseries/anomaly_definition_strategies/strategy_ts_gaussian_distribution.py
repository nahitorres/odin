import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from odin.classes.timeseries.anomaly_definition_strategies import AnomalyDefinitionStrategyTSMeanCovarianceInterface


class AnomalyDefinitionStrategyTSGaussianDistribution(AnomalyDefinitionStrategyTSMeanCovarianceInterface):
    """Evaluates the errors using Gaussian distribution.
    """
    def __init__(self, mean_vec,
                 covariance_mat):
        super().__init__(mean_vec, covariance_mat)
        
        self._inverse_threshold = True
        
    def get_anomaly_scores(self, observations: pd.DataFrame,
                           proposals: pd.DataFrame) -> np.ndarray:
        c_name = observations.columns.difference(['anomaly', 'anomaly_window'])[0]
        gt_values = observations[c_name].values
        props_values = proposals[c_name].tolist()

        ahead_steps = len(max(props_values,key=len))
        
        # we compute the difference between true and predicted value at time j
        scores = np.array([np.abs(props_values[i] - gt_values[i]) for i in range(ahead_steps-1, len(gt_values)-ahead_steps)])
        scores = multivariate_normal.pdf(scores, mean=self.mean_vec, cov=self.covariance_mat, allow_singular=True)
        return scores