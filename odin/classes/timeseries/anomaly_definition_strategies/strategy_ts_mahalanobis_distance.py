import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from odin.classes.timeseries.anomaly_definition_strategies import AnomalyDefinitionStrategyTSMeanCovarianceInterface

class AnomalyDefinitionStrategyTSMahalanobisDistance(AnomalyDefinitionStrategyTSMeanCovarianceInterface):
    """Evaluates the errors using Mahalanobis distance.
    """
    def __init__(self, mean_vec,
                 covariance_mat):
        super().__init__(mean_vec, covariance_mat)
        
    def get_anomaly_scores(self, observations: pd.DataFrame,
                           proposals: pd.DataFrame) -> np.ndarray:
        gt_values = observations[observations.columns.difference(['anomaly', 'anomaly_window'])].values
        props_values = np.array(proposals["value"].tolist())
        
        # we compute the difference absolute error between true and predicted
        # value at time j
        errors = np.array([np.abs(gt_values[i, 0] - props_values[i, :]).tolist()
                           for i in range(props_values.shape[0])])
        diff = errors - self.mean_vec
        
        # covariance matrix is identical for each element
        try:
            inverse_mat = np.linalg.inv(self.covariance_mat)
        except LinAlgError as e:
            inverse_mat = np.linalg.pinv(self.covariance_mat)
        
        # from the matrix (N,N) the diagonal represent the matrix product between
        # the ith row of diff and the ith column of transpose(diff)
        scores_mat = diff @ inverse_mat @ np.transpose(diff)
        scores = np.diag(scores_mat)
        
        return scores