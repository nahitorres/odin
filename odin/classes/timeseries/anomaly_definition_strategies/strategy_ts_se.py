import numpy as np
import pandas as pd

from odin.classes.timeseries.anomaly_definition_strategies import AnomalyDefinitionStrategyTSInterface


class AnomalyDefinitionStrategyTSSE(AnomalyDefinitionStrategyTSInterface):
    """Evaluates the errors using Squared Error.
    """
    def __init__(self):
        super().__init__()
    
    def get_anomaly_scores(self, observations: pd.DataFrame,
                           proposals: pd.DataFrame) -> np.ndarray:
        c_name = observations.columns.difference(['anomaly', 'anomaly_window'])[0]
        gt_values = observations[c_name].values.reshape(-1, 1)
        props_values = proposals[c_name].values.reshape(-1, 1)
        
        return np.sum(np.abs(gt_values - props_values)**2, axis=1)
    
    def is_proposals_file_valid(self, proposals: pd.DataFrame) -> bool:
        c_name = proposals.columns[0]
        props_values = proposals[c_name].values
        return len(props_values.shape) == 1