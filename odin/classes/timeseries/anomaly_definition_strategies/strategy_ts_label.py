import numpy as np
import pandas as pd

from odin.classes.timeseries.anomaly_definition_strategies import AnomalyDefinitionStrategyTSInterface


class AnomalyDefinitionStrategyTSLabel(AnomalyDefinitionStrategyTSInterface):
    """It is a proxy strategy to be used with tasks of type label.
    """
    def __init__(self):
        super().__init__()
        
    def get_anomaly_scores(self, observations: pd.DataFrame,
                           proposals: pd.DataFrame) -> np.ndarray:
        return proposals['confidence'].values
    
    def is_proposals_file_valid(self, proposals: pd.DataFrame) -> bool:
        return "confidence" in proposals.columns
        