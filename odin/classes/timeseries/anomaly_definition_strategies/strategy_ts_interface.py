import abc

import numpy as np
import pandas as pd


class AnomalyDefinitionStrategyTSInterface(metaclass=abc.ABCMeta):
    """It is the interface for evaluator objects for time series.
    """
    
    def __init__(self):
        super().__init__()
        
        self._inverse_threshold = False
    
    @abc.abstractmethod
    def get_anomaly_scores(self, observations: pd.DataFrame,
                           proposals: pd.DataFrame) -> np.ndarray:
        """Gets the anomaly errors of the model.
        
        Parameters
        ----------
        observations : DataFrame
            The DataFrame containing the ground truth.
            
        proposals : DataFrame
            The DataFrame containing the proposals to be used to compute the
            anomaly errors.

        Returns
        -------
        errors : ndarray
            The errors computed by between observations and proposals.
        """
        pass
    
    # TODO: refactor in check_proposals_format
    @abc.abstractmethod
    def is_proposals_file_valid(self, proposals: pd.DataFrame) -> bool:
        """Detects if the proposals file is correctly formatted.
        
        Parameters
        ----------
        proposals : DataFrame
            The DataFrame containing the proposals to be used to compute the
            anomaly errors.

        Returns
        -------
        is_proposal_file_valid : bool
            True if the proposals file is correctly formatted, False otherwise.
        """
        pass
    
    def is_inverse_threshold(self):
        """States if the threshold should be treated inversely.
        
        Returns
        -------
        is_inverse_threshold : bool
            True if threshold must be treated differently.
        """
        return self._inverse_threshold