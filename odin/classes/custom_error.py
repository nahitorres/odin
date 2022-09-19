import abc

class CustomError(metaclass = abc.ABCMeta):
    def __init__(self, name):
        """
        This class provides an interface for a custom error.

        Parameters
        ----------
        name: str
            Name of the custom error
        """
        
        self.__name = name
        
    @abc.abstractmethod
    def compute_error(self, y_true, y_score, threshold, observations, parameters_dict = None):
        """
        Custom error evaluation

        Parameters
        ----------
        y_true: array-like
            Array containing binary values indicating the presence (1) or absence (0) of an anomaly in the GT
        
        y_score: array-like
            Array containing the predicted values
            
        threshold: float
            Threshold to transform y_score in y_pred, a binary array
            
        observations: DataFrame
            DataFrame with the observations on which the anomalies are computed
            
        parameters_dict: dict
            Dictionary of the values associated with parameters used to compute the custom error
            

        Returns
        -------
        count: int
            Represents the number of errors for this category as a value
            
        distances: array-like
            List of distances of each error from the GT
            
        errors_index: array-like
            List of indexes of each error
            
        matching: DataFrame
            DataFrame indicating, in the 'eval' column, the presence of a true anomaly (1) or a FP anomaly (-1) or something else (0)
            
        """
        pass