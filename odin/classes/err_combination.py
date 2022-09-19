import numpy as np
import pandas as pd
from odin.classes import Errors, CustomError
from odin.classes.preset_errors import before_error, closely_before_error, after_error, closely_after_error

class ErrCombination:
    def __init__(self, name, errors, error_names):
        """
        This class provides an interface for combining custom errors.

        Parameters
        ----------
        name: str
            Name of the custom error combination
            
        errors: array-like
            List of errors to consider
            
        error_names: array-like
            List of errors names to assign in the plots
        """
        
        self.__name = name
        self.__errors = errors
        self.__error_names = error_names
        
        
    def compute_errors_combination(self, y_true, y_score, threshold, observations, parameters_dicts):
        """
        This function allows to combine errors, which can be either custom or pre-defined (preset).

        Parameters
        ----------
        y_true: array-like
            Array containing the GT values
            
        y_score: array-like
            Array containing the predicted values
            
        threshold: float
            Number to convert y_score into a binary array
            
        observations: DataFrame
            DataFrame containing the data
            
        parameters_dicts: array-like
            List of dictionary containing the parameters for each error in the list
            
        
        Returns
        -------
        count_dict: dict
            Dictionary containing, for each error type, the number of occurrencies
        
        errors_index_dict: dict
            Dictionary containing, for each error type, the list of indexes where it is found
            
        matching: DataFrame
            DataFrame indicating the presence of false positives
                    
        """
        
        errors = self.__errors
        
        count_dict = dict()
        distances_dict = dict()
        errors_index_dict = dict()
        
        y_pred = np.where(y_score >= threshold, 1, 0)

        matching = pd.DataFrame(data={'y_true': y_true,
                                      'y_pred': y_pred},
                                index=observations.index)
        matching['eval'] = 0
        matching.loc[matching['y_true'] == 1, 'eval'] = 1
        matching.loc[(matching['y_true'] == 0) & (matching['y_pred'] == 1), 'eval'] = -1
        
        count, distances, errors_index = None, None, None
            
        for idx, error in enumerate(errors):
            
            if not isinstance(error, Errors) and not isinstance(error, CustomError):
                raise TypeError(err_type.format('error'))
                
            parameters_dict = None
            
            if len(parameters_dicts) > 1:
                parameters_dict = parameters_dicts[idx]
                
                if not isinstance(parameters_dict, dict):
                    raise TypeError(err_type.format('parameters_dict'))
                
            elif len(parameters_dicts) == 1:
                parameters_dict = parameters_dicts[0]
                
                if not isinstance(parameters_dict, dict):
                    raise TypeError(err_type.format('parameters_dict'))
                
            error_name = self.__error_names[idx]
            
            if isinstance(error, Errors):
                if error == Errors.BEFORE:
                    count, distances, errors_index, _ = before_error(y_true, y_score, threshold, observations, parameters_dict)
                elif error == Errors.CLOSELY_BEFORE:
                    count, distances, errors_index, _ = closely_before_error(y_true, y_score, threshold, observations, parameters_dict)
                elif error == Errors.AFTER:
                    count, distances, errors_index, _ = after_error(y_true, y_score, threshold, observations, parameters_dict)
                elif error == Errors.CLOSELY_AFTER:
                    count, distances, errors_index, _ = closely_after_error(y_true, y_score, threshold, observations, parameters_dict)
            elif isinstance(error, CustomError):            
                count, distances, errors_index, _ = error.compute_error(y_true, y_score, threshold, observations, parameters_dict)
            
            elif isinstance(error, CustomError):            
                count, distances, errors_index, _ = error.compute_error(y_true, y_score, threshold, observations, parameters_dict)
            
            count_dict[error_name] = count
            distances_dict[error_name] = distances
            errors_index_dict[error_name] = errors_index
            
        return count_dict, distances_dict, errors_index_dict, matching