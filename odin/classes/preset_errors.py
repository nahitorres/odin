import numpy as np
import pandas as pd
from odin.utils import get_root_logger
from odin.classes.strings import err_type
from collections.abc import Sequence
from numbers import Number

def before_error(y_true, y_score, threshold, observations = None, parameters_dict = None):
    '''
    Computes the errors of the 'BEFORE' type among FP errors.
    
    Parameters
    ----------
    y_true: array-like
        Array containing the GT values
        
    y_score: array-like
        Array containing the predicted values
        
    threshold: float
        Threshold to transform y_score into a binary array
        
    observations: DataFrame
        DataFrame containing the dataset observations
        
    parameters_dict: dict
        Dictionary containing the parameters needed to compute the error.
        
    Returns
    -------
    before: int
        Count of errors of this type
        
    distances: array-like
        Array containing the distances of each error with respect to the reference GT anomaly
        
    errors_index: array-like
        Array containing the indexes of the errors of this type
        
    matching: DataFrame
        Pandas DataFrame containing the GT and predictions and the type of error (e.g., FP)
    
    '''
    
    if not isinstance(y_true, (Sequence, np.ndarray)):
        raise TypeError(err_type.format('y_true'))
        
    if not isinstance(y_score, (Sequence, np.ndarray)):
        raise TypeError(err_type.format('y_score'))
        
    assert len(y_true) == len(y_score), "y_true and y_score must have the same length"
    
    if not isinstance(threshold, Number):
        raise TypeError(err_type.format('threshold'))
        
    if not isinstance(observations, pd.DataFrame):
        raise TypeError(err_type.format('observations'))
        
    if not isinstance(parameters_dict, dict):
        raise TypeError(err_type.format('parameters_dict'))
            
            
    closely_threshold = parameters_dict['closely_threshold']
    
    y_pred = np.where(y_score >= threshold, 1, 0)

    matching = pd.DataFrame(data={'y_true': y_true,
                                  'y_pred': y_pred},
                            index=observations.index)
    matching['eval'] = 0
    matching.loc[matching['y_true'] == 1, 'eval'] = 1
    matching.loc[(matching['y_true'] == 0) & (matching['y_pred'] == 1), 'eval'] = -1

    before = 0

    anomalies_pos = np.where(matching['eval'] == 1)[0]

    previous_anomaly_pos = -1
    anomaly_pos_index = 0
    next_anomaly_pos = anomalies_pos[0]
    is_previous_anomaly = False
    is_next_anomaly = False

    distances = []
    index_values = matching.index
    errors_index = []
    
    for i, v in enumerate(matching['eval'].values):
        if (i > next_anomaly_pos) and (next_anomaly_pos != -1):
            previous_anomaly_pos = next_anomaly_pos
            anomaly_pos_index += 1
            next_anomaly_pos = anomalies_pos[anomaly_pos_index] if anomaly_pos_index < len(anomalies_pos) else -1
        is_next_anomaly = False
        if i < len(matching)-1 and matching['y_pred'].values[i+1] == 1:
            is_next_anomaly = True
        if v == -1:
            previous_d = i - previous_anomaly_pos if previous_anomaly_pos != -1 else float('inf')
            next_d = i - next_anomaly_pos if next_anomaly_pos != -1 else float('inf')

            d = previous_d if previous_d < np.abs(next_d) else next_d
            
            if np.abs(d) >= closely_threshold and d < 0:
                before += 1
                distances.append(d)
                errors_index.append(index_values[i])
            

            is_previous_anomaly = True

        else:
            is_previous_anomaly = False
                        
    return before, distances, errors_index, matching


def closely_before_error(y_true, y_score, threshold, observations = None, parameters_dict = None):
    '''
    Computes the errors of the 'CLOSELY BEFORE' type among FP errors.
    
    Parameters
    ----------
    y_true: array-like
        Array containing the GT values
        
    y_score: array-like
        Array containing the predicted values
        
    threshold: float
        Threshold to transform y_score into a binary array
        
    observations: DataFrame
        DataFrame containing the dataset observations
        
    parameters_dict: dict
        Dictionary containing the parameters needed to compute the error.
        
    Returns
    -------
    before: int
        Count of errors of this type
        
    distances: array-like
        Array containing the distances of each error with respect to the reference GT anomaly
        
    errors_index: array-like
        Array containing the indexes of the errors of this type
        
    matching: DataFrame
        Pandas DataFrame containing the GT and predictions and the type of error (e.g., FP)
    
    '''
    
    if not isinstance(y_true, (Sequence, np.ndarray)):
        raise TypeError(err_type.format('y_true'))
        
    if not isinstance(y_score, (Sequence, np.ndarray)):
        raise TypeError(err_type.format('y_score'))
        
    assert len(y_true) == len(y_score), "y_true and y_score must have the same length"
    
    if not isinstance(threshold, Number):
        raise TypeError(err_type.format('threshold'))
        
    if not isinstance(observations, pd.DataFrame):
        raise TypeError(err_type.format('observations'))
        
    if not isinstance(parameters_dict, dict):
        raise TypeError(err_type.format('parameters_dict'))
        
    closely_threshold = parameters_dict['closely_threshold']
    
    y_pred = np.where(y_score >= threshold, 1, 0)

    matching = pd.DataFrame(data={'y_true': y_true,
                                  'y_pred': y_pred},
                            index=observations.index)
    matching['eval'] = 0
    matching.loc[matching['y_true'] == 1, 'eval'] = 1
    matching.loc[(matching['y_true'] == 0) & (matching['y_pred'] == 1), 'eval'] = -1

    closely_before = 0

    anomalies_pos = np.where(matching['eval'] == 1)[0]

    previous_anomaly_pos = -1
    anomaly_pos_index = 0
    next_anomaly_pos = anomalies_pos[0]
    is_previous_anomaly = False
    is_next_anomaly = False

    distances = []
    index_values = matching.index
    errors_index = []
    
    for i, v in enumerate(matching['eval'].values):
        if (i > next_anomaly_pos) and (next_anomaly_pos != -1):
            previous_anomaly_pos = next_anomaly_pos
            anomaly_pos_index += 1
            next_anomaly_pos = anomalies_pos[anomaly_pos_index] if anomaly_pos_index < len(anomalies_pos) else -1
        is_next_anomaly = False
        if i < len(matching)-1 and matching['y_pred'].values[i+1] == 1:
            is_next_anomaly = True
        if v == -1:
            previous_d = i - previous_anomaly_pos if previous_anomaly_pos != -1 else float('inf')
            next_d = i - next_anomaly_pos if next_anomaly_pos != -1 else float('inf')

            d = previous_d if previous_d < np.abs(next_d) else next_d
            
            if np.abs(d) <= closely_threshold and d < 0:
                closely_before += 1
                distances.append(d)
                errors_index.append(index_values[i])
            

            is_previous_anomaly = True

        else:
            is_previous_anomaly = False
                        
    return closely_before, distances, errors_index, matching



def closely_after_error(y_true, y_score, threshold, observations = None, parameters_dict = None):
    '''
    Computes the errors of the 'CLOSELY AFTER' type among FP errors.
    
    Parameters
    ----------
    y_true: array-like
        Array containing the GT values
        
    y_score: array-like
        Array containing the predicted values
        
    threshold: float
        Threshold to transform y_score into a binary array
        
    observations: DataFrame
        DataFrame containing the dataset observations
        
    parameters_dict: dict
        Dictionary containing the parameters needed to compute the error.
        
    Returns
    -------
    before: int
        Count of errors of this type
        
    distances: array-like
        Array containing the distances of each error with respect to the reference GT anomaly
        
    errors_index: array-like
        Array containing the indexes of the errors of this type
        
    matching: DataFrame
        Pandas DataFrame containing the GT and predictions and the type of error (e.g., FP)
    
    '''
    
    if not isinstance(y_true, (Sequence, np.ndarray)):
        raise TypeError(err_type.format('y_true'))
        
    if not isinstance(y_score, (Sequence, np.ndarray)):
        raise TypeError(err_type.format('y_score'))
        
    assert len(y_true) == len(y_score), "y_true and y_score must have the same length"
    
    if not isinstance(threshold, Number):
        raise TypeError(err_type.format('threshold'))
        
    if not isinstance(observations, pd.DataFrame):
        raise TypeError(err_type.format('observations'))
        
    if not isinstance(parameters_dict, dict):
        raise TypeError(err_type.format('parameters_dict'))
        
        
    closely_threshold = parameters_dict['closely_threshold']
    
    y_pred = np.where(y_score >= threshold, 1, 0)

    matching = pd.DataFrame(data={'y_true': y_true,
                                  'y_pred': y_pred},
                            index=observations.index)
    matching['eval'] = 0
    matching.loc[matching['y_true'] == 1, 'eval'] = 1
    matching.loc[(matching['y_true'] == 0) & (matching['y_pred'] == 1), 'eval'] = -1

    closely_after = 0

    anomalies_pos = np.where(matching['eval'] == 1)[0]

    previous_anomaly_pos = -1
    anomaly_pos_index = 0
    next_anomaly_pos = anomalies_pos[0]
    is_previous_anomaly = False
    is_next_anomaly = False

    distances = []
    index_values = matching.index
    errors_index = []
    
    for i, v in enumerate(matching['eval'].values):
        if (i > next_anomaly_pos) and (next_anomaly_pos != -1):
            previous_anomaly_pos = next_anomaly_pos
            anomaly_pos_index += 1
            next_anomaly_pos = anomalies_pos[anomaly_pos_index] if anomaly_pos_index < len(anomalies_pos) else -1
        is_next_anomaly = False
        if i < len(matching)-1 and matching['y_pred'].values[i+1] == 1:
            is_next_anomaly = True
        if v == -1:
            previous_d = i - previous_anomaly_pos if previous_anomaly_pos != -1 else float('inf')
            next_d = i - next_anomaly_pos if next_anomaly_pos != -1 else float('inf')

            d = previous_d if previous_d < np.abs(next_d) else next_d
            
            if np.abs(d) <= closely_threshold and d > 0:
                closely_after += 1
                distances.append(d)
                errors_index.append(index_values[i])
            

            is_previous_anomaly = True

        else:
            is_previous_anomaly = False
                        
    return closely_after, distances, errors_index, matching



def after_error(y_true, y_score, threshold, observations = None, parameters_dict = None):
    '''
    Computes the errors of the 'AFTER' type among FP errors.
    
    Parameters
    ----------
    y_true: array-like
        Array containing the GT values
        
    y_score: array-like
        Array containing the predicted values
        
    threshold: float
        Threshold to transform y_score into a binary array
        
    observations: DataFrame
        DataFrame containing the dataset observations
        
    parameters_dict: dict
        Dictionary containing the parameters needed to compute the error.
        
    Returns
    -------
    before: int
        Count of errors of this type
        
    distances: array-like
        Array containing the distances of each error with respect to the reference GT anomaly
        
    errors_index: array-like
        Array containing the indexes of the errors of this type
        
    matching: DataFrame
        Pandas DataFrame containing the GT and predictions and the type of error (e.g., FP)
    
    '''
    
    if not isinstance(y_true, (Sequence, np.ndarray)):
        raise TypeError(err_type.format('y_true'))
        
    if not isinstance(y_score, (Sequence, np.ndarray)):
        raise TypeError(err_type.format('y_score'))
        
    assert len(y_true) == len(y_score), "y_true and y_score must have the same length"
    
    if not isinstance(threshold, Number):
        raise TypeError(err_type.format('threshold'))
        
    if not isinstance(observations, pd.DataFrame):
        raise TypeError(err_type.format('observations'))
        
    if not isinstance(parameters_dict, dict):
        raise TypeError(err_type.format('parameters_dict'))
        
        
    closely_threshold = parameters_dict['closely_threshold']
    
    y_pred = np.where(y_score >= threshold, 1, 0)

    matching = pd.DataFrame(data={'y_true': y_true,
                                  'y_pred': y_pred},
                            index=observations.index)
    matching['eval'] = 0
    matching.loc[matching['y_true'] == 1, 'eval'] = 1
    matching.loc[(matching['y_true'] == 0) & (matching['y_pred'] == 1), 'eval'] = -1

    after = 0

    anomalies_pos = np.where(matching['eval'] == 1)[0]

    previous_anomaly_pos = -1
    anomaly_pos_index = 0
    next_anomaly_pos = anomalies_pos[0]
    is_previous_anomaly = False
    is_next_anomaly = False

    distances = []
    index_values = matching.index
    errors_index = []
    
    for i, v in enumerate(matching['eval'].values):
        if (i > next_anomaly_pos) and (next_anomaly_pos != -1):
            previous_anomaly_pos = next_anomaly_pos
            anomaly_pos_index += 1
            next_anomaly_pos = anomalies_pos[anomaly_pos_index] if anomaly_pos_index < len(anomalies_pos) else -1
        is_next_anomaly = False
        if i < len(matching)-1 and matching['y_pred'].values[i+1] == 1:
            is_next_anomaly = True
        if v == -1:
            previous_d = i - previous_anomaly_pos if previous_anomaly_pos != -1 else float('inf')
            next_d = i - next_anomaly_pos if next_anomaly_pos != -1 else float('inf')

            d = previous_d if previous_d < np.abs(next_d) else next_d
            
            if np.abs(d) >= closely_threshold and d > 0:
                after += 1
                distances.append(d)
                errors_index.append(index_values[i])
            

            is_previous_anomaly = True

        else:
            is_previous_anomaly = False
                        
    return after, distances, errors_index, matching