import abc
import numpy as np
import pandas as pd

class AnomalyMatchingStrategyInterface(metaclass=abc.ABCMeta):
    """
        Interface class for the strategy definition for the True Positive, False Positive, False Negative 
        and True Negative identification.
    """
    
    @abc.abstractmethod
    def get_confusion_matrix(self, y_true, y_score, threshold, inverse_threshold=False, min_consecutive_samples=1):
        """
        Calculates TP, FP, FN, TN based on the strategy

        Parameters
        ----------
        y_true: array-like
            Ground Truth
        y_score: array-like
            Predictions scores
        threshold: float
            Threshold value applied to y_score
        inverse_threshold: bool, optional
            Indicates whether the threshold must be applied inverted (default is False)
        min_consecutive_samples: int, optional
            Minimum consecutive samples to be considered as anomalous. The consecutive samples below this threshold are ignored (default is 1)
        Returns
        -------
        np.array
            Confusion Matrix
        """
        pass

    def _get_y_pred(self, y_score, threshold, inverse_threshold=False, min_consecutive_samples=1):
        """
        Calculates the predictions based on the anomaly score

        Parameters
        ----------
        y_score: array-like
            Predictions scores
        threshold: float
            Threshold value applied to y_score
        inverse_threshold: bool, optional
            Indicates whether the threshold must be applied inverted (default is False)
        min_consecutive_samples: int, optional
            Minimum consecutive samples to be considered as anomalous. The consecutive samples below this threshold are ignored (default is 1)
        Returns
        -------
        np.array
            Predictions in boolean format
        """
        if len(y_score.shape) > 1:
            anomalous_sequences = y_score >= threshold
            y_pred = np.where(np.all(anomalous_sequences, axis=1), 1, 0)
        else:
            y_pred = np.where(y_score >= threshold, 1, 0) if not inverse_threshold else np.where(y_score <= threshold, 1, 0)
        
        # filter on min_consecutive_salmples
        y_pred = self._filter_min_consecutive_samples(y_pred.copy(), min_consecutive_samples)
        return y_pred

    def _get_data(self, y_true, y_score, threshold, inverse_threshold=False, min_consecutive_samples=1):
        """
        Creates a DataFrame with the ground truth, the predictions scores and the predictions in boolean format

        Parameters
        ----------
        y_true: array-like
            Ground Truth
        y_score: array-like
            Predictions scores
        threshold: float
            Threshold value applied to y_score
        inverse_threshold: bool, optional
            Indicates whether the threshold must be applied inverted (default is False)
        min_consecutive_samples: int, optional
            Minimum consecutive samples to be considered as anomalous. The consecutive samples below this threshold are ignored (default is 1)
        Returns
        -------
        pd.DataFrame
            Matching DataFrame
        """
        data = pd.DataFrame(data={'y_true': y_true,
                                'y_score': list(y_score)})
        data['y_pred'] = self._get_y_pred(y_score, threshold, inverse_threshold, min_consecutive_samples)

        return data.copy()


    def _filter_min_consecutive_samples(self, v, min_consecutive_samples):
        """
        Filters the predictions based on the minimum consecutive samples threshold

        Parameters
        ----------
        v: array-like
            Predictions in boolean format
        min_consecutive_samples: int
            Minimum consecutive samples to be considered as anomalous. The consecutive samples below this threshold are ignored (default is 1)
        Returns
        -------
        np.array
            Predictions in boolean format filtered
        """
        convert_to_zero_index = []
        for i in np.where(v == 1)[0]:
            skip = False
            for k in range(1, min_consecutive_samples+1):
                if np.sum(v[i-(min_consecutive_samples-1):i+k]) == min_consecutive_samples:
                    skip = True
                    break
            if skip:
                continue
            convert_to_zero_index.append(i)
        v[convert_to_zero_index] = 0
        return v.copy()