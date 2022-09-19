import os
import numpy as np
import pandas as pd
from odin.annotator.meta_annotator_extractor import PropertyAnnotatorInterface


class AnnotatorMonth(PropertyAnnotatorInterface):
    """
    Subclass of PropertyAnnotatorInterface.
    Implement the color annotation for each input image.
    """

    NAME = "month"
    DEFAULT_VALUE = 0

    def __init__(self):
        property_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        super().__init__(self.NAME, property_values, self.DEFAULT_VALUE)

    def process_object(self, data_object): 
        """
        Check the month in which the point is
        """
        try:
            return data_object._name.month
        except (OSError, KeyError, TypeError, IOError):
            return self.DEFAULT_VALUE

        
        
class AnnotatorDuration(PropertyAnnotatorInterface):
    """
    Subclass of PropertyAnnotatorInterface.
    Implement the color annotation for each input image.
    """

    NAME = "duration"
    DEFAULT_VALUE = 0

    def __init__(self):
        property_values = np.arange(1, 10000) # aribtrary value
        super().__init__(self.NAME, property_values, self.DEFAULT_VALUE)

    def process_object(self, data_object): # here I pass the entire dataset, not only a row
        """
        Check the duration of the time series
        """
        try:
            series = pd.Series()
            
            series = data_object["anomaly"].groupby((data_object["anomaly"] != data_object["anomaly"].shift()).cumsum()).transform('size') * data_object["anomaly"]
            series = series.loc[series.shift(-1) != series]
            series = series.fillna(0)
            return series
        
        except (OSError, KeyError, TypeError, IOError):
            return self.DEFAULT_VALUE
