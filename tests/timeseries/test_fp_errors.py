import os
import unittest
import pandas as pd
from pandas import Timestamp

from odin.classes.timeseries import DatasetTSAnomalyDetection, TimeSeriesType, TSProposalsType, AnalyzerTSAnomalyDetection
from odin.annotator.ts_meta_annotator_extractor import MetaPropertiesExtractor, MetaProperties
from odin.classes import Errors, CustomError, ErrCombination
from odin.classes import Metrics, Curves


class FPErrorsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        ds_path = os.path.join(dir_path, 'data/fp_errors/test_gt.csv')
        anomalies_path = os.path.join(dir_path, 'data/fp_errors/anomalies.json')
        props_path = [("TEST", os.path.join(dir_path, 'data/fp_errors/predictions.csv'), TSProposalsType.REGRESSION)]

        index_gt = "timestamp"
        index_props = "timestamp"
        cls.dataset = DatasetTSAnomalyDetection(ds_path,
                                                        TimeSeriesType.UNIVARIATE,
                                                        anomalies_path=anomalies_path,
                                                        proposals_paths=props_path,
                                                        index_gt=index_gt,
                                                        index_proposals=index_props)
        
        
       
    def setUp(self):
        self.analyzer = AnalyzerTSAnomalyDetection("TEST",
                                                    self.dataset,
                                                    scaler_values=(False, False))
        
        
    def test_fp_errors(self):
        anticipation = ErrCombination("ANTICIPATION", [Errors.BEFORE, Errors.CLOSELY_BEFORE, Errors.CLOSELY_AFTER, Errors.AFTER], ["Before", "Closely before", "Closely after", "After"])
        
        categories_dict, distances, errors_index, matching = self.analyzer.analyze_false_positive_errors(metric = Metrics.ACCURACY, error_combination = anticipation, parameters_dicts = [{'closely_threshold': 2}], show = False)
        
        assert categories_dict["Before"] == 1, "Error in computing the 'before' FP errors"
        assert categories_dict["Closely before"] == 1, "Error in computing the 'closely before' FP errors"
        assert categories_dict["Closely after"] == 1,  "Error in computing the 'closely after' FP errors"
        assert categories_dict["After"] == 1, "Error in computing the 'after' FP errors"
        
        assert distances == {'Before': [-5], 'Closely before': [-1], 'Closely after': [1], 'After': [5]}, "Distances were computed incorrectly"
        
        assert errors_index == {'Before': [Timestamp('2020-01-21 00:21:00')], 'Closely before': [Timestamp('2020-01-21 00:25:00')], 'Closely after': [Timestamp('2020-01-21 00:13:00')], 'After': [Timestamp('2020-01-21 00:17:00')]}, "Errors located at unexpected positions"
        
        assert matching._get_value(Timestamp('2020-01-21 00:06:00'), 'eval') == 1, "Incorrect FN detection"
        assert matching._get_value(Timestamp('2020-01-21 00:08:00'), 'eval') == 1, "Incorrect FN detection"
        assert matching._get_value(Timestamp('2020-01-21 00:13:00'), 'eval') == -1, "Incorrect FP detection"
        assert matching._get_value(Timestamp('2020-01-21 00:17:00'), 'eval') == -1, "Incorrect FP detection"
        assert matching._get_value(Timestamp('2020-01-21 00:21:00'), 'eval') == -1, "Incorrect FP detection"
        assert matching._get_value(Timestamp('2020-01-21 00:25:00'), 'eval') == -1, "Incorrect FP detection"
        assert matching._get_value(Timestamp('2020-01-21 00:26:00'), 'eval') == 1, "Incorrect TP detection"



            
if __name__ == '__main__':
    unittest.main()
