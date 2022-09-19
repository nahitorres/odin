import os
import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal

from odin.classes.timeseries import DatasetTSAnomalyDetection, TimeSeriesType, TSProposalsType, AnalyzerTSAnomalyDetection
from odin.annotator.ts_meta_annotator_extractor import MetaPropertiesExtractor, MetaProperties


class PredefinedMetapropertiesExtractorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        ds_path = os.path.join(dir_path, 'data/perfect_case/test_gt_month.csv')
        anomalies_path = os.path.join(dir_path, 'data/perfect_case/test_anomalies_month.json')
        cls.perfect_properties_path = os.path.join(dir_path, 'data/perfect_case/test_month.csv')
        props_path = [("TEST", os.path.join(dir_path, 'data/perfect_case/test_proposals_month.csv'), TSProposalsType.LABEL)]

        index_gt = "timestamp"
        index_props = "timestamp"
        cls.dataset_perfect = DatasetTSAnomalyDetection(ds_path,
                                                        TimeSeriesType.UNIVARIATE,
                                                        anomalies_path=anomalies_path,
                                                        proposals_paths=props_path,
                                                        properties_path=cls.perfect_properties_path,
                                                        index_gt=index_gt,
                                                        index_proposals=index_props)
        
        
        ds_path = os.path.join(dir_path, 'data/imperfect_case/test_gt_month.csv')
        anomalies_path = os.path.join(dir_path, 'data/imperfect_case/test_anomalies_month.json')
        cls.imperfect_properties_path = os.path.join(dir_path, 'data/imperfect_case/test_month.csv')
        props_path = [("TEST", os.path.join(dir_path, 'data/imperfect_case/test_proposals_month.csv'), TSProposalsType.LABEL)]

        index_gt = "timestamp"
        index_props = "timestamp"
        cls.dataset_imperfect = DatasetTSAnomalyDetection(ds_path,
                                                        TimeSeriesType.UNIVARIATE,
                                                        anomalies_path=anomalies_path,
                                                        proposals_paths=props_path,
                                                        properties_path=cls.imperfect_properties_path,
                                                        index_gt=index_gt,
                                                        index_proposals=index_props)
        
        
    def setUp(self):
        self.analyzer_perfect_case = AnalyzerTSAnomalyDetection("TEST",
                                                                self.dataset_perfect,
                                                                scaler_values=(False, False))
        self.analyzer_imperfect_case = AnalyzerTSAnomalyDetection("TEST",
                                                                  self.dataset_imperfect,
                                                                  scaler_values=(False, False))
        
    def test_correctness_months(self):
        output_path_csv = "data/generated_month.csv"
        
        # Test with perfect data
        my_annotator = MetaPropertiesExtractor(self.dataset_perfect, [MetaProperties.MONTH], output_path=output_path_csv)
        my_annotator.start_annotation(single_row = True)
        
        # Load generated data (the ones to be tested for correctness)
        generated_data = pd.read_csv(output_path_csv, index_col="timestamp").sort_index().sort_index(axis = 1)
        generated_data.index = pd.to_datetime(generated_data.index)
        
        # Recover the GT data (here, the correct ones)
        gt_data = pd.read_csv(self.perfect_properties_path, index_col="timestamp").sort_index().sort_index(axis = 1)
        gt_data.index = pd.to_datetime(gt_data.index)
        
        assert_frame_equal(generated_data, gt_data)
        
        
        
        # Test with imperfect data
        my_annotator = MetaPropertiesExtractor(self.dataset_imperfect, [MetaProperties.MONTH], output_path=output_path_csv)
        my_annotator.start_annotation(single_row = True)
                
        # Load generated data (the ones to be tested for correctness)
        generated_data = pd.read_csv(output_path_csv, index_col="timestamp").sort_index().sort_index(axis = 1)
        generated_data.index = pd.to_datetime(generated_data.index)
        
        # Recover the GT data (here, the wrong ones)
        gt_data = pd.read_csv(self.imperfect_properties_path, index_col="timestamp").sort_index().sort_index(axis = 1)
        gt_data.index = pd.to_datetime(gt_data.index)
        
        self.assertEqual(False, generated_data.equals(gt_data))        
            
if __name__ == '__main__':
    unittest.main()
