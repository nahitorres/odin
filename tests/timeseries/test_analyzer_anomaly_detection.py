import os
import unittest

from odin.classes.timeseries.anomaly_definition_strategies import \
    AnomalyDefinitionStrategyTSLabel
from odin.classes.timeseries.anomaly_matching_strategies import \
    AnomalyMatchingStrategyIntervalToInterval
from odin.classes.timeseries import AnalyzerTSAnomalyDetection, TSProposalsType, \
    DatasetTSAnomalyDetection, TimeSeriesType

class AnomalyDetectionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        ds_path = os.path.join(dir_path, 'data/perfect_case/test_gt.csv')
        props_path = [("TEST", os.path.join(dir_path, 'data/perfect_case/test_proposals.csv'), TSProposalsType.LABEL)]
        anomalies_path = os.path.join(dir_path, 'data/perfect_case/test_anomalies.json')
        properties_path = os.path.join(dir_path, 'data/perfect_case/test_properties.csv')
        properties_json_path = os.path.join(dir_path, 'data/perfect_case/test_properties.json')
        index_gt = "timestamp"
        index_props = "timestamp"
        cls.dataset_perfect = DatasetTSAnomalyDetection(ds_path,
                                                        TimeSeriesType.UNIVARIATE,
                                                        anomalies_path=anomalies_path,
                                                        proposals_paths=props_path,
                                                        properties_path=properties_path,
                                                        properties_json=properties_json_path,
                                                        index_gt=index_gt,
                                                        index_proposals=index_props)

        ds_path = os.path.join(dir_path, 'data/imperfect_case/test_gt.csv')
        props_path = [("TEST", os.path.join(dir_path, 'data/imperfect_case/test_proposals.csv'), TSProposalsType.LABEL)]
        anomalies_path = os.path.join(dir_path, 'data/imperfect_case/test_anomalies.json')
        properties_path = os.path.join(dir_path, 'data/imperfect_case/test_properties.csv')
        properties_json_path = os.path.join(dir_path, 'data/imperfect_case/test_properties.json')
        index_gt = "timestamp"
        index_props = "timestamp"
        cls.dataset_imperfect = DatasetTSAnomalyDetection(ds_path,
                                                          TimeSeriesType.UNIVARIATE,
                                                          anomalies_path=anomalies_path,
                                                          proposals_paths=props_path,
                                                          properties_path=properties_path,
                                                          properties_json=properties_json_path,
                                                          index_gt=index_gt,
                                                          index_proposals=index_props)
    
    def setUp(self):
        self.analyzer_perfect_case = AnalyzerTSAnomalyDetection("TEST",
                                                                self.dataset_perfect,
                                                                threshold=0.45,
                                                                anomaly_evaluation=AnomalyDefinitionStrategyTSLabel(),
                                                                matching_strategy=AnomalyMatchingStrategyIntervalToInterval(),
                                                                scaler_values=(False, False))
        self.analyzer_imperfect_case = AnalyzerTSAnomalyDetection("TEST",
                                                                  self.dataset_imperfect,
                                                                  threshold=0.45,
                                                                  anomaly_evaluation=AnomalyDefinitionStrategyTSLabel(),
                                                                  matching_strategy=AnomalyMatchingStrategyIntervalToInterval(),
                                                                  scaler_values=(False, False))
        
    # TEST ANOMALIES DIFFERENCE DISTRIBUTION AND HISTOGRAM
    
    def test_anomaly_difference_distributions(self):
        # test that invalid parameters stop the execution
        results = self.analyzer_perfect_case.analyze_true_predicted_distributions(groups="MUST BE INT", show=False)
        self.assertEqual(results, -1)
        results = self.analyzer_perfect_case.analyze_true_predicted_distributions(groups=3.45, show=False)
        self.assertEqual(results, -1)
        results = self.analyzer_perfect_case.analyze_true_predicted_distributions(groups=0, show=False)
        self.assertEqual(results, -1)
        results = self.analyzer_perfect_case.analyze_true_predicted_distributions(groups=-1, show=False)
        self.assertEqual(results, -1)
        results = self.analyzer_perfect_case.analyze_true_predicted_distributions(groups=5, threshold="MUST BE FLOAT OR NONE", show=False)
        self.assertEqual(results, -1)
        
        # test if the results are correctly computed
        results = self.analyzer_perfect_case.analyze_true_predicted_distributions(groups=3, show=False)
        self.assertIn("gt", results.keys())
        self.assertIn("predictions", results.keys())
        self.assertIn("groups", results.keys())
        self.assertIsInstance(results["groups"], list)
        self.assertIsInstance(results["predictions"], list)
        self.assertIsInstance(results["gt"], list)
        self.assertTrue(len(results["gt"]) == len(results["predictions"]) and len(results["predictions"]) == len(results["groups"]) and len(results["groups"]) == 3)
        self.assertTrue(results["predictions"][2] == 7)
        self.assertTrue(results["gt"][2] == 3)
        other_preds = sum(results["predictions"]) - results["predictions"][2]
        other_gt = sum(results["gt"]) - results["gt"][2]
        self.assertTrue(other_gt == other_preds and other_preds == 0)
        
    def test_anomaly_difference_histogram(self):
        # test that invalid parameters stop the execution
        results = self.analyzer_perfect_case.analyze_true_predicted_difference_distribution(nbins="MUST BE INT", show=False)
        self.assertEqual(results, -1)
        results = self.analyzer_perfect_case.analyze_true_predicted_difference_distribution(nbins=5.24, show=False)
        self.assertEqual(results, -1)
        results = self.analyzer_perfect_case.analyze_true_predicted_difference_distribution(nbins=-1, show=False)
        self.assertEqual(results, -1)
        
        # tests if the results are correctly computed
        results = self.analyzer_perfect_case.analyze_true_predicted_difference_distribution(nbins=20, iou_threshold=0.7, show=False)
        self.assertIn("tp_differences", results.keys())
        self.assertTrue(len(results["tp_differences"]) == 3)
        
        self.analyzer_perfect_case._threshold = 0.6
        results = self.analyzer_perfect_case.analyze_true_predicted_difference_distribution(nbins=20, iou_threshold=0.7, show=False)
        self.assertTrue(results == -1)
        
        self.analyzer_perfect_case._threshold = 0.45
        results = self.analyzer_imperfect_case.analyze_true_predicted_difference_distribution(nbins=20, iou_threshold=0.4, show=False)
        self.assertIn("tp_differences", results.keys())
        self.assertTrue(len(results["tp_differences"]) == 3)
        self.assertTrue(sum(results["tp_differences"]) == -4)
        
        results = self.analyzer_imperfect_case.analyze_true_predicted_difference_distribution(nbins=20, iou_threshold=0.7, show=False)
        self.assertIn("tp_differences", results.keys())
        self.assertTrue(len(results["tp_differences"]) == 2)
        self.assertTrue(sum(results["tp_differences"]) == 0)
    
    def test_anomaly_iou_distribution(self):
        # test that invalid parameters stop the execution
        results = self.analyzer_perfect_case.analyze_iou_distribution(nbins="MUST BE INT", show=False)
        self.assertEqual(results, -1)
        results = self.analyzer_perfect_case.analyze_iou_distribution(nbins=5.24, show=False)
        self.assertEqual(results, -1)
        results = self.analyzer_perfect_case.analyze_iou_distribution(nbins=-1, show=False)
        self.assertEqual(results, -1)
        results = self.analyzer_perfect_case.analyze_iou_distribution(nbins=20, threshold="MUST BE FLOAT", show=False)
        self.assertEqual(results, -1)
        
        # tests if the results are correctly computed
        results = self.analyzer_perfect_case.analyze_iou_distribution(nbins=10, show=False)
        self.assertTrue(len([el for el in results["iou_values"] if el == 1]) == 3)
        self.assertTrue(len([el for el in results["iou_values"] if el == 0]) == 4)
        
        results = self.analyzer_imperfect_case.analyze_iou_distribution(nbins=10, show=False)
        self.assertTrue(len([el for el in results["iou_values"] if el == 1]) == 2)
        self.assertTrue(len([el for el in results["iou_values"] if el == 0]) == 3)
        self.assertTrue(len([el for el in results["iou_values"] if el == 3/7]) == 1)
        
        
if __name__ == '__main__':
    unittest.main()

        