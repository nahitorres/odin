import json
import os

import numpy as np
import unittest

from odin.classes import TaskType, DatasetClassification, AnalyzerClassification


class ClassificationInputCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        properties_path = os.path.join(dir_path, "properties.json")

        ds_path = os.path.join(dir_path, "multilabel/data/gt.json")
        task = TaskType.CLASSIFICATION_MULTI_LABEL
        props_path = [("Test", os.path.join(dir_path, "multilabel/data/predictions"))]
        cls.my_dataset = DatasetClassification(ds_path, task, props_path, save_graphs_as_png=False,
                                               properties_file=properties_path)
        cls.categories = cls.my_dataset.get_categories_names()
        file = open(os.path.join(dir_path, "multilabel/data/expected_results_ML.json"), "r")
        cls.expected_results = json.load(file)
        file.close()

    def setUp(self):
        self.my_analyzer = AnalyzerClassification('Test', self.my_dataset, save_graphs_as_png=False)

    # -- THRESHOLD TEST -- #

    def test_set_threshold(self):
        self.my_analyzer.set_confidence_threshold(0.2)
        self.assertEqual(0.2, self.my_analyzer._conf_thresh)

    # -- PRIVATE FUNCTIONS TEST -- #

    def test_support_metric(self):
        gt = self.expected_results["convert_for_category"]["dog"]["y_true"]
        detections = self.expected_results["convert_for_category"]["dog"]["y_score"]

        expected_gt_ord = np.array(self.expected_results["support_metric"]["dog"]["gt_ord"])
        expected_det_ord = np.array(self.expected_results["support_metric"]["dog"]["det_ord"])
        expected_tp = np.array(self.expected_results["support_metric"]["dog"]["tp"])
        expected_tn = self.expected_results["support_metric"]["dog"]["tn"]
        expected_fp = np.array(self.expected_results["support_metric"]["dog"]["fp"])

        gt_ord, det_ord, tp, tn, fp = self.my_analyzer._support_metric(gt, detections, None)

        self.assertTrue((expected_gt_ord == gt_ord).all())
        self.assertTrue((expected_det_ord == det_ord).all())
        self.assertTrue((expected_tp == tp).all())
        self.assertEqual(expected_tn, tn)
        self.assertTrue((expected_fp == fp).all())

    def test_support_metric_threshold(self):
        n_true_gt = self.expected_results["support_metric_threshold"]["dog"]["n_true"]
        n_normalized = 0.1
        gt_ord = np.array(self.expected_results["support_metric"]["dog"]["gt_ord"])
        det_ord = np.array(self.expected_results["support_metric"]["dog"]["det_ord"])
        tp = np.array(self.expected_results["support_metric"]["dog"]["tp"])
        fp = np.array(self.expected_results["support_metric"]["dog"]["fp"])
        threshold = 0.5

        expected_tp = self.expected_results["support_metric_threshold"]["dog"]["tp"]
        expected_tp_norm = expected_tp * n_normalized / n_true_gt
        expected_fp = self.expected_results["support_metric_threshold"]["dog"]["fp"]
        expected_tn = self.expected_results["support_metric_threshold"]["dog"]["tn"]

        tp, tp_norm, fp, tn = self.my_analyzer._support_metric_threshold(n_true_gt, n_normalized, gt_ord, det_ord, tp, fp, threshold)

        self.assertEqual(expected_tp, tp)
        self.assertEqual(expected_tp_norm, tp_norm)
        self.assertEqual(expected_fp, fp)
        self.assertEqual(expected_tn, tn)

    # -- INPUT PARSING TEST -- #

    def test_AnalyzerClassification(self):
        with self.assertRaises(TypeError):
            AnalyzerClassification(10, self.my_dataset)
        with self.assertRaises(TypeError):
            AnalyzerClassification("Test", 10)
        with self.assertRaises(TypeError):
            AnalyzerClassification("Test", self.my_dataset, result_saving_path=10)
        with self.assertRaises(TypeError):
            AnalyzerClassification("Test", self.my_dataset, norm_factor_categories="10")
        with self.assertRaises(TypeError):
            AnalyzerClassification("Test", self.my_dataset, norm_factors_properties=10)
        with self.assertRaises(TypeError):
            AnalyzerClassification("Test", self.my_dataset, norm_factors_properties=["10"])
        with self.assertRaises(TypeError):
            AnalyzerClassification("Test", self.my_dataset, norm_factors_properties=[(10, 22, 22)])
        with self.assertRaises(TypeError):
            AnalyzerClassification("Test", self.my_dataset, conf_thresh="10")
        with self.assertRaises(TypeError):
            AnalyzerClassification("Test", self.my_dataset, metric="10")
        with self.assertRaises(TypeError):
            AnalyzerClassification("Test", self.my_dataset, save_graphs_as_png=10)

    def test_add_custom_metric_input(self):
        error = self.my_analyzer.add_custom_metric(10)
        self.assertEqual(-1, error)

    def test_analyze_property_input(self):
        error = self.my_analyzer.analyze_property(10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_property("10")
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_property("size", possible_values=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_property("size", possible_values="10")
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_property("size", possible_values=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_property("size", show=2)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_property("size", show="2")
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_property("size", metric=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_property("size", metric="10")
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_property("size", split_by=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_property("size", split_by="10")
        self.assertEqual(-1, error)

    def test_analyze_properties_input(self):
        error = self.my_analyzer.analyze_properties(properties=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_properties(properties=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_properties(metric=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_properties(metric="10")
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_properties(split_by=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_properties(split_by="10")
        self.assertEqual(-1, error)

    def test_analyze_sensitivity_impact_of_properties_input(self):
        error = self.my_analyzer.analyze_sensitivity_impact_of_properties(properties=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_sensitivity_impact_of_properties(properties=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_sensitivity_impact_of_properties(metric=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_sensitivity_impact_of_properties(metric="10")
        self.assertEqual(-1, error)

    def test_analyze_false_positive_errors_input(self):
        error = self.my_analyzer.analyze_false_positive_errors(categories=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_false_positive_errors(categories=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_false_positive_errors(metric=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_false_positive_errors(metric="10")
        self.assertEqual(-1, error)

    def test_analyze_curve_for_categories_input(self):
        error = self.my_analyzer.analyze_curve_for_categories(categories=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_curve_for_categories(categories=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_curve_for_categories(curve=10)
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_curve_for_categories(show=10)
        self.assertEqual(-1, error)

    def test_true_positive_distribution_input(self):
        error = self.my_analyzer.show_true_positive_distribution(categories=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.show_true_positive_distribution(categories=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.show_true_positive_distribution(show=10)
        self.assertEqual(-1, error)

    def test_false_negative_distribution_input(self):
        error = self.my_analyzer.show_false_negative_distribution(categories=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.show_false_negative_distribution(categories=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.show_false_negative_distribution(show=10)
        self.assertEqual(-1, error)

    def test_set_normalization_input(self):
        error = self.my_analyzer.set_normalization(10)
        self.assertEqual(-1, error)

        error = self.my_analyzer.set_normalization(True, with_properties=10)
        self.assertEqual(-1, error)

        error = self.my_analyzer.set_normalization(True, norm_factor_categories="10")
        self.assertEqual(-1, error)

        error = self.my_analyzer.set_normalization(True, norm_factors_properties="10")
        self.assertEqual(-1, error)
        error = self.my_analyzer.set_normalization(True, norm_factors_properties=[("1", "2", "3")])
        self.assertEqual(-1, error)

    def test_set_confidence_threshold_input(self):
        error = self.my_analyzer.set_confidence_threshold(10)
        self.assertEqual(-1, error)

        error = self.my_analyzer.set_confidence_threshold([10])
        self.assertEqual(-1, error)

    def test_analyze_reliability_input(self):
        error = self.my_analyzer.analyze_reliability(num_bins="10")
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_reliability(num_bins=-1)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_reliability(num_bins=500)
        self.assertEqual(-1, error)

    def test_analyze_false_positive_error_for_category_input(self):
        error = self.my_analyzer.analyze_false_positive_errors_for_category(10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_false_positive_errors_for_category("10")
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_false_positive_errors_for_category("cat", metric=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.analyze_false_positive_errors_for_category("cat", metric="10")
        self.assertEqual(-1, error)

        error = self.my_analyzer.analyze_false_positive_errors_for_category("cat", show=10)
        self.assertEqual(-1, error)

    def test_analyze_confusion_matrix_input(self):
        error = self.my_analyzer.show_confusion_matrix(categories=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.show_confusion_matrix(categories=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.show_confusion_matrix(properties_names=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.show_confusion_matrix(properties_names=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.show_confusion_matrix(properties_values=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.show_confusion_matrix(properties_names=["size"], properties_values=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.show_confusion_matrix(properties_names=["size"], properties_values=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.show_confusion_matrix(properties_names=["size"], properties_values=[["10"]])
        self.assertEqual(-1, error)

        error = self.my_analyzer.show_confusion_matrix(show=10)
        self.assertEqual(-1, error)

    def test_base_report_input(self):
        error = self.my_analyzer.base_report(metrics=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.base_report(metrics=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.base_report(categories=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.base_report(categories=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.base_report(properties=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.base_report(properties=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.base_report(show_categories=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.base_report(show_properties=10)
        self.assertEqual(-1, error)

    def test_show_true_negative_distribution_input(self):
        error = self.my_analyzer.show_true_negative_distribution(categories=10)
        self.assertEqual(-1, error)
        error = self.my_analyzer.show_true_negative_distribution(categories=["10"])
        self.assertEqual(-1, error)

        error = self.my_analyzer.show_true_negative_distribution(show=10)
        self.assertEqual(-1, error)


if __name__ == '__main__':
    unittest.main()
