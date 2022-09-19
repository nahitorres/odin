import json
import os

import numpy as np
import unittest

from odin.classes import DatasetClassification, AnalyzerClassification, TaskType, Metrics, Curves


class ClassificationMLTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        task = TaskType.CLASSIFICATION_MULTI_LABEL

        dir_path = os.path.dirname(os.path.realpath(__file__))
        properties_path = os.path.join(dir_path, "properties.json")

        # GENERIC CASE
        ds_path = os.path.join(dir_path, "data/gt.json")
        props_path = [("Test", os.path.join(dir_path, "data/predictions"))]
        cls.dataset_generic = DatasetClassification(ds_path, task, props_path, save_graphs_as_png=False,
                                                    properties_file=properties_path)
        file = open(os.path.join(dir_path, "data/expected_results_ML.json"), "r")
        cls.expected_results_generic = json.load(file)
        file.close()

        # PERFECT CASE
        ds_path = os.path.join(dir_path, "data/perfect_case/gt.json")
        props_path = [("Test", os.path.join(dir_path, "data/perfect_case/predictions"))]
        cls.dataset_perfect = DatasetClassification(ds_path, task, props_path, save_graphs_as_png=False,
                                                    properties_file=properties_path)
        file = open(os.path.join(dir_path, "data/perfect_case/expected_results_ML.json"), "r")
        cls.expected_results_perfect = json.load(file)
        file.close()

        # WORST CASE
        ds_path = os.path.join(dir_path, "data/worst_case/gt.json")
        props_path = [("Test", os.path.join(dir_path, "data/worst_case/predictions"))]
        cls.dataset_worst = DatasetClassification(ds_path, task, props_path, save_graphs_as_png=False,
                                                  properties_file=properties_path)
        file = open(os.path.join(dir_path, "data/worst_case/expected_results_ML.json"), "r")
        cls.expected_results_worst = json.load(file)
        file.close()

        cls.categories = cls.dataset_generic.get_categories_names()

    def setUp(self):
        self.analyzer_generic = AnalyzerClassification('Test', self.dataset_generic, save_graphs_as_png=False)
        self.analyzer_perfect = AnalyzerClassification('Test', self.dataset_perfect, save_graphs_as_png=False)
        self.analyzer_worst = AnalyzerClassification('Test', self.dataset_worst, save_graphs_as_png=False)

    # -- SUPPORT FUNCTIONS BASIC METRICS -- #

    def _support_test_accuracy(self, y_true, y_pred, expected_result):
        res, _ = self.analyzer_generic._compute_metric_accuracy(y_true, y_pred)
        self.assertEqual(expected_result, res)

    def _support_test_error_rate(self, y_true, y_pred, expected_result):
        res, _ = self.analyzer_generic._compute_metric_error_rate(y_true, y_pred)
        self.assertEqual(expected_result, res)

    def _support_test_precision(self, y_true, y_pred, expected_result):
        res, _ = self.analyzer_generic._compute_metric_precision_score(y_true, y_pred, None)
        self.assertEqual(expected_result, res)

    def _support_test_recall(self, y_true, y_pred, expected_result):
        res, _ = self.analyzer_generic._compute_metric_recall_score(y_true, y_pred, None)
        self.assertEqual(expected_result, res)

    def _support_test_f1(self, y_true, y_pred, expected_result):
        res, _ = self.analyzer_generic._compute_metric_f1_score(y_true, y_pred, None)
        self.assertEqual(expected_result, res)

    # -- BASIC METRICS TEST -- #

    def test_accuracy_evaluation_metric(self):
        self._support_test_accuracy(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                    y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                                    expected_result=1)

        self._support_test_accuracy(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                    y_pred=[[0.4, 1, 0.7], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                    expected_result=3/4)

        self._support_test_accuracy(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                    y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    expected_result=0)

        self._support_test_accuracy(y_true=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    expected_result=1)

        self._support_test_accuracy(y_true=[],
                                    y_pred=[],
                                    expected_result=0)

    def test_error_rate_evaluation_metric(self):
        self._support_test_error_rate(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                      y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                                      expected_result=0)

        self._support_test_error_rate(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                      y_pred=[[0.4, 1, 0.7], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                      expected_result=1/4)

        self._support_test_error_rate(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                      y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                      expected_result=1)

        self._support_test_error_rate(y_true=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                      y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                      expected_result=0)

        self._support_test_error_rate(y_true=[],
                                      y_pred=[],
                                      expected_result=0)

    def test_precision_evaluation_metric(self):
        self._support_test_precision(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                     y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                                     expected_result=1)

        self._support_test_precision(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                     y_pred=[[0.4, 1, 0.7], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                     expected_result=10/11)

        self._support_test_precision(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                     y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                     expected_result=0)

        self._support_test_precision(y_true=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                     y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                     expected_result=0)

        self._support_test_precision(y_true=[],
                                     y_pred=[],
                                     expected_result=0)

    def test_recall_evaluation_metric(self):
        self._support_test_recall(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                  y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                                  expected_result=1)

        self._support_test_recall(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                  y_pred=[[0.4, 1, 0.7], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                  expected_result=10/11)

        self._support_test_recall(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                  y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                  expected_result=0)

        self._support_test_recall(y_true=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                  y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                  expected_result=0)

        self._support_test_recall(y_true=[],
                                  y_pred=[],
                                  expected_result=0)

    def test_f1_evaluation_metric(self):
        self._support_test_f1(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                              expected_result=1)

        self._support_test_f1(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              y_pred=[[0.4, 1, 0.7], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              expected_result=10/11)

        self._support_test_f1(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              expected_result=0)

        self._support_test_f1(y_true=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              y_pred=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              expected_result=0)

        self._support_test_f1(y_true=[],
                              y_pred=[],
                              expected_result=0)

    # -- BASIC METRICS MULTIPLE THRESHOLDS TEST -- #

    def test_accuracy_evaluation_metric_different_threshold(self):
        self.analyzer_generic.set_confidence_threshold(0.9)
        self._support_test_accuracy(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                    y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                                    expected_result=0)

        self.analyzer_generic.set_confidence_threshold(0.3)
        self._support_test_accuracy(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                    y_pred=[[0.4, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                    expected_result=1)

    def test_error_rate_evaluation_metric_different_threshold(self):
        self.analyzer_generic.set_confidence_threshold(0.9)
        self._support_test_error_rate(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                      y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                                      expected_result=1)

        self.analyzer_generic.set_confidence_threshold(0.3)
        self._support_test_error_rate(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                      y_pred=[[0.4, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                      expected_result=0)

    def test_precision_evaluation_metric_different_threshold(self):
        self.analyzer_generic.set_confidence_threshold(0.9)
        self._support_test_precision(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                     y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                                     expected_result=1)

        self.analyzer_generic.set_confidence_threshold(0.3)
        self._support_test_precision(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                     y_pred=[[0.4, 1, 0.3], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                     expected_result=11/12)

    def test_recall_evaluation_metric_different_threshold(self):
        self.analyzer_generic.set_confidence_threshold(0.9)
        self._support_test_recall(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                  y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                                  expected_result=4/12)

        self.analyzer_generic.set_confidence_threshold(0.3)
        self._support_test_recall(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                  y_pred=[[0.4, 1, 0.3], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                  expected_result=1)

    def test_f1_evaluation_metric_different_threshold(self):
        self.analyzer_generic.set_confidence_threshold(0.9)
        self._support_test_f1(y_true=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              y_pred=[[0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1], [0.6, 0.7, 1]],
                              expected_result=(2 * (1/3) / (1 + 1/3)))

        self.analyzer_generic.set_confidence_threshold(0.3)
        self._support_test_f1(y_true=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              y_pred=[[0.4, 1, 0.3], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              expected_result=(2 * (11/12) / (1 + 11/12)))

    # -- THRESHOLD TEST -- #

    def test_default_threshold(self):
        self.assertEqual(0.5, self.analyzer_generic._conf_thresh)

    # -- PUBLIC FUNCTIONS TEST -- #

    def _support_test_analyze_property(self, analyzer, expected_results):
        for category in self.categories:
            res = analyzer.analyze_property("size", ["small", "medium", "large"], show=False, metric=Metrics.F1_SCORE)
            self.assertEqual(expected_results["analyze_properties"][category][Metrics.F1_SCORE.value],
                             res[category]['all']['value'])
            for m_a_value in ["small", "medium", "large"]:
                self.assertEqual(
                    expected_results["analyze_properties"][category]["meta_annotations"]["size"][m_a_value][
                        Metrics.F1_SCORE.value],
                    res[category]["size"][m_a_value]["value"])

    def test_analyze_property(self):
        self._support_test_analyze_property(self.analyzer_generic, self.expected_results_generic)
        self._support_test_analyze_property(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_analyze_property(self.analyzer_worst, self.expected_results_worst)

    def _support_test_true_positive_distribution(self, analyzer, expected_results):
        res = analyzer.show_true_positive_distribution(show=False)

        for category in self.categories:
            self.assertEqual(expected_results["tp_distribution"][category], res[category])

    def test_true_positive_distribution(self):
        self._support_test_true_positive_distribution(self.analyzer_generic, self.expected_results_generic)
        self._support_test_true_positive_distribution(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_true_positive_distribution(self.analyzer_worst, self.expected_results_worst)

    def _support_test_true_negative_distribution(self, analyzer, expected_results):
        res = analyzer.show_true_negative_distribution(show=False)

        for category in self.categories:
            self.assertEqual(expected_results["tn_distribution"][category], res[category])

    def test_true_negative_distribution(self):
        self._support_test_true_negative_distribution(self.analyzer_generic, self.expected_results_generic)
        self._support_test_true_negative_distribution(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_true_negative_distribution(self.analyzer_worst, self.expected_results_worst)

    def _support_test_false_positive_distribution(self, analyzer, expected_results):
        res = analyzer.show_false_positive_distribution(show=False)

        for category in self.categories:
            self.assertEqual(expected_results["fp_distribution"][category], res[category])

    def test_false_positive_distribution(self):
        self._support_test_false_positive_distribution(self.analyzer_generic, self.expected_results_generic)
        self._support_test_false_positive_distribution(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_false_positive_distribution(self.analyzer_worst, self.expected_results_worst)

    def _support_test_false_negative_distribution(self, analyzer, expected_results):
        res = analyzer.show_false_negative_distribution(show=False)

        for category in self.categories:
            self.assertEqual(expected_results["fn_distribution"][category], res[category])

    def test_false_negative_distribution(self):
        self._support_test_false_negative_distribution(self.analyzer_generic, self.expected_results_generic)
        self._support_test_false_negative_distribution(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_false_negative_distribution(self.analyzer_worst, self.expected_results_worst)

    def _support_test_precision_recall_curve(self, analyzer, expected_results):
        res = analyzer.analyze_curve_for_categories(curve=Curves.PRECISION_RECALL_CURVE, show=False)
        for category in self.categories:
            expected_recall = np.array(expected_results["pr_curve"][category]["Recall"])
            expected_precision = np.array(expected_results["pr_curve"][category]["Precision"])
            self.assertTrue((expected_recall == res[category]["x"]).all())
            self.assertTrue((expected_precision == res[category]["y"]).all())

    def test_precision_recall_curve(self):
        self._support_test_precision_recall_curve(self.analyzer_generic, self.expected_results_generic)
        self._support_test_precision_recall_curve(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_precision_recall_curve(self.analyzer_worst, self.expected_results_worst)

    def _support_test_f1_curve(self, analyzer, expected_results):
        res = analyzer.analyze_curve_for_categories(curve=Curves.F1_CURVE, show=False)
        for category in self.categories:
            expected_threshold = np.array(expected_results["f1_curve"][category]["Threshold"])
            expected_f1 = np.array(expected_results["f1_curve"][category]["F1"])
            self.assertTrue((expected_threshold == res[category]["x"]).all())
            self.assertTrue((expected_f1 == res[category]["y"]).all())

    def test_f1_curve(self):
        self._support_test_f1_curve(self.analyzer_generic, self.expected_results_generic)
        self._support_test_f1_curve(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_f1_curve(self.analyzer_worst, self.expected_results_worst)

    def _support_test_roc_curve(self, analyzer, expected_results):
        res = analyzer.analyze_curve_for_categories(curve=Curves.ROC_CURVE, show=False)
        for category in self.categories:
            expected_fpr = np.flip(expected_results["roc_curve"][category]["fpr"])
            expected_tpr = np.flip(expected_results["roc_curve"][category]["tpr"])
            self.assertTrue((expected_fpr == res[category]["x"]).all())
            self.assertTrue((expected_tpr == res[category]["y"]).all())

    def test_roc_curve(self):
        self._support_test_roc_curve(self.analyzer_generic, self.expected_results_generic)
        self._support_test_roc_curve(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_roc_curve(self.analyzer_worst, self.expected_results_worst)

    def _support_test_confusion_matrix(self, analyzer, expected_results):
        for category in self.categories:
            cm = analyzer.show_confusion_matrix(categories=[category], show=False)
            expected_cm = np.array(expected_results["cm"][category])
            self.assertTrue((expected_cm == cm).all())

    def test_confusion_matrix(self):
        self._support_test_confusion_matrix(self.analyzer_generic, self.expected_results_generic)
        self._support_test_confusion_matrix(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_confusion_matrix(self.analyzer_worst, self.expected_results_worst)

    def _support_test_base_report(self, analyzer, expected_results):
        metrics = [Metrics.ACCURACY, Metrics.ERROR_RATE, Metrics.RECALL_SCORE, Metrics.PRECISION_SCORE, Metrics.F1_SCORE]

        report =analyzer.base_report(metrics=metrics)

        for metric in metrics:
            # check macro micro
            report_total = report.loc["Total"]
            self.assertEqual(expected_results["base_report"][metric.value]["macro"], report_total.loc["avg macro"][metric.value])
            self.assertEqual(expected_results["base_report"][metric.value]["micro"], report_total.loc["avg micro"][metric.value])

            # check categories
            report_categories = report.loc["Category"]
            for category in self.categories:
                self.assertEqual(expected_results["base_report"][metric.value][category], report_categories.loc[category][metric.value])

            # check properties
            report_properties = report.loc["Property"]
            for report_value in report_properties.index.values:
                self.assertEqual(expected_results["base_report"][metric.value][report_value],
                                 report_properties.loc[report_value][metric.value])

    def test_base_report(self):
        self._support_test_base_report(self.analyzer_generic, self.expected_results_generic)
        self._support_test_base_report(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_base_report(self.analyzer_worst, self.expected_results_worst)

    def _support_test_analyze_reliability(self, analyzer, expected_results):
        expected_values = np.array(expected_results["reliability"]["all"]["values"])
        expected_gaps = np.array(expected_results["reliability"]["all"]["gaps"])
        expected_counts = np.array(expected_results["reliability"]["all"]["counts"])
        expected_bins = np.array(expected_results["reliability"]["all"]["bins"])

        expected_avg_values = expected_results["reliability"]["all"]["avg_values"]
        expected_avg_conf = expected_results["reliability"]["all"]["avg_conf"]
        expected_ece = expected_results["reliability"]["all"]["ece"]
        expected_mce = expected_results["reliability"]["all"]["mce"]

        res = analyzer.analyze_reliability(show=False)

        self.assertTrue((expected_values.round(10) == res["values"].round(10)).all())
        self.assertTrue((expected_gaps.round(10) == res["gaps"].round(10)).all())
        self.assertTrue((expected_counts.round(10) == res["counts"].round(10)).all())
        self.assertTrue((expected_bins.round(10) == res["bins"].round(10)).all())
        self.assertEqual(np.around(expected_avg_values, 10), np.around(res["avg_value"], 10))
        self.assertEqual(np.around(expected_avg_conf, 10), np.around(res["avg_conf"], 10))
        self.assertEqual(np.around(expected_ece, 10), np.around(res["ece"], 10))
        self.assertEqual(np.around(expected_mce, 10), np.around(res["mce"], 10))

    def test_analyze_reliability(self):
        self._support_test_analyze_reliability(self.analyzer_generic, self.expected_results_generic)
        self._support_test_analyze_reliability(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_analyze_reliability(self.analyzer_worst, self.expected_results_worst)

    # -- PRIVATE FUNCTIONS TEST -- #

    def _support_test_calculate_metric_for_category(self, analyzer, expected_results):
        for cat in self.categories:
            res = analyzer._calculate_metric_for_category(cat, Metrics.F1_SCORE)

            self.assertEqual(expected_results["analyze_properties"][cat][Metrics.F1_SCORE.value], res["value"])
            self.assertIsNone(res["std"])
            self.assertIsNone(res["matching"])

    def test_calculate_metric_for_category(self):
        self._support_test_calculate_metric_for_category(self.analyzer_generic, self.expected_results_generic)
        self._support_test_calculate_metric_for_category(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_calculate_metric_for_category(self.analyzer_worst, self.expected_results_worst)

    def _support_test_calculate_metric_for_properties_of_category(self, analyzer, expected_results):
        for id, cat in enumerate(self.categories):
            res = analyzer._calculate_metric_for_properties_of_category(cat, id+1, "size", ["small", "medium", "large"], None, Metrics.F1_SCORE)

            for v in ["small", "medium", "large"]:
                self.assertEqual(expected_results["analyze_properties"][cat]["meta_annotations"]["size"][v][Metrics.F1_SCORE.value],
                                 res[v]["value"])
                self.assertIsNone(res[v]["std"])

    def test_calculate_metric_for_properties_of_category(self):
        self._support_test_calculate_metric_for_properties_of_category(self.analyzer_generic, self.expected_results_generic)
        self._support_test_calculate_metric_for_properties_of_category(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_calculate_metric_for_properties_of_category(self.analyzer_worst, self.expected_results_worst)

    def _support_test_convert_input_format_for_category(self, dataset, analyzer, expected_results):
        observations = dataset.get_all_observations()
        proposals = dataset.get_proposals("Test")

        obs_sorted = observations.sort_values(by="id")
        matching = analyzer._match_classification_with_ground_truth(obs_sorted, proposals)
        for cat in self.categories:
            y_true, y_scores = analyzer._AnalyzerClassification__convert_input_format_for_category(matching, cat)

            expected_y_true = np.array(expected_results["convert_for_category"][cat]["y_true"])
            expected_y_scores = np.array(expected_results["convert_for_category"][cat]["y_score"])

            self.assertTrue((expected_y_true == y_true).all())
            self.assertTrue((expected_y_scores == y_scores).all())

    def test_convert_input_format_for_category(self):
        self._support_test_convert_input_format_for_category(self.dataset_generic, self.analyzer_generic, self.expected_results_generic)
        self._support_test_convert_input_format_for_category(self.dataset_perfect, self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_convert_input_format_for_category(self.dataset_worst, self.analyzer_worst, self.expected_results_worst)

    def _support_test_convert_input_ml_sl(self, analyzer, expected_results):
        gt = expected_results["support_get_input_report"]["all"]["y_true"]
        detections = expected_results["support_get_input_report"]["all"]["y_score"]

        expected_y_true = np.array(expected_results["convert_input_ml_sl"]["y_true"])
        expected_y_true_all = np.array(expected_results["convert_input_ml_sl"]["y_true_all"])
        expected_y_pred = np.array(expected_results["convert_input_ml_sl"]["y_pred"])
        expected_y_score_all = np.array(expected_results["convert_input_ml_sl"]["y_score_all"])

        y_true, y_pred, y_true_all, y_score_all = analyzer._AnalyzerClassification__convert_input_ml_sl(gt, detections)

        self.assertTrue((expected_y_true == y_true).all())
        self.assertTrue((expected_y_pred == y_pred).all())
        self.assertTrue((expected_y_true_all == y_true_all).all())
        self.assertTrue((expected_y_score_all == y_score_all).all())

    def test_convert_input_ml_sl(self):
        self._support_test_convert_input_ml_sl(self.analyzer_generic, self.expected_results_generic)
        self._support_test_convert_input_ml_sl(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_convert_input_ml_sl(self.analyzer_worst, self.expected_results_worst)

    def _support_test_convert_input_confusion_matrix(self, dataset, analyzer, expected_results):
        gt = dataset.get_all_observations()
        proposals = dataset.get_proposals(analyzer._model_name)

        gt_ord = gt.sort_values(by="id")

        expected_y_true = np.array(expected_results["convert_input_ml_sl"]["y_true"])
        expected_y_pred = np.array(expected_results["convert_input_ml_sl"]["y_pred"])

        expected_labels = np.array([0, 1, 2])
        expected_cat_ids = np.array([1, 2, 3])

        matching = analyzer._match_classification_with_ground_truth(gt_ord, proposals)

        y_true, y_pred, labels, cat_ids = analyzer._AnalyzerClassification__convert_input_confusion_matrix(matching, self.categories, None)

        self.assertTrue((expected_y_true == y_true).all())
        self.assertTrue((expected_y_pred == y_pred).all())
        self.assertTrue((expected_labels == labels).all())
        self.assertTrue((expected_cat_ids == cat_ids).all())

    def test_convert_input_confusion_matrix(self):
        self._support_test_convert_input_confusion_matrix(self.dataset_generic, self.analyzer_generic, self.expected_results_generic)
        self._support_test_convert_input_confusion_matrix(self.dataset_perfect, self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_convert_input_confusion_matrix(self.dataset_worst, self.analyzer_worst, self.expected_results_worst)

    def _support_test_convert_input_reliability(self, dataset, analyzer, expected_results):
        gt = dataset.get_all_observations()
        proposals = dataset.get_proposals("Test")

        gt_ord = gt.sort_values(by="id")

        expected_y_true = np.array(expected_results["convert_input_reliability"]["all"]["y_true"])
        expected_y_pred = np.array(expected_results["convert_input_reliability"]["all"]["y_pred"])
        expected_y_score = np.array(expected_results["convert_input_reliability"]["all"]["y_score"])

        matching = analyzer._match_classification_with_ground_truth(gt_ord, proposals)
        y_true, y_pred, y_score = analyzer._AnalyzerClassification__convert_input_reliability(matching)

        self.assertTrue((expected_y_true == y_true).all())
        self.assertTrue((expected_y_pred == y_pred).all())
        self.assertTrue((expected_y_score == y_score).all())

    def test_convert_input_reliability(self):
        self._support_test_convert_input_reliability(self.dataset_generic, self.analyzer_generic, self.expected_results_generic)
        self._support_test_convert_input_reliability(self.dataset_perfect, self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_convert_input_reliability(self.dataset_worst, self.analyzer_worst, self.expected_results_worst)

    # -- NOT SUPPORTED FUNCTIONS TEST -- #

    def test_analyze_top1_top5_not_supported(self):
        error = self.analyzer_generic.analyze_top1_top5_error()
        self.assertEqual(-1, error)


if __name__ == '__main__':
    unittest.main()
