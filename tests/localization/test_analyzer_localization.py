import json
import os
import unittest
import pandas as pd
import numpy as np

from odin.classes import DatasetLocalization, TaskType, AnalyzerLocalization, Metrics, Curves


class ObjectDetectionTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        task = TaskType.OBJECT_DETECTION

        dir_path = os.path.dirname(os.path.realpath(__file__))
        properties_path = os.path.join(dir_path, "properties.json")

        ds_path = os.path.join(dir_path, 'data/gt.json')
        props_path = [("Test", os.path.join(dir_path, "data/predictions"))]
        cls.dataset_generic = DatasetLocalization(ds_path, task, props_path, save_graphs_as_png=False,
                                                  properties_file=properties_path)
        file = open(os.path.join(dir_path, "data/expected_results.json"), "r")
        cls.expected_results_generic = json.load(file)
        file.close()

        ds_path = os.path.join(dir_path, "data/perfect_case/gt.json")
        props_path = [("Test", os.path.join(dir_path, "data/perfect_case/predictions"))]
        cls.dataset_perfect = DatasetLocalization(ds_path, task, props_path, save_graphs_as_png=False,
                                                  properties_file=properties_path)
        file = open(os.path.join(dir_path, "data/perfect_case/expected_results.json"), "r")
        cls.expected_results_perfect = json.load(file)
        file.close()

        ds_path = os.path.join(dir_path, "data/worst_case/gt.json")
        props_path = [("Test", os.path.join(dir_path, "data/worst_case/predictions"))]
        cls.dataset_worst = DatasetLocalization(ds_path, task, props_path, save_graphs_as_png=False,
                                                properties_file=properties_path)
        file = open(os.path.join(dir_path, "data/worst_case/expected_results.json"), "r")
        cls.expected_results_worst = json.load(file)
        file.close()

        cls.categories = cls.dataset_generic.get_categories_names()

    def setUp(self):
        self.analyzer_generic = AnalyzerLocalization('Test', self.dataset_generic, use_normalization=False, save_graphs_as_png=False)
        self.analyzer_perfect = AnalyzerLocalization('Test', self.dataset_perfect, use_normalization=False, save_graphs_as_png=False)
        self.analyzer_worst = AnalyzerLocalization('Test', self.dataset_worst, use_normalization=False, save_graphs_as_png=False)

    # -- SUPPORT FUNCTIONS BASIC METRICS -- #

    def _support_test_precision(self, gt, matching, expected_result):
        res, _ = self.analyzer_generic._compute_metric_precision_score(gt, matching)
        self.assertEqual(expected_result, res)

    def _support_test_recall(self, gt, matching, expected_result):
        res, _ = self.analyzer_generic._compute_metric_recall_score(gt, matching)
        self.assertEqual(expected_result, res)

    def _support_test_f1(self, gt, matching, expected_result):
        res, _ = self.analyzer_generic._compute_metric_f1_score(gt, matching)
        self.assertEqual(expected_result, res)

    # -- BASIC METRICS TEST -- #

    def test_precision_evaluation_metric(self):
        gt = pd.DataFrame([1, 2, 3, 4, 5])  # need only length

        matching = pd.DataFrame([{"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1}])
        self._support_test_precision(gt, matching, 1)

        matching = pd.DataFrame([{"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1}])

        self._support_test_precision(gt, matching, 3/5)

        matching = pd.DataFrame([{"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0}])

        self._support_test_precision(gt, matching, 0)

    def test_recall_evaluation_metric(self):
        gt = pd.DataFrame([1, 2, 3, 4, 5])  # need only length

        matching = pd.DataFrame([{"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1}])
        self._support_test_recall(gt, matching, 1)

        matching = pd.DataFrame([{"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1}])

        self._support_test_recall(gt, matching, 3 / 5)

        matching = pd.DataFrame([{"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0}])

        self._support_test_recall(gt, matching, 0)

    def test_f1_evaluation_metric(self):
        gt = pd.DataFrame([1, 2, 3, 4, 5])  # need only length

        matching = pd.DataFrame([{"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1}])
        self._support_test_recall(gt, matching, 1)

        matching = pd.DataFrame([{"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1}])

        self._support_test_recall(gt, matching, 3 / 5)

        matching = pd.DataFrame([{"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0},
                                 {"confidence": 0.9, "label": 0}])

        self._support_test_recall(gt, matching, 0)

    # -- BASIC METRICS MULTIPLE THRESHOLDS TEST -- #

    def test_precision_evaluation_metric_multiple_threshold(self):
        gt = pd.DataFrame([1, 2, 3, 4, 5])  # need only length
        matching = pd.DataFrame([{"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1}])

        self.analyzer_generic.set_confidence_threshold(0.95)
        self._support_test_precision(gt, matching, 0)

        self.analyzer_generic.set_confidence_threshold(0.4)
        self._support_test_precision(gt, matching, 3/5)

    def test_recall_evaluation_metric_multiple_threshold(self):
        gt = pd.DataFrame([1, 2, 3, 4, 5])  # need only length
        matching = pd.DataFrame([{"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1}])
        self.analyzer_generic.set_confidence_threshold(0.95)
        self._support_test_recall(gt, matching, 0)

        self.analyzer_generic.set_confidence_threshold(0.4)
        self._support_test_recall(gt, matching, 3 / 5)

    def test_f1_evaluation_metric_multiple_threshold(self):
        gt = pd.DataFrame([1, 2, 3, 4, 5])  # need only length
        matching = pd.DataFrame([{"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": -1},
                                 {"confidence": 0.9, "label": 1},
                                 {"confidence": 0.9, "label": 1}])
        self.analyzer_generic.set_confidence_threshold(0.95)
        self._support_test_f1(gt, matching, 0)

        self.analyzer_generic.set_confidence_threshold(0.4)
        self._support_test_f1(gt, matching, 3 / 5)

    # -- THRESHOLD TEST -- #

    def test_default_threshold(self):
        self.assertEqual(0.5, self.analyzer_generic._conf_thresh)
        self.assertEqual(0.5, self.analyzer_generic._iou_thresh_strong)

    # -- PUBLIC FUNCTIONS TEST -- #

    def _support_test_analyze_intersection_over_union_for_category(self, analyzer, expected_results):
        for c in self.categories:
            res = analyzer.analyze_intersection_over_union_for_category(c, metric=Metrics.F1_SCORE, show=False)

            ious = np.array(res["iou"])
            values = np.array(res["metric_values"])

            expected_ious = np.array(expected_results["analyze_iou_for_category"][c]["iou"])
            expected_values = np.array(expected_results["analyze_iou_for_category"][c]["value"])

            self.assertTrue((expected_ious == ious).all())
            self.assertTrue((expected_values.round(10) == values.round(10)).all())

    def test_analyze_intersection_over_union_for_category(self):
        self._support_test_analyze_intersection_over_union_for_category(self.analyzer_generic, self.expected_results_generic)
        self._support_test_analyze_intersection_over_union_for_category(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_analyze_intersection_over_union_for_category(self.analyzer_worst, self.expected_results_worst)

    def _support_test_analyze_property(self, analyzer, expected_results):
        for c in self.categories:
            res = analyzer.analyze_property("size", ["small", "medium", "large"], show=False, metric=Metrics.F1_SCORE)
            self.assertEqual(expected_results["analyze_property"][c]["value"],
                             res[c]['all']['value'])
            for m_a_value in ["small", "medium", "large"]:
                self.assertEqual(
                    expected_results["analyze_property"][c]["size"][m_a_value],
                    res[c]["size"][m_a_value]["value"])

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

    def _support_test_analyze_reliability(self, analyzer, expected_results):
        expected_values = np.array(expected_results["analyze_reliability"]["values"])
        expected_gaps = np.array(expected_results["analyze_reliability"]["gaps"])
        expected_counts = np.array(expected_results["analyze_reliability"]["counts"])
        expected_bins = np.array(expected_results["analyze_reliability"]["bins"])

        expected_avg_values = expected_results["analyze_reliability"]["avg_value"]
        expected_avg_conf = expected_results["analyze_reliability"]["avg_conf"]
        expected_ece = expected_results["analyze_reliability"]["ece"]
        expected_mce = expected_results["analyze_reliability"]["mce"]

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

    def _support_test_base_report(self, analyzer, expected_results):
        metrics = [Metrics.RECALL_SCORE, Metrics.PRECISION_SCORE, Metrics.F1_SCORE]

        report = analyzer.base_report(metrics=metrics, properties=["size"])

        for metric in metrics:
            # check macro micro
            report_total = report.loc["Total"]
            self.assertEqual(expected_results["base_report"]["macro"][metric.value],
                             report_total.loc["avg macro"][metric.value])
            self.assertEqual(expected_results["base_report"]["micro"][metric.value],
                             report_total.loc["avg micro"][metric.value])

            # check categories
            report_categories = report.loc["Category"]
            for category in self.categories:
                self.assertEqual(expected_results["base_report"][category][metric.value],
                                 report_categories.loc[category][metric.value])

            # check properties
            report_properties = report.loc["Property"]
            for report_value in report_properties.index.values:
                self.assertEqual(expected_results["base_report"][report_value][metric.value],
                                 report_properties.loc[report_value][metric.value])

    def test_base_report(self):
        self._support_test_base_report(self.analyzer_generic, self.expected_results_generic)
        self._support_test_base_report(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_base_report(self.analyzer_worst, self.expected_results_worst)

    # -- PRIVATE FUNCTIONS TEST -- #

    def test_support_metric(self):
        gt = self.dataset_generic.get_anns_for_category(1)
        gt_ord = gt.sort_values(by="id")
        matching = pd.DataFrame(self.expected_results_generic["matching"]["cat"])
        matching_ord = matching.sort_values(by=["confidence", "det_id"], ascending=False)

        n_anns, _, numpy_confidence, numpy_label, tp, fp = self.analyzer_generic._support_metric(gt_ord, matching_ord)

        exp_confidence = np.array(self.expected_results_generic["support_metric"]["cat"]["confidence"])
        exp_label = np.array(self.expected_results_generic["support_metric"]["cat"]["label"])
        exp_tp = np.array(self.expected_results_generic["support_metric"]["cat"]["tp"])
        exp_fp = np.array(self.expected_results_generic["support_metric"]["cat"]["fp"])

        self.assertEqual(self.expected_results_generic["support_metric"]["cat"]["n_anns"], n_anns)
        self.assertTrue((exp_confidence == numpy_confidence).all())
        self.assertTrue((exp_label == numpy_label).all())
        self.assertTrue((exp_tp == tp).all())
        self.assertTrue((exp_fp == fp).all())

    def test_support_metric_threshold(self):

        n_true_gt = 24
        n_normalized = 30
        det_ord = np.array(self.expected_results_generic["support_metric"]["cat"]["confidence"])
        tp = np.array(self.expected_results_generic["support_metric"]["cat"]["tp"])
        fp = np.array(self.expected_results_generic["support_metric"]["cat"]["fp"])
        threshold = 0.5

        tp, tp_norm, fp = self.analyzer_generic._support_metric_threshold(n_true_gt, n_normalized, det_ord, tp, fp, threshold)

        self.assertEqual(self.expected_results_generic["support_metric_threshold"]["cat"]["tp"], tp)
        self.assertEqual(self.expected_results_generic["support_metric_threshold"]["cat"]["tp_norm"], tp_norm)
        self.assertEqual(self.expected_results_generic["support_metric_threshold"]["cat"]["fp"], fp)

    def _support_test_match_detection_with_ground_truth(self, dataset, analyzer, expected_results):
        gt = dataset.get_annotations()
        gt_ord = gt.sort_values(by="id")
        iou_thres = 0.5

        for c in self.categories:
            proposals = dataset.get_proposals_of_category(c, "Test")
            res = analyzer._match_detection_with_ground_truth(gt_ord, proposals, iou_thres)
            res.sort_values(by="det_id", inplace=True)

            for exp_match, (i, match) in zip(expected_results["matching"][c], res.iterrows()):
                self.assertEqual(exp_match["confidence"], match["confidence"])
                self.assertEqual(exp_match["difficult"], match["difficult"])
                self.assertEqual(exp_match["label"], match["label"])
                self.assertEqual(exp_match["iou"], match["iou"])
                self.assertEqual(exp_match["det_id"], match["det_id"])
                self.assertEqual(exp_match["ann_id"], match["ann_id"])
                self.assertEqual(exp_match["category_det"], match["category_det"])
                self.assertEqual(exp_match["category_ann"], match["category_ann"])

    def test_match_detection_with_ground_truth(self):
        self._support_test_match_detection_with_ground_truth(self.dataset_generic, self.analyzer_generic, self.expected_results_generic)
        self._support_test_match_detection_with_ground_truth(self.dataset_perfect, self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_match_detection_with_ground_truth(self.dataset_worst, self.analyzer_worst, self.expected_results_worst)

    def _support_test_calculate_metric_for_category(self, analyzer, expected_results):
        for c in self.categories:
            res = analyzer._calculate_metric_for_category(c, Metrics.F1_SCORE)
            self.assertEqual(expected_results["metric_for_category"][c]["F1"], res["value"])

    def test_calculate_metric_for_category(self):
        self._support_test_calculate_metric_for_category(self.analyzer_generic, self.expected_results_generic)
        self._support_test_calculate_metric_for_category(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_calculate_metric_for_category(self.analyzer_worst, self.expected_results_worst)

    def _support_test_calculate_metric_for_properties_of_category(self, analyzer, expected_results):
        for i, c in enumerate(self.categories):
            matching = pd.DataFrame(expected_results["matching"][c])
            matching.sort_values(by=["confidence", "det_id"], ascending=[False, True], inplace=True)
            res = analyzer._calculate_metric_for_properties_of_category(c, i+1, "size", ["small", "medium", "large"],
                                                         matching, Metrics.F1_SCORE)
            for p_v in ["small", "medium", "large"]:
                self.assertEqual(expected_results["metric_for_prop_for_cat"][c]["size"][p_v]["F1"], res[p_v]["value"])

    def test_calculate_metric_for_properties_of_category(self):
        self._support_test_calculate_metric_for_properties_of_category(self.analyzer_generic, self.expected_results_generic)
        self._support_test_calculate_metric_for_properties_of_category(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_calculate_metric_for_properties_of_category(self.analyzer_worst, self.expected_results_worst)

    def _support_test_support_reliability(self, analyzer, expected_results):
        matching = pd.concat([pd.DataFrame(expected_results["matching"]["cat"]),
                              pd.DataFrame(expected_results["matching"]["dog"]),
                              pd.DataFrame(expected_results["matching"]["fox"])])
        matching.sort_values(by=["confidence", "det_id"], ascending=[False, True], inplace=True)
        numpy_confidence, numpy_label = analyzer._AnalyzerLocalization__support_reliability(matching)

        exp_conf = np.array(expected_results["support_reliability"]["confidence"])
        exp_label = np.array(expected_results["support_reliability"]["label"])

        self.assertTrue((exp_conf == numpy_confidence).all())
        self.assertTrue((exp_label == exp_label).all())

    def test_support_reliability(self):
        self._support_test_support_reliability(self.analyzer_generic, self.expected_results_generic)
        self._support_test_support_reliability(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_support_reliability(self.analyzer_worst, self.expected_results_worst)

    def _support_test_get_input_report(self, analyzer, expected_results):
        results = analyzer._get_input_report(["size"], True)

        # check all
        exp_n_anns = expected_results["get_input_report"]["all"]["n_anns"]
        exp_matching = expected_results["get_input_report"]["all"]["matching"]

        self.assertEqual(exp_n_anns, len(results["total"]["all"]["anns"].index))
        res = results["total"]["all"]["matching"]
        res_ord = res.sort_values(by="det_id")

        for exp_match, (i, match) in zip(exp_matching, res_ord.iterrows()):
            self.assertEqual(exp_match["confidence"], match["confidence"])
            self.assertEqual(exp_match["difficult"], match["difficult"])
            self.assertEqual(exp_match["label"], match["label"])
            self.assertEqual(exp_match["iou"], match["iou"])
            self.assertEqual(exp_match["det_id"], match["det_id"])
            self.assertEqual(exp_match["ann_id"], match["ann_id"])
            self.assertEqual(exp_match["category_det"], match["category_det"])
            self.assertEqual(exp_match["category_ann"], match["category_ann"])

    def test_get_input_report(self):
        self._support_test_get_input_report(self.analyzer_generic, self.expected_results_generic)
        self._support_test_get_input_report(self.analyzer_perfect, self.expected_results_perfect)
        self._support_test_get_input_report(self.analyzer_worst, self.expected_results_worst)

    # TODO Fix test with new error categorization
    # def test_analyze_fp_errors(self):
    #     results = self.analyzer_generic.analyze_false_positive_errors_for_category("cat", show=False)

    #     for i, error in enumerate(results[1]):
    #         self.assertEqual(self.expected_results_generic["fp_categorization"][error], results[0][i][1])

    # -- INPUT PARSING TEST

    def test_AnalyzerLocalization(self):
        with self.assertRaises(TypeError):
            AnalyzerLocalization(10, self.dataset_generic)
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", 10)
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", self.dataset_generic, result_saving_path=10)
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", self.dataset_generic, norm_factor_categories="10")
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", self.dataset_generic, norm_factors_properties=10)
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", self.dataset_generic, norm_factors_properties=["10"])
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", self.dataset_generic, norm_factors_properties=[(10, 22, 22)])
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", self.dataset_generic, iou="10")
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", self.dataset_generic, conf_thresh="10")
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", self.dataset_generic, metric="10")
        with self.assertRaises(TypeError):
            AnalyzerLocalization("Test", self.dataset_generic, save_graphs_as_png=10)

    def test_add_custom_metric_input(self):
        error = self.analyzer_generic.add_custom_metric(10)
        self.assertEqual(-1, error)

    def test_analyze_property_input(self):
        error = self.analyzer_generic.analyze_property(10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_property("10")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_property("size", possible_values=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_property("size", possible_values="10")
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_property("size", possible_values=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_property("size", show=2)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_property("size", show="2")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_property("size", metric=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_property("size", metric="10")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_property("size", split_by=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_property("size", split_by="10")
        self.assertEqual(-1, error)

    def test_analyze_properties_input(self):
        error = self.analyzer_generic.analyze_properties(properties=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_properties(properties=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_properties(metric=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_properties(metric="10")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_properties(split_by=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_properties(split_by="10")
        self.assertEqual(-1, error)

    def test_analyze_sensitivity_impact_of_properties_input(self):
        error = self.analyzer_generic.analyze_sensitivity_impact_of_properties(properties=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_sensitivity_impact_of_properties(properties=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_sensitivity_impact_of_properties(metric=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_sensitivity_impact_of_properties(metric="10")
        self.assertEqual(-1, error)

    def test_analyze_false_positive_errors_input(self):
        error = self.analyzer_generic.analyze_false_positive_errors(categories=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_false_positive_errors(categories=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_false_positive_errors(metric=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_false_positive_errors(metric="10")
        self.assertEqual(-1, error)

    def test_analyze_curve_for_categories_input(self):
        error = self.analyzer_generic.analyze_curve_for_categories(categories=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_curve_for_categories(categories=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_curve_for_categories(curve=10)
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_curve_for_categories(show=10)
        self.assertEqual(-1, error)

    def test_true_positive_distribution_input(self):
        error = self.analyzer_generic.show_true_positive_distribution(categories=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.show_true_positive_distribution(categories=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.show_true_positive_distribution(show=10)
        self.assertEqual(-1, error)

    def test_false_negative_distribution_input(self):
        error = self.analyzer_generic.show_false_negative_distribution(categories=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.show_false_negative_distribution(categories=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.show_false_negative_distribution(show=10)
        self.assertEqual(-1, error)

    def test_set_normalization_input(self):
        error = self.analyzer_generic.set_normalization(10)
        self.assertEqual(-1, error)

        error = self.analyzer_generic.set_normalization(True, with_properties=10)
        self.assertEqual(-1, error)

        error = self.analyzer_generic.set_normalization(True, norm_factor_categories="10")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.set_normalization(True, norm_factors_properties="10")
        self.assertEqual(-1, error)
        error = self.analyzer_generic.set_normalization(True, norm_factors_properties=[("1", "2", "3")])
        self.assertEqual(-1, error)

    def test_set_confidence_threshold_input(self):
        error = self.analyzer_generic.set_confidence_threshold(10)
        self.assertEqual(-1, error)

        error = self.analyzer_generic.set_confidence_threshold([10])
        self.assertEqual(-1, error)

    def test_analyze_intersection_over_union_input(self):
        error = self.analyzer_generic.analyze_intersection_over_union(categories=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_intersection_over_union(categories=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_intersection_over_union(metric=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_intersection_over_union(metric="10")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_intersection_over_union(show=10)
        self.assertEqual(-1, error)

    def test_analyze_intersection_over_union_for_category_input(self):
        error = self.analyzer_generic.analyze_intersection_over_union_for_category(10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_intersection_over_union_for_category("10")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_intersection_over_union_for_category("cat", metric=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_intersection_over_union_for_category("cat", metric="10")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_intersection_over_union_for_category("cat", show=10)
        self.assertEqual(-1, error)

    def test_analyze_reliability_input(self):
        error = self.analyzer_generic.analyze_reliability(num_bins="10")
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_reliability(num_bins=-10)
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_reliability(show=10)
        self.assertEqual(-1, error)

    def test_analyze_false_positive_error_for_category_input(self):
        error = self.analyzer_generic.analyze_false_positive_errors_for_category(10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_false_positive_errors_for_category("aaa")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_false_positive_errors_for_category("cat", metric=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_false_positive_errors_for_category("cat", metric="10")
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_false_positive_errors_for_category("cat", show=10)
        self.assertEqual(-1, error)

    def test_base_report_input(self):
        error = self.analyzer_generic.base_report(metrics=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.base_report(metrics=[])
        self.assertEqual(-1, error)
        error = self.analyzer_generic.base_report(metrics=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.base_report(categories=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.base_report(categories=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.base_report(properties=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.base_report(properties=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.base_report(show_categories=10)
        self.assertEqual(-1, error)

        error = self.analyzer_generic.base_report(show_properties=10)
        self.assertEqual(-1, error)


if __name__ == '__main__':
    unittest.main()
