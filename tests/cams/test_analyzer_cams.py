import json
import os
import unittest
import numpy as np

from odin.classes import TaskType, DatasetCAMs, AnalyzerCAMs


class AnalyzerCAMsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        task = TaskType.CLASSIFICATION_SINGLE_LABEL

        dir_path = os.path.dirname(os.path.realpath(__file__))

        ds_path = os.path.join(dir_path, "data/gt.json")
        cams_path = [("Test", os.path.join(dir_path, "data/predictions"))]
        cls.dataset_generic = DatasetCAMs(ds_path, task, cams_path, save_graphs_as_png=False, load_properties=False)
        file = open(os.path.join(dir_path, "data/expected_results.json"), "r")
        cls.expected_results_generic = json.load(file)
        file.close()

        ds_path = os.path.join(dir_path, "data/perfect_case/gt.json")
        cams_path = [("Test", os.path.join(dir_path, "data/perfect_case/predictions"))]
        cls.dataset_perfect = DatasetCAMs(ds_path, task, cams_path, save_graphs_as_png=False, load_properties=False)
        file = open(os.path.join(dir_path, "data/perfect_case/expected_results.json"), "r")
        cls.expected_results_perfect = json.load(file)
        file.close()

        ds_path = os.path.join(dir_path, "data/worst_case/gt.json")
        cams_path = [("Test", os.path.join(dir_path, "data/worst_case/predictions"))]
        cls.dataset_worst = DatasetCAMs(ds_path, task, cams_path, save_graphs_as_png=False, load_properties=False)
        file = open(os.path.join(dir_path, "data/worst_case/expected_results.json"), "r")
        cls.expected_results_worst = json.load(file)
        file.close()

        cls.categories = cls.dataset_generic.get_categories_names()

    def setUp(self):
        self.analyzer_generic = AnalyzerCAMs('Test', self.dataset_generic, save_graphs_as_png=False)
        self.analyzer_perfect = AnalyzerCAMs('Test', self.dataset_perfect, save_graphs_as_png=False)
        self.analyzer_worst = AnalyzerCAMs('Test', self.dataset_worst, save_graphs_as_png=False)

    # --- TEST METRICS --- #

    def _support_mask_coverage(self, cam, masks, threshold, threshold_coverage, expected_result):
        result = self.analyzer_generic._AnalyzerCAMs__gt_mask_coverage(cam, masks, threshold, threshold_coverage)
        self.assertEqual(expected_result, result)

    def test_mask_coverage(self):
        self._support_mask_coverage(
            cam=np.array([[1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 1, 1]]),
            masks=[np.array([[1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]])
                   ],
            threshold=0.5, threshold_coverage=0.2, expected_result=1
        )

        self._support_mask_coverage(
            cam=np.array([[1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            masks=[np.array([[1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]])
                   ],
            threshold=0.5, threshold_coverage=0.2, expected_result=0.5
        )

        self._support_mask_coverage(
            cam=np.array([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            masks=[np.array([[1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]])
                   ],
            threshold=0.5, threshold_coverage=0.2, expected_result=0
        )

    def _support_global_iou(self, cam, mask, threshold, expected_result):
        result, _ = self.analyzer_generic._AnalyzerCAMs__calc_global_metrics(cam, mask, threshold)
        self.assertEqual(expected_result, result)

    def test_global_iou(self):
        self._support_global_iou(
            cam=np.array([[1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 1, 1]]),
            mask=np.array([[1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]]),
            threshold=0.5, expected_result=1
        )

        self._support_global_iou(
            cam=np.array([[1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            mask=np.array([[1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]]),
            threshold=0.5, expected_result=0.5
        )

        self._support_global_iou(
            cam=np.array([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            mask=np.array([[1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]]),
            threshold=0.5, expected_result=0
        )

    def _support_components_iou(self, cam, masks, threshold, expected_result):
        result = self.analyzer_generic._AnalyzerCAMs__calc_components_iou(cam, masks, threshold)
        self.assertEqual(expected_result, result)

    def test_components_iou(self):
        self._support_components_iou(
            cam=np.array([[1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 1, 1]]),
            masks=[np.array([[1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]])
                   ],
            threshold=0.5, expected_result=1
        )

        self._support_components_iou(
            cam=np.array([[1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            masks=[np.array([[1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]])
                   ],
            threshold=0.5, expected_result=0.5
        )

        self._support_components_iou(
            cam=np.array([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            masks=[np.array([[1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]])
                   ],
            threshold=0.5, expected_result=0
        )

    def _support_irrelevant_attention(self, cam, mask, threshold, expected_result):
        _, result = self.analyzer_generic._AnalyzerCAMs__calc_global_metrics(cam, mask, threshold)
        self.assertEqual(expected_result, result)

    def test_irrelevant_attention(self):
        self._support_irrelevant_attention(
            cam=np.array([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            mask=np.array([[1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]]),
            threshold=0.5, expected_result=1
        )

        self._support_irrelevant_attention(
            cam=np.array([[1, 1, 1, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            mask=np.array([[1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]]),
            threshold=0.5, expected_result=0.5
        )

        self._support_irrelevant_attention(
            cam=np.array([[1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 1, 1]]),
            mask=np.array([[1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]]),
            threshold=0.5, expected_result=0
        )

    def _support_analyze_cams(self, analyzer, expected_results):
        results = analyzer.analyze_cams(show=False)["overall"]
        for i, t in enumerate(results["threshold"]):
            self.assertEqual(round(results["metric_values"][i], 5), round(expected_results[str(t)]["global_iou"], 5))

    # -- TEST ANALYSES -- #

    def test_analyze_cams(self):
        self._support_analyze_cams(self.analyzer_generic, self.expected_results_generic)
        self._support_analyze_cams(self.analyzer_perfect, self.expected_results_perfect)
        self._support_analyze_cams(self.analyzer_worst, self.expected_results_worst)

    def _support_analyze_cams_for_categories(self, analyzer, expected_results):
        results = analyzer.analyze_cams_for_categories(show=False)
        for category in self.categories:
            for i, t in enumerate(results[category]["threshold"]):
                self.assertEqual(round(results[category]["metric_values"][i], 5),
                                 round(expected_results[str(t)]["global_iou"], 5))

    def test_analyze_cams_for_categories(self):
        self._support_analyze_cams_for_categories(self.analyzer_generic, self.expected_results_generic)
        self._support_analyze_cams_for_categories(self.analyzer_perfect, self.expected_results_perfect)
        self._support_analyze_cams_for_categories(self.analyzer_worst, self.expected_results_worst)

    # -- TEST INPUT PARSING -- #

    def test_AnalyzerCAMs(self):
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", 10)
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", self.dataset_generic, cam_thresh="10")
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", self.dataset_generic, cam_coverage_thresh="10")
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", self.dataset_generic, result_saving_path=10)
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", self.dataset_generic, use_normalization=10)
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", self.dataset_generic, norm_factor_categories="10")
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", self.dataset_generic, norm_factors_properties=10)
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", self.dataset_generic, conf_thresh="10")
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", self.dataset_generic, metric=10)
        with self.assertRaises(TypeError):
            AnalyzerCAMs("Test", self.dataset_generic, save_graphs_as_png=10)


    def test_analyze_cams_input(self):
        error = self.analyzer_generic.analyze_cams(metric=10)
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_cams(show=10)
        self.assertEqual(-1, error)

    def test_analyze_cams_for_categories_input(self):
        error = self.analyzer_generic.analyze_cams_for_categories(categories=10)
        self.assertEqual(-1, error)
        error = self.analyzer_generic.analyze_cams_for_categories(categories=["10"])
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_cams_for_categories(metric=10)
        self.assertEqual(-1, error)

        error = self.analyzer_generic.analyze_cams_for_categories(show=10)
        self.assertEqual(-1, error)


if __name__ == '__main__':
    unittest.main()
