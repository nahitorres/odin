import os
import unittest

from odin.classes import ComparatorLocalization, TaskType


class ComparatorLocalizationTest(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__))
        cls.ds_path = os.path.join(cls.dir_path, "data/gt.json")
        cls.models_props = [('model_A', os.path.join(cls.dir_path, "data/predictions")),
                            ('model_B', os.path.join(cls.dir_path, "data/predictions"))]

    def setUp(self):
        properties_path = os.path.join(self.dir_path, "properties.json")
        self.comparator = ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION,
                                                   multiple_proposals_path=self.models_props,
                                                   save_graph_as_png=False,
                                                 properties_file=properties_path)

    def test_ComparatorClassification(self):
        with self.assertRaises(TypeError):
            ComparatorLocalization(10, TaskType.OBJECT_DETECTION, self.models_props)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, 10, self.models_props)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, 10)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     result_saving_path=10)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     properties_file=10)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     use_normalization=10)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     norm_factor_categories="10")
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     norm_factors_properties=10)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     conf_thresh="10")
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     iou="10")
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     metric=10)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     similar_classes=10)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     load_properties=10)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     match_on_filename=10)
        with self.assertRaises(TypeError):
            ComparatorLocalization(self.ds_path, TaskType.OBJECT_DETECTION, self.models_props,
                                     save_graph_as_png=10)

    def test_analyze_property(self):
        error = self.comparator.analyze_property(10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_property("size", possible_values=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_property("size", metric=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_property("size", show=10)
        self.assertEqual(-1, error)

    def test_analyze_sensitivity_impact_of_properties(self):
        error = self.comparator.analyze_sensitivity_impact_of_properties(properties=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_sensitivity_impact_of_properties(metric=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_sensitivity_impact_of_properties(show=10)
        self.assertEqual(-1, error)

    def test_analyze_curve(self):
        error = self.comparator.analyze_curve(curve=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_curve(average=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_curve(show=10)
        self.assertEqual(-1, error)

    def test_analyze_curve_for_categories(self):
        error = self.comparator.analyze_curve_for_categories(curve=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_curve_for_categories(categories=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_curve_for_categories(show=10)
        self.assertEqual(-1, error)

    def test_analyze_false_positive_errors(self):
        error = self.comparator.analyze_false_positive_errors(metric=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_false_positive_errors(categories=10)
        self.assertEqual(-1, error)
        error = self.comparator.analyze_false_positive_errors(show=10)
        self.assertEqual(-1, error)

    def test_show_true_positive_distribution(self):
        error = self.comparator.show_true_positive_distribution(categories=10)
        self.assertEqual(-1, error)
        error = self.comparator.show_true_positive_distribution(show=10)
        self.assertEqual(-1, error)

    def test_show_false_positive_distribution(self):
        error = self.comparator.show_false_positive_distribution(categories=10)
        self.assertEqual(-1, error)
        error = self.comparator.show_false_positive_distribution(show=10)
        self.assertEqual(-1, error)

    def test_show_false_negative_distribution(self):
        error = self.comparator.show_false_negative_distribution(categories=10)
        self.assertEqual(-1, error)
        error = self.comparator.show_false_negative_distribution(show=10)
        self.assertEqual(-1, error)

    def test_base_report(self):
        error = self.comparator.base_report(metrics=10)
        self.assertEqual(-1, error)
        error = self.comparator.base_report(categories=10)
        self.assertEqual(-1, error)
        error = self.comparator.base_report(properties=10)
        self.assertEqual(-1, error)
        error = self.comparator.base_report(show_categories=10)
        self.assertEqual(-1, error)
        error = self.comparator.base_report(show_properties=10)
        self.assertEqual(-1, error)
        error = self.comparator.base_report(include_reliability=10)
        self.assertEqual(-1, error)

    # property_name, property_values = None, categories = None, show = True
    def test_show_true_positive_distribution_for_categories_for_property(self):
        error = self.comparator.show_true_positive_distribution_for_categories_for_property(10)
        self.assertEqual(-1, error)
        error = self.comparator.show_true_positive_distribution_for_categories_for_property("size", property_values=10)
        self.assertEqual(-1, error)
        error = self.comparator.show_true_positive_distribution_for_categories_for_property("size", categories=10)
        self.assertEqual(-1, error)
        error = self.comparator.show_true_positive_distribution_for_categories_for_property("size", show=10)
        self.assertEqual(-1, error)

    def test_show_false_negative_distribution_for_categories_for_property(self):
        error = self.comparator.show_false_negative_distribution_for_categories_for_property(10)
        self.assertEqual(-1, error)
        error = self.comparator.show_false_negative_distribution_for_categories_for_property("size", property_values=10)
        self.assertEqual(-1, error)
        error = self.comparator.show_false_negative_distribution_for_categories_for_property("size", categories=10)
        self.assertEqual(-1, error)
        error = self.comparator.show_false_negative_distribution_for_categories_for_property("size", show=10)
        self.assertEqual(-1, error)


if __name__ == '__main__':
    unittest.main()
