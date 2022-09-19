import os
import unittest

from odin.classes import DatasetClassification, TaskType


class DatasetClassificationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__))


        cls.ds_path = os.path.join(cls.dir_path, "multilabel/data/gt.json")
        cls.props_path = [("Test", os.path.join(cls.dir_path, "multilabel/data/predictions"))]

    def setUp(self):
        properties_path = os.path.join(self.dir_path, "properties.json")
        self.dataset = DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL,
                                             proposals_paths=self.props_path, similar_classes=[[1, 2, 3]],
                                             save_graphs_as_png=False,
                                             properties_file=properties_path)

    def test_DatasetClassification(self):
        with self.assertRaises(TypeError):
            DatasetClassification(10, TaskType.CLASSIFICATION_MULTI_LABEL)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, 10)
        with self.assertRaises(ValueError):
            DatasetClassification(self.ds_path, TaskType.OBJECT_DETECTION)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL, proposals_paths=10)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL, observations_set_name=10)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL, observations_abs_path=10)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL, result_saving_path=10)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL, similar_classes=10)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL, properties_file=10)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL, for_analysis=10)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL, match_on_filename=10)
        with self.assertRaises(TypeError):
            DatasetClassification(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL, save_graphs_as_png=10)

    # -- DISTRIBUTION TEST -- #

    def test_show_distribution_of_property_for_category(self):
        error = self.dataset.show_distribution_of_property_for_category(10, "cat")
        self.assertEqual(-1, error)

        error = self.dataset.show_distribution_of_property_for_category("size", 10)
        self.assertEqual(-1, error)

        error = self.dataset.show_distribution_of_property_for_category("aaa", "cat")
        self.assertEqual(-1, error)

        error = self.dataset.show_distribution_of_property_for_category("size", "10")
        self.assertEqual(-1, error)

        result = self.dataset.show_distribution_of_property_for_category("size", "cat", show=False)
        self.assertEqual(10, result["small"])
        self.assertEqual(5, result["medium"])
        self.assertEqual(0, result["large"])

    def test_show_distribution_of_property(self):
        error = self.dataset.show_distribution_of_property(10)
        self.assertEqual(-1, error)

        result = self.dataset.show_distribution_of_property("size", show=False)
        self.assertEqual(10, result["property"]["small"])
        self.assertEqual(10, result["property"]["medium"])
        self.assertEqual(10, result["property"]["large"])

        for exp_res, cat in zip([10, 0, 2], ["cat", "dog", "fox"]):
            self.assertEqual(exp_res, result["small"][cat])
        for exp_res, cat in zip([5, 9, 10], ["cat", "dog", "fox"]):
            self.assertEqual(exp_res, result["medium"][cat])
        for exp_res, cat in zip([0, 9, 2], ["cat", "dog", "fox"]):
            self.assertEqual(exp_res, result["large"][cat])

    def test_show_distribution_of_categories(self):
        result = self.dataset.show_distribution_of_categories(show=False)
        for exp_res, cat in zip([15, 18, 14], ["cat", "dog", "fox"]):
            self.assertEqual(exp_res, result[cat])

    # -- QUERY TEST -- #

    def test_get_observation_id_from_file_name(self):
        error = self.dataset.get_observation_id_from_file_name(10)
        self.assertEqual(-1, error)

        error = self.dataset.get_observation_id_from_file_name("10")
        self.assertEqual(-1, error)

    def test_get_observations_from_ids(self):
        error = self.dataset.get_observations_from_ids(10)
        self.assertEqual(-1, error)

        result = self.dataset.get_observations_from_ids(["10"])
        self.assertTrue(result.empty)

        result = self.dataset.get_observations_from_ids([1, 2])
        for expected_index, (i, row) in zip([1, 2], result.iterrows()):
            self.assertEqual(expected_index, row["id"])

    def test_get_observations_from_categories(self):
        error = self.dataset.get_observations_from_categories(10)
        self.assertEqual(-1, error)

        result = self.dataset.get_observations_from_categories([10])
        self.assertTrue(result.empty)

        result = self.dataset.get_observations_from_categories(["cat", "dog"])
        self.assertEqual(29, len(result.index))
        # expected_ids = [1, 2]
        for i, row in result.iterrows():
            self.assertTrue(1 in row["categories"] or 2 in row["categories"])

    def test_get_observations_from_property_category(self):
        error = self.dataset.get_observations_from_property_category("10", "size", "small")
        self.assertEqual(-1, error)

        error = self.dataset.get_observations_from_property_category(1, 10, "small")
        self.assertEqual(-1, error)

        error = self.dataset.get_observations_from_property_category(1, "aaa", "small")
        self.assertEqual(-1, error)

        error = self.dataset.get_observations_from_property_category(1, "size", "aaa")
        self.assertTrue(error.empty)

        result = self.dataset.get_observations_from_property_category(1, "size", "small")
        self.assertEqual(10, len(result.index))
        for i, row in result.iterrows():
            self.assertIn(1, row["categories"])

    def test_get_observations_from_property(self):
        error = self.dataset.get_observations_from_property(10, "small")
        self.assertEqual(-1, error)

        error = self.dataset.get_observations_from_property("size", 10)
        self.assertTrue(error.empty)

        result = self.dataset.get_observations_from_property("size", "small")
        self.assertEqual(10, len(result.index))
        for i, row in result.iterrows():
            self.assertIn(1, row["categories"])

    def test_get_number_of_observations(self):
        result = self.dataset.get_number_of_observations()
        self.assertEqual(30, result)

    def test_get_all_observations(self):
        result = self.dataset.get_all_observations()
        self.assertEqual(30, len(result.index))

    def test_is_similar(self):
        error = self.dataset.is_similar("10", [1, 2])
        self.assertEqual(-1, error)

        error = self.dataset.is_similar(1, "[1, 2]")
        self.assertEqual(-1, error)

        result = self.dataset.is_similar(1, [1, 2])
        self.assertTrue(result)

    def test_get_proposals_from_observation_id(self):
        error = self.dataset.get_proposals_from_observation_id("10", "Test")
        self.assertEqual(-1, error)

        result = self.dataset.get_proposals_from_observation_id(100, "Test")
        self.assertTrue(result.empty)

        result = self.dataset.get_proposals_from_observation_id(15, "Test")
        self.assertEqual(3, len(result.index))

    def test_get_proposals_from_observation_id_and_categories(self):
        error = self.dataset.get_proposals_from_observation_id_and_categories("10", [1, 2], "Test")
        self.assertEqual(-1, error)

        error = self.dataset.get_proposals_from_observation_id_and_categories(10, "[1, 2]", "Test")
        self.assertEqual(-1, error)

        result = self.dataset.get_proposals_from_observation_id_and_categories(100, [1, 2], "Test")
        self.assertTrue(result.empty)

        result = self.dataset.get_proposals_from_observation_id_and_categories(15, [1, 2], "Test")
        self.assertEqual(2, len(result.index))

    def test_get_category_name_from_id(self):
        error = self.dataset.get_category_name_from_id("10")
        self.assertEqual(-1, error)

        result = self.dataset.get_category_name_from_id(10)
        self.assertIsNone(result)

        result = self.dataset.get_category_name_from_id(1)
        self.assertEqual("cat", result)

    def test_is_valid_category(self):
        error = self.dataset.is_valid_category(10)
        self.assertFalse(error)

        result = self.dataset.is_valid_category("aaa")
        self.assertFalse(result)

        result = self.dataset.is_valid_category("cat")
        self.assertTrue(result)

    def test_get_categories_names(self):
        result = self.dataset.get_categories_names()
        self.assertEqual(["cat", "dog", "fox"], result)

    def test_get_categories_names_from_ids(self):
        error = self.dataset.get_categories_names_from_ids(10)
        self.assertEqual(-1, error)

        result = self.dataset.get_categories_names_from_ids([10, 12])
        self.assertEqual([None, None], result)

        result = self.dataset.get_categories_names_from_ids([1, 2])
        self.assertEqual(["cat", "dog"], result)

    def test_get_category_id_from_name(self):
        error = self.dataset.get_category_id_from_name(10)
        self.assertEqual(-1, error)

        error = self.dataset.get_category_id_from_name("10")
        self.assertEqual(None, error)

        result = self.dataset.get_category_id_from_name("cat")
        self.assertEqual(1, result)

    def test_get_categories_id_from_names(self):
        error = self.dataset.get_categories_id_from_names(10)
        self.assertEqual(-1, error)

        result = self.dataset.get_categories_id_from_names(["aaa"])
        self.assertEqual([None], result)

        result = self.dataset.get_categories_id_from_names(["cat", "fox"])
        self.assertEqual([1, 3], result)

    def test_are_valid_categories(self):
        error = self.dataset.are_valid_categories(10)
        self.assertFalse(error)

        result = self.dataset.are_valid_categories(["10"])
        self.assertFalse(result)

        result = self.dataset.are_valid_categories(["cat", "dog"])
        self.assertTrue(result)

    def test_get_values_for_property(self):
        error = self.dataset.get_values_for_property([10])
        self.assertEqual(-1, error)

        result = self.dataset.get_values_for_property("size")
        for v in ["small", "medium", "large"]:
            self.assertIn(v, result)

    def test_are_valid_properties(self):
        error = self.dataset.are_valid_properties(10)
        self.assertFalse(error)

        result = self.dataset.are_valid_properties(["10"])
        self.assertFalse(result)

        result = self.dataset.are_valid_properties(["size"])
        self.assertTrue(result)

    def test_is_valid_property(self):
        error = self.dataset.is_valid_property(10, ["small", "medium"])
        self.assertFalse(error)

        error = self.dataset.is_valid_property("size", 10)
        self.assertFalse(error)

        error = self.dataset.is_valid_property("size", ["small", "medium"])
        self.assertTrue(error)

    def test_get_proposals(self):
        result = self.dataset.get_proposals("Test")
        self.assertEqual(90, len(result.index))

    def test_get_proposals_of_category(self):
        error = self.dataset.get_proposals_of_category(10, "Test")
        self.assertEqual(-1, error)

        result = self.dataset.get_proposals_of_category("cat", "Test")
        self.assertEqual(30, len(result.index))
        for i, row in result.iterrows():
            self.assertEqual(1, row["category_id"])


if __name__ == '__main__':
    unittest.main()
