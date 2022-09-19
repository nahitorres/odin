import os
import unittest

from odin.classes import DatasetLocalization, TaskType


class DatasetLocalizationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir_path = os.path.dirname(os.path.realpath(__file__))
        cls.ds_path = os.path.join(cls.dir_path, "data/gt.json")
        cls.props_path = [("Test", os.path.join(cls.dir_path, "data/predictions"))]

    def setUp(self):
        properties_path = os.path.join(self.dir_path, "properties.json")
        self.dataset = DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION,
                                           proposals_paths=self.props_path, similar_classes=[[1, 2, 3]],
                                           save_graphs_as_png=False,
                                           properties_file=properties_path)

    def test_DatasetLocalization(self):
        with self.assertRaises(TypeError):
            DatasetLocalization(10, TaskType.OBJECT_DETECTION)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, 10)
        with self.assertRaises(ValueError):
            DatasetLocalization(self.ds_path, TaskType.CLASSIFICATION_MULTI_LABEL)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION, proposals_paths=10)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION, images_set_name=10)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION, images_abs_path=10)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION, result_saving_path=10)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION, similar_classes=10)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION, properties_file=10)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION, for_analysis=10)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION, match_on_filename=10)
        with self.assertRaises(TypeError):
            DatasetLocalization(self.ds_path, TaskType.OBJECT_DETECTION, save_graphs_as_png=10)

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
        self.assertEqual(5, result["small"])
        self.assertEqual(11, result["medium"])
        self.assertEqual(8, result["large"])

    def test_show_distribution_of_property(self):
        error = self.dataset.show_distribution_of_property(10)
        self.assertEqual(-1, error)

        result = self.dataset.show_distribution_of_property("size", show=False)
        self.assertEqual(23, result["property"]["small"])
        self.assertEqual(22, result["property"]["medium"])
        self.assertEqual(23, result["property"]["large"])

        for exp_res, cat in zip([5, 9, 9], ["cat", "dog", "fox"]):
            self.assertEqual(exp_res, result["small"][cat])
        for exp_res, cat in zip([11, 6, 5], ["cat", "dog", "fox"]):
            self.assertEqual(exp_res, result["medium"][cat])
        for exp_res, cat in zip([8, 10, 5], ["cat", "dog", "fox"]):
            self.assertEqual(exp_res, result["large"][cat])

    def test_show_distribution_of_categories(self):
        result = self.dataset.show_distribution_of_categories(show=False)
        for exp_res, cat in zip([24, 25, 19], ["cat", "dog", "fox"]):
            self.assertEqual(exp_res, result[cat])

    # -- QUERY TEST -- #

    def test_get_images_from_categories(self):
        error = self.dataset.get_images_from_categories(10)
        self.assertEqual(-1, error)

        result = self.dataset.get_images_from_categories(["10"])
        self.assertTrue(result.empty)

        result = self.dataset.get_images_from_categories(["cat"])
        self.assertEqual(18, len(result.index))

    def test_get_number_of_images(self):
        result = self.dataset.get_number_of_images()
        self.assertEqual(30, result)

    def test_is_similar(self):
        error = self.dataset.is_similar("10", 2)
        self.assertFalse(error)

        error = self.dataset.is_similar(1, "10")
        self.assertFalse(error)

        result = self.dataset.is_similar(1, 2)
        self.assertTrue(result)

    def test_get_anns_for_category(self):
        error = self.dataset.get_anns_for_category("cat")
        self.assertEqual(-1, error)

        error = self.dataset.get_anns_for_category(22)
        self.assertTrue(error.empty)

        result = self.dataset.get_anns_for_category(1)
        self.assertEqual(24, len(result.index))

        for i, row in result.iterrows():
            self.assertEqual(1, row["category_id"])

    def test_get_number_of_annotations(self):
        result = self.dataset.get_number_of_annotations()
        self.assertEqual(68, result)

    def test_get_annotations_of_class_with_property(self):
        error = self.dataset.get_annotations_of_class_with_property("aa", "size", "small")
        self.assertEqual(-1, error)
        error = self.dataset.get_annotations_of_class_with_property(1, 10, "small")
        self.assertEqual(-1, error)
        error = self.dataset.get_annotations_of_class_with_property(1, "size", ["small"])
        self.assertEqual(-1, error)

        result = self.dataset.get_annotations_of_class_with_property(1, "size", "small")
        self.assertEqual(5, len(result.index))
        for i, row in result.iterrows():
            self.assertEqual(1, row["category_id"])

    def test_get_annotations_with_property(self):
        error = self.dataset.get_annotations_with_property("aaa", "small")
        self.assertEqual(-1, error)
        error = self.dataset.get_annotations_with_property("size", "aaa")
        self.assertTrue(error.empty)

        result = self.dataset.get_annotations_with_property("size", "small")
        self.assertEqual(23, len(result.index))

    def test_get_annotations(self):
        result = self.dataset.get_annotations()
        self.assertEqual(68, len(result.index))

    def test_get_annotations_from_image(self):
        error = self.dataset.get_annotations_from_image("10")
        self.assertEqual(-1, error)

        result = self.dataset.get_annotations_from_image(1)
        self.assertEqual(3, len(result.index))

    def test_get_annotations_from_image_and_categories(self):
        error = self.dataset.get_annotations_from_image_and_categories("10", [1, 2])
        self.assertEqual(-1, error)

        error = self.dataset.get_annotations_from_image_and_categories(1, "[1, 2]")
        self.assertEqual(-1, error)

        result = self.dataset.get_annotations_from_image_and_categories(100, [1, 2])
        self.assertTrue(result.empty)

        result = self.dataset.get_annotations_from_image_and_categories(1, [100, 200])
        self.assertTrue(result.empty)

        result = self.dataset.get_annotations_from_image_and_categories(1, [1, 2])
        self.assertEqual(2, len(result.index))

    def test_get_annotation_from_id(self):
        error = self.dataset.get_annotation_from_id("aaa")
        self.assertEqual(-1, error)

        result = self.dataset.get_annotation_from_id(100)
        self.assertTrue(result.empty)

        result = self.dataset.get_annotation_from_id(5)
        self.assertEqual(5, result["id"])

    def test_get_proposals_with_ids(self):
        error = self.dataset.get_proposals_with_ids("aaa", "Test")
        self.assertEqual(-1, error)

        result = self.dataset.get_proposals_with_ids(["aaa"], "Test")
        self.assertTrue(result.empty)

        result = self.dataset.get_proposals_with_ids([1, 10], "Test")
        self.assertEqual(2, len(result.index))

    def test_get_proposals_from_image_id(self):
        error = self.dataset.get_proposals_from_image_id("10", "Test")
        self.assertEqual(-1, error)

        result = self.dataset.get_proposals_from_image_id(100, "Test")
        self.assertTrue(result.empty)

        result = self.dataset.get_proposals_from_image_id(1, "Test")
        self.assertEqual(3, len(result.index))

    def test_get_proposals_from_image_id_and_categories(self):
        error = self.dataset.get_proposals_from_image_id_and_categories("10", [1, 2], "Test")
        self.assertEqual(-1, error)

        error = self.dataset.get_proposals_from_image_id_and_categories(1, "[1, 2]", "Test")
        self.assertEqual(-1, error)

        result = self.dataset.get_proposals_from_image_id_and_categories(100, [1, 2], "Test")
        self.assertTrue(result.empty)

        result = self.dataset.get_proposals_from_image_id_and_categories(1, [1, 2], "Test")
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
        self.assertEqual(104, len(result.index))

    def test_get_proposals_of_category(self):
        error = self.dataset.get_proposals_of_category(10, "Test")
        self.assertEqual(-1, error)

        result = self.dataset.get_proposals_of_category("cat", "Test")
        self.assertEqual(34, len(result.index))
        for i, row in result.iterrows():
            self.assertEqual(1, row["category_id"])


if __name__ == '__main__':
    unittest.main()
