import os

import json
from numbers import Number

import numpy as np
import pandas as pd

from odin.classes import DatasetInterface, TaskType
from odin.classes.strings import *
from odin.utils import *
from odin.utils.draw_utils import pie_plot, display_co_occurrence_matrix
from odin.utils.env import is_notebook
from odin.utils.utils import encode_segmentation, compute_aspect_ratio_of_segmentation
from pycocotools import mask

logger = get_root_logger()


class DatasetLocalization(DatasetInterface):

    annotations = None
    images = None

    area_size_computed = False
    aspect_ratio = False

    common_properties = {'index', 'area', 'bbox', 'path', 'category_id', 'id', 'image_id', 'iscrowd', 'segmentation'}
    supported_types = [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]

    def __init__(self,
                 dataset_gt_param,
                 task_type,
                 proposals_paths=None,
                 images_set_name='test',
                 images_abs_path=None,
                 result_saving_path='./results/',
                 similar_classes=None,
                 properties_file=None,
                 for_analysis=True,
                 load_properties=True,
                 match_on_filename=False,
                 save_graphs_as_png=True,
                 ignore_proposals_threshold=0.01):
        """
        The DatasetLocalization class can be used to store the ground truth and predictions for localization models, such as object detection and instance segmentation.

        Parameters
        ----------
        dataset_gt_param: str
            Path of the ground truth .json file.
        task_type: TaskType
            Problem task type. It can be: TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION<.
        proposals_paths: list of tuple, optional
            List of couples. Each couple contains the model name and the corresponding proposals path. (default is None)
        images_set_name:str, optional
            Name of the data set. (default is 'test')
        images_abs_path: str, optional
            Path of the images directory. (default is None)
        result_saving_path: str, optional
            Path used to save results. (default is './results/')
        similar_classes: list of list, optional
            List of groups of ids of categories which are similar to each other. (default is None)
        properties_file: str, optional
            The name of the file used to store the names of and values of the properties and the names of the categories. (default is 'properties.json')
        for_analysis: bool, optional
            Indicates whether the properties and the predictions have to be loaded. If False, only the ground truth is loaded. (default is True)
        load_properties: bool, optional
            Indicates whether the properties should be loaded. (default is True)
        match_on_filename: bool, optional
            Indicates whether the predictions refer to the ground truth by file_name (set to True) or by id (set to False). (default is False)
        save_graphs_as_png: bool, optional
            Indicates whether plots should be saved as .png images. (default is True)
        ignore_proposals_threshold: float, optional
            All the proposals with a confidence score lower than the threshold are not loaded. (Default is 0.01)
        """
        if not isinstance(dataset_gt_param, str):
            raise TypeError(err_type.format("dataset_gt_param"))

        if not isinstance(task_type, TaskType):
            raise TypeError(err_type.format("task_type"))
        elif task_type not in self.supported_types:
            raise ValueError(err_value.format("task_type", self.supported_types))

        if proposals_paths is not None:
            if isinstance(proposals_paths, str):
                proposals_paths = [("model", proposals_paths)]
            elif (not isinstance(proposals_paths, list) or
                  not all(isinstance(c, tuple) for c in proposals_paths) or
                  not all(isinstance(n, str) and isinstance(p, str) for n, p in proposals_paths)):
                raise TypeError(err_type.format("proposals_paths"))

        if not isinstance(images_set_name, str):
            raise TypeError(err_type.format("images_set_name"))

        if images_abs_path is not None and not isinstance(images_abs_path, str):
            raise TypeError(err_type.format("images_abs_path"))

        if similar_classes is not None and (not isinstance(similar_classes, list) or not all(isinstance(v, list) for v in similar_classes)):
            raise TypeError(err_type.format("similar_classes"))

        if properties_file is not None and not isinstance(properties_file, str):
            raise TypeError(err_type.format("properties_file"))

        if not isinstance(for_analysis, bool):
            raise TypeError(err_type.format("for_analysis"))

        if not isinstance(match_on_filename, bool):
            raise TypeError(err_type.format("match_on_filename"))

        if not isinstance(save_graphs_as_png, bool):
            raise TypeError(err_type.format("save_graphs_as_png"))

        if not isinstance(ignore_proposals_threshold, Number):
            raise TypeError(err_type.format("ignore_proposals_threshold"))
        elif ignore_proposals_threshold < 0 or ignore_proposals_threshold > 1:
            raise ValueError(err_value.format("ignore_proposals_threshold", "0 <= x <= 1"))

        super().__init__(dataset_gt_param, proposals_paths, task_type, images_set_name, images_abs_path,
                         result_saving_path, similar_classes, properties_file, match_on_filename, save_graphs_as_png)

        self._ignore_proposals_threshold = ignore_proposals_threshold
        self._aspect_ratio_boundaries = []
        self._area_size_boundaries = []

        self.for_analysis = for_analysis
        self.load(load_properties=load_properties)
        if similar_classes is not None:
            self.similar_groups = similar_classes

    # -- DISTRIBUTION ANALYSES -- #

    def show_co_occurrence_matrix(self, show=True):
        """
        It provides the co-occurrence matrix of the categories.

        Parameters
        ----------
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the matrix. (default is True)
        """

        if not self._analyses_without_properties_available:
            logger.error("Please select the properties first or set 'for_analysis=False' when instantiate the "
                         "DatasetLocalization")
            return -1

        cat_ids = np.array(self.get_categories_id_from_names(self.get_categories_names()))
        n_size = len(cat_ids)
        occ_matrix = np.zeros((n_size, n_size))

        annotations = self.get_annotations()
        co_list = annotations.groupby("image_id")["category_id"].apply(list)

        for row in co_list:
            row = np.unique(row)
            for i in range(0, len(row)):
                id_a = np.where(cat_ids == row[i])[0][0]
                for j in range(i+1, len(row)):
                    id_b = np.where(cat_ids == row[j])[0][0]
                    occ_matrix[id_a][id_b] += 1
                    occ_matrix[id_b][id_a] += 1

        if not show:
            return occ_matrix

        labels = [self.get_display_name_of_category(self.get_category_name_from_id(i)) for i in cat_ids.tolist()]
        display_co_occurrence_matrix(occ_matrix, labels, self._get_save_graphs_as_png(), self.result_saving_path)

    def show_distribution_of_property_for_category(self, property_name, category, property_values=None, show=True):
        """
        It provides the distribution of a property for a specific category.
        Parameters
        ----------
        property_name: str
            Name of the property to be analyzed.
        category: str
            Name of the category to be analyzed.
        property_values: list, optional
            List of the property values to be included in the analysis. If not specified, all the values are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if not self._analyses_with_properties_available:
            if not self._analyses_without_properties_available:
                logger.error("Please complete the properties selection first")
            else:
                logger.error("No properties available. Please make sure to load the properties")
            return -1

        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        elif property_name not in self.get_property_keys():
            logger.error(
                err_value.format("property_name", list(self.get_property_keys())))
            return -1

        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1
        elif not self.is_valid_category(category):
            logger.error(err_value.format("category", self.get_categories_names()))
            return -1

        if property_values is None:
            property_values = self.get_values_for_property(property_name)
        elif not isinstance(property_values, list):
            logger.error(err_type.format("property_values"))
            return -1
        elif not self.is_valid_property(property_name, property_values):
            return -1

        property_name_to_show = self.get_display_name_of_property(property_name)
        cat_name_to_show = self.get_display_name_of_category(category)

        display_names = [self.get_display_name_of_property_value(property_name, v) for v in property_values]

        anns = self.get_anns_for_category(self.get_category_id_from_name(category))
        count = anns.groupby(property_name).size()
        p_values = count.index.tolist()
        sizes = []
        result = {}
        for pv in property_values:
            if pv not in p_values:
                sizes.append(0)
                result[pv] = 0
            else:
                value = count[pv]
                sizes.append(value)
                result[pv] = value

        if not show:
            return result

        title = "Distribution of {} for {}".format(property_name_to_show, cat_name_to_show)
        output_path = os.path.join(self.result_saving_path, f"distribution_{str(property_name).replace('/', '_')}_{category}"
                                                            f".png")
        pie_plot(sizes, display_names, title, output_path, self._get_save_graphs_as_png())

    def show_distribution_of_property(self, property_name, property_values=None, show=True):
        """
        It provides the distribution of the different values of a property and for each value shows the distribution of the categories.

        Parameters
        ----------
        property_name: str
            Name of the property to be analyzed.
        property_values: list, optional
            List of the property values to be included in the analysis. If not specified, all the values are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)

        """
        if not self._analyses_with_properties_available:
            if not self._analyses_without_properties_available:
                logger.error("Please complete the properties selection first")
            else:
                logger.error("No properties available. Please make sure to load the properties")
            return -1

        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        elif not self.are_valid_properties([property_name]):
            return -1

        if property_values is None:
            property_values = self.get_values_for_property(property_name)
        elif not isinstance(property_values, list):
            logger.error(err_type.format("property_values"))
            return -1
        elif not self.is_valid_property(property_name, property_values):
            return -1

        property_name_to_show = self.get_display_name_of_property(property_name)

        display_names = [self.get_display_name_of_property_value(property_name, v) for v in property_values]

        anns = self.get_annotations()
        count = anns.groupby(property_name).size()
        p_values = count.index.tolist()
        sizes = []
        results = {"property": {}}
        for pv in property_values:
            if pv not in p_values:
                sizes.append(0)
                results["property"][pv] = 0
            else:
                value = count[pv]
                sizes.append(value)
                results["property"][pv] = value
        if show:
            title = "Distribution of {}".format(property_name_to_show)
            output_path = os.path.join(self.result_saving_path, f"distribution_total_{str(property_name).replace('/', '_')}"
                                                                f".png")
            pie_plot(sizes, display_names, title, output_path, self._get_save_graphs_as_png())

        labels = [self.get_display_name_of_category(cat) for cat in self.get_categories_names()]
        for pv in property_values:
            results[pv] = {}
            sizes = []
            for cat_name in self.get_categories_names():
                cat_id = self.get_category_id_from_name(cat_name)
                value = len(self.get_annotations_of_class_with_property(cat_id, property_name, pv).index)
                sizes.append(value)
                results[pv][cat_name] = value
            if show:
                if np.sum(sizes) > 0:
                    title = "Distribution of {} among categories".format(self.get_display_name_of_property_value(
                        property_name, pv))
                    output_path = os.path.join(self.result_saving_path, f"distribution_{str(pv).replace('/', '_')}"
                                                                        f"_in_categories.png")
                    pie_plot(sizes, labels, title, output_path, self._get_save_graphs_as_png())
                else:
                    logger.warning(f"No samples for property value: {pv}")
        if not show:
            return results

    def show_distribution_of_categories(self, show_avg_size=True, show=True):
        """
        It provides the distribution of the categories in the data set.

        Parameters
        ----------
        show_avg_size: bool, optional
            Indicates whether to calculate the average size of the area of the annotations. (default is True)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not self._analyses_without_properties_available:
            logger.error("Please select the properties first or set 'for_analysis=False' when instantiate the "
                         "DatasetLocalization")
            return -1
        if not isinstance(show_avg_size, bool):
            logger.error(err_type.format("show_avg_size"))
            return -1

        annotations = self.annotations.copy()
        if show_avg_size:
            annotations["area"] = annotations.apply(lambda x: self._calculate_annotation_area(x), axis=1)
        cats_groups = annotations.groupby(level="cat_id")
        sizes, display_names = [], []
        results = {}
        for name in self.get_categories_names():
            group = cats_groups.get_group(self.get_category_id_from_name(name))
            name = self.get_display_name_of_category(name)
            value = len(group.index)
            sizes.append(value)
            results[name] = value
            if show_avg_size:
                name += f" [avg size: {group['area'].mean():.2f}]"
            display_names.append(name)

        if not show:
            return results

        title = "Distribution of categories"
        output_path = os.path.join(self.result_saving_path, "distribution_categories.png")
        pie_plot(sizes, display_names, title, output_path, self._get_save_graphs_as_png())

    # -- IMAGES -- #

    def get_image_from_id(self, image_id):
        """
        Returns the image having the specified id
        Parameters
        ----------
        image_id: int
            Image id

        Returns
        -------
        pandas.Series
        """
        if not isinstance(image_id, int):
            logger.error(err_type.format("image_id"))
            return -1
        return self.images.loc[self.images["id"] == image_id].iloc[0]

    def get_images_from_ids(self, images_ids):
        """
        Returns the image having the specified id
        Parameters
        ----------
        images_ids: list
            Images ids

        Returns
        -------
        pandas.Series
        """
        if not isinstance(images_ids, list) or not all(isinstance(v, int) for v in images_ids):
            logger.error(err_type.format("images_ids"))
            return -1
        return self.images.loc[self.images["id"].isin(images_ids)]

    def get_image_id_from_image_name(self, filename):
        """
        Returns the image id from the image name
        Parameters
        ----------
        filename: str
            Name of the image

        Returns
        -------
        int
            Id of the image
        """
        if not isinstance(filename, str):
            logger.error(err_type.format("filename"))
            return -1

        imgs = self.images.loc[self.images["file_name"] == filename]["id"].tolist()
        if len(imgs) > 0:
            return imgs[0]

    def get_images_from_categories(self, categories):
        """
        Returns all the images that belong to the categories specified
        Parameters
        ----------
        categories: list
            Names of the categories

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1

        cat_ids = self.get_categories_id_from_names(categories)
        img_ids = self.annotations.loc[self.annotations["category_id"].isin(cat_ids)]["image_id"].unique()
        return self.images.loc[self.images["id"].isin(img_ids)]

    def get_height_width_from_image(self, image_id):
        """
        Returns the height and the width of the specified image
        Parameters
        ----------
        image_id: int
            Id of the image

        Returns
        -------
        height, width
        """
        if not isinstance(image_id, int):
            logger.error(err_type.format("image_id"))
            return -1

        image = self.get_image_from_id(image_id)
        return image["height"], image["width"]

    def get_images_id_with_path(self):
        """
        Returns all the ids of the images with the corresponding path
        Returns
        -------
        dict
        """
        imgs_with_path = self.images.copy()
        imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
        return imgs_with_path[["path", "id"]].to_dict("records")

    def get_images_id_with_path_from_ids(self, img_ids):
        imgs_with_path = self.get_images_from_ids(img_ids).copy()
        imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
        return imgs_with_path[["path", "id"]].to_dict("records")

    def get_images_id_with_path_for_category_with_property_value(self, category, property_name, property_value):
        """
        Returns all the ids with the corresponding path of the images that belong to the specified category and having
        the specified property value
        Parameters
        ----------
        category: str
            Name of the category
        property_name: str
            Name of the property
        property_value: str or Number
            Property value

        Returns
        -------
        dict
        """
        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1
        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        if not isinstance(property_value, str) and not isinstance(property_value, Number):
            logger.error(err_type.format("property_value"))
            return -1

        category_id = self.get_category_id_from_name(category)
        images = self.get_annotations_of_class_with_property(category_id, property_name, property_value)

        if len(images) > 0:
            img_ids = images["image_id"].unique()
            imgs_with_path = self.images.copy(deep=True)
            imgs_with_path = imgs_with_path[imgs_with_path["id"].isin(img_ids)]
            imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
            return imgs_with_path[["path", "id"]].to_dict("records")
        else:
            return []

    def get_images_id_with_path_for_category(self, category):
        """
        Returns all the ids with the corresponding path of the images that belong to the specified category
        Parameters
        ----------
        category: str
            Name of the category

        Returns
        -------
        dict
        """
        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1

        category_id = self.get_category_id_from_name(category)
        img_ids = self.get_anns_for_category(category_id)["image_id"].unique()
        imgs_with_path = self.images.copy(deep=True)
        imgs_with_path = imgs_with_path[imgs_with_path["id"].isin(img_ids)]
        imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
        return imgs_with_path[["path", "id"]].to_dict("records")

    def get_images_id_with_path_with_property_value(self, property_name, property_value):
        """
        Returns all the ids with the corresponding path of the images that have the specified property value
        Parameters
        ----------
        property_name: str
            Name of the property
        property_value: str or Number
            Property value

        Returns
        -------
        dict
        """
        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        if not isinstance(property_value, str) and not isinstance(property_value, Number):
            logger.error(err_type.format("property_value"))
            return -1

        images = self.get_annotations_with_property(property_name, property_value)
        if len(images) > 0:
            img_ids = images["image_id"].unique()
            imgs_with_path = self.images.copy(deep=True)
            imgs_with_path = imgs_with_path[imgs_with_path["id"].isin(img_ids)]
            imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
            return imgs_with_path[["path", "id"]].to_dict("records")
        else:
            return []

    def get_images_id_with_path_from_annotation_ids(self, ann_ids):
        """
        Returns all the ids with the corresponding path of the images that correspond to the specified annotations ids.
        Parameters
        ----------
        ann_ids: list
            Annotations ids

        Returns
        -------
        dict
        """
        if not isinstance(ann_ids, list):
            logger.error(err_type.format("ann_ids"))
            return -1

        img_ids = self.annotations.loc[self.annotations["id"].isin(ann_ids)]["image_id"].unique()
        imgs_with_path = self.images.copy(deep=True)
        imgs_with_path = imgs_with_path[imgs_with_path["id"].isin(img_ids)]
        imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
        return imgs_with_path[["path", "id"]].to_dict("records")

    def get_images_id_with_path_from_proposals_ids(self, props_ids, model_name):
        if not isinstance(props_ids, list):
            logger.error(err_type.format("props_ids"))
            return -1

        img_ids = self.proposals[model_name].loc[self.proposals[model_name]["id"].isin(props_ids)][self.match_param_props].unique()
        imgs_with_path = self.images.copy(deep=True)
        imgs_with_path = imgs_with_path[imgs_with_path["id"].isin(img_ids)]
        imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
        return imgs_with_path[["path", "id"]].to_dict("records")

    def get_number_of_images(self):
        """
        Returns the total number of images in the dataset
        Returns
        -------
        int
        """
        return len(self.images.index)

    # -- CATEGORIES -- #

    def is_similar(self, cat1, cat2):
        """
        Checks if two categories belongs to the same similar_group
        Parameters
        ----------
        cat1: int
            id of the first category
        cat2: int
            id of the second category

        Returns
        -------
        bool
        """
        if not isinstance(cat1, int):
            logger.error(err_type.format("cat1"))
            return False

        if not isinstance(cat2, int):
            logger.error(err_type.format("cat2"))
            return False

        for groups in self.similar_groups:
            if cat1 in groups and cat2 in groups:
                return True
        return False

    def get_property_values_for_category(self, property_name, category):
        """
        Returns all the property values for a category
        Parameters
        ----------
        property_name: str
            Name of the property
        category: str
            Name of the category

        Returns
        -------
        list
        """
        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        elif not self.is_possible_property(property_name):
            logger.error(err_value.format("property_name", list(self.get_property_keys())))
            return -1

        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1
        anns = self.get_anns_for_category(self.get_category_id_from_name(category))
        return list(anns.index.get_level_values(property_name).unique())

    # -- ANNOTATIONS -- #

    def get_annotations_from_class_list(self, categories):
        """
        Returns all the annotations belonging to a group of categories
        Parameters
        ----------
        categories: list
            Categories names that the annotations belong to

        Returns
        -------
        dict
        """
        if not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        classes_id_filter = self.get_categories_id_from_names(categories)
        anns_filtered = self.annotations.loc[self.annotations.index.get_level_values("cat_id").isin(classes_id_filter)]
        return anns_filtered.to_dict("records")

    def get_anns_for_category(self, category_id):
        """
        Returns the annotations that belong to the specified category
        Parameters
        ----------
        category_id: int
            Id of the category that the annotations belong to

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(category_id, int):
            logger.error(err_type.format("category_id"))
            return -1

        return self.annotations.loc[self.annotations.index.get_level_values("cat_id") == category_id]

    def get_number_of_annotations(self):
        """
        Returns the total number of annotations
        Returns
        -------
        int
        """
        return len(self.annotations.index)

    def get_annotations_of_class_with_property(self, category_id, property_name, value):
        """
        Returns all the annotations belonging to a specific category and having a specific property value
        Parameters
        ----------
        category_id: int
            Id of the category
        property_name: str
            Name of the property that the value belongs to
        value: str or Number
            Property value

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(category_id, int):
            logger.error(err_type.format("category_id"))
            return -1

        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1

        if not isinstance(value, str) and not isinstance(value, Number):
            logger.error(err_type.format("value"))
            return -1

        return self.annotations.loc[(self.annotations.index.get_level_values(property_name) == value) &
                                    (self.annotations.index.get_level_values("cat_id") == category_id)]

    def get_annotations_with_property(self, property_name, value):
        """
        Returns all the annotations having a specific proprerty value
        Parameters
        ----------
        property_name: str
            Name of the property that the value belongs to
        value: str or Number
            Property value
        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        elif property_name not in self.get_property_keys():
            logger.error(err_value.format("property_name", list(self.get_property_keys())))
            return -1
        if not isinstance(value, str) and not isinstance(value, Number):
            logger.error(err_type.format("value"))
            return -1

        return self.annotations.loc[self.annotations.index.get_level_values(property_name) == value]

    def get_area_size_label(self, area_size):
        for i, label in zip(range(0, len(self._area_size_boundaries)-1), ["XS", "S", "M", "L"]):
            if area_size <= self._area_size_boundaries[i]:
                return label
        return "XL"

    def get_aspect_ratio_label(self, aspect_ratio):
        for i, label in zip(range(0, len(self._aspect_ratio_boundaries)-1), ['XT', 'T', 'M', 'W']):
            if aspect_ratio <= self._aspect_ratio_boundaries[i]:
                return label
        return "XW"

    def compute_area_size(self):
        """
        Computes the area size of each annotation

        """
        if self.area_size_computed:
            return
        areas = []
        anns = self.annotations.to_dict("records")
        for index, annotation in enumerate(anns):
            if self.task_type == TaskType.INSTANCE_SEGMENTATION:
                img = self.get_image_from_id(annotation["image_id"])
                encoded_mask = encode_segmentation(annotation["segmentation"][0], img["height"],
                                                   img["width"])
                area = mask.area(encoded_mask)
            else:
                _, _, w, h = annotation['bbox']
                area = w * h
            areas.append([area, index])

        labels = ["XS", "S", "M", "L", "XL"]

        # The areas are sorted and then the labels are assigned: 0.1%-XS, 0.2%-S, 0.4%-M, 0.2%-L, 0.1%-XL
        var_values = np.asarray([1 / 10, 3 / 10, 7 / 10, 9 / 10, 1], dtype=float) * len(areas)
        positions = np.around(var_values).astype(int)

        areas = sorted(areas, key=lambda x: x[0])
        prev = 0
        for i, pos in enumerate(positions):
            for j in range(prev, pos):
                real_index = areas[j][1]
                anns[real_index]["AreaSize"] = labels[i]
            prev = pos
            self._area_size_boundaries.append(areas[pos-1][0])
        self.area_size_computed = True
        self.annotations = pd.DataFrame(anns)

    def compute_aspect_ratio(self):
        """
        Computes the aspect_ratio of each annotation

        """
        if self.aspect_ratio:
            return
        aspect_ratios = []
        anns = self.annotations.to_dict("records")
        for index, annotation in enumerate(anns):

            if self.task_type == TaskType.INSTANCE_SEGMENTATION:
                aspect_ratio = compute_aspect_ratio_of_segmentation(annotation["segmentation"][0])

            else:
                _, _, w, h = annotation['bbox']
                aspect_ratio = (w + 1) / (h + 1)
            aspect_ratios.append([aspect_ratio, index])

        labels = ['XT', 'T', 'M', 'W', 'XW']

        # The areas are sorted and then the labels are assigned: 0.1%-XT, 0.2%-T, 0.4%-M, 0.2%-W, 0.1%-XW
        var_values = np.asarray([1 / 10, 3 / 10, 7 / 10, 9 / 10, 1], dtype=float) * len(aspect_ratios)
        positions = np.around(var_values).astype(int)

        areas = sorted(aspect_ratios, key=lambda x: x[0])
        prev = 0
        for i, pos in enumerate(positions):
            for j in range(prev, pos):
                real_index = areas[j][1]
                anns[real_index]["AspectRatio"] = labels[i]
            prev = pos
            self._aspect_ratio_boundaries.append(areas[pos-1][0])
        self.aspect_ratio = True
        self.annotations = pd.DataFrame(anns)

    def get_annotations(self):
        """
        Returns all the annotations
        Returns
        -------
        pandas.DataFrame
        """
        return self.annotations

    def get_annotations_from_image(self, image_id):
        """
        Returns all the annotations that belong to the same image_id
        Parameters
        ----------
        image_id: int
            Id of the image that the annotations belong to

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(image_id, int):
            logger.error(err_type.format("image_id"))
            return -1

        return self.annotations.loc[self.annotations["image_id"] == image_id]

    def get_annotations_from_image_and_categories(self, image_id, categories_ids):
        """
        Returns all the annotations that belong to a specific image and specific categories
        Parameters
        ----------
        image_id: int
            Id of the image
        categories_ids: list
            Ids of the categories

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(image_id, int):
            logger.error(err_type.format("image_id"))
            return -1
        if not isinstance(categories_ids, list):
            logger.error(err_type.format("categories_ids"))
            return -1

        return self.annotations.loc[(self.annotations.index.get_level_values("img_id") == image_id) &
                                    (self.annotations.index.get_level_values("cat_id").isin(categories_ids))]

    def get_annotation_from_id(self, ann_id):
        """
        Returns the annotation with the specified id
        Parameters
        ----------
        ann_id: int
            Id of the annotation

        Returns
        -------
        pandas.Series
        """
        if not isinstance(ann_id, int):
            logger.error(err_type.format("ann_id"))
            return -1
        result = self.annotations.loc[self.annotations["id"] == ann_id]
        if result.empty:
            return result
        return result.iloc[0]

    def _calculate_annotation_area(self, annotation):
        if self.task_type == TaskType.INSTANCE_SEGMENTATION:
            img = self.get_image_from_id(annotation["image_id"])
            encoded_mask = encode_segmentation(annotation["segmentation"][0], img["height"],
                                               img["width"])
            return mask.area(encoded_mask)
        if self.task_type == TaskType.OBJECT_DETECTION:
            _, _, w, h = annotation['bbox']
            return w * h

    # -- PREDICTIONS -- #

    def get_proposals_with_ids(self, ids, model_name):
        """
        Returns all the proposals that have the specified ids
        Parameters
        ----------
        ids: list
            Ids of the proposals
        model_name: str
            Name of the model used for retrieving the proposals

        Returns
        -------
        pandas.DataFrame
        """

        if not isinstance(ids, list):
            logger.error(err_type.format("ids"))
            return -1

        if not isinstance(model_name, str):
            logger.error(err_type.format("model_name"))
            return -1

        if not self.proposals:
            logger.error("No proposals loaded")
            return -1
        elif model_name not in self.proposals:
            logger.error(err_value.format("model_name", list(self.proposals.keys())))
            return -1

        return self.proposals[model_name].loc[self.proposals[model_name]["id"].isin(ids)]

    def get_proposals_from_image_id(self, image_id, model_name):
        """
        Returns all the proposals that refer to a specific image

        Parameters
        ----------
        image_id: int
            Id of the image
        model_name: str
            Name of the model used for retrieving the proposals

        Returns
        -------
        pandas.DataFrame
        """

        if not isinstance(image_id, int):
            logger.error(err_type.format("image_id"))
            return -1

        if not isinstance(model_name, str):
            logger.error(err_type.format("model_name"))
            return -1

        if not self.proposals:
            logger.error("No proposals loaded")
            return -1
        elif model_name not in self.proposals:
            logger.error(err_value.format("model_name", list(self.proposals.keys())))
            return -1

        return self.proposals[model_name][self.proposals[model_name][self.match_param_props] == image_id]

    def get_proposals_from_image_id_and_categories(self, image_id, categories_ids, model_name):
        """
        Returns all the proposals of specific categories that refer to a specific image
        Parameters
        ----------
        image_id: int
            Id of the image
        categories_ids: list
            List of categories ids
        model_name: str
            Name of the model used for retrieving the proposals

        Returns
        -------
        pandas.DataFrame
        """

        if not isinstance(image_id, int):
            logger.error(err_type.format("image_id"))
            return -1

        if not isinstance(categories_ids, list):
            logger.error(err_type.format("categories_ids"))
            return -1

        if not isinstance(model_name, str):
            logger.error(err_type.format("model_name"))
            return -1

        if not self.proposals:
            logger.error("No proposals loaded")
            return -1
        elif model_name not in self.proposals:
            logger.error(err_value.format("model_name", list(self.proposals.keys())))
            return -1

        return self.proposals[model_name].loc[(self.proposals[model_name][self.match_param_props] == image_id) &
                                  (self.proposals[model_name]["category_id"].isin(categories_ids))]

    def get_proposals_by_category_and_ids(self, category, ids, model_name):
        """
        Returns all the proposals that belong to the specified category and that have the specified ids
        Parameters
        ----------
        category: str
            Name of the category
        ids: list
            Ids of the proposals
        model_name: str
            Name of the model used for retrieving the proposals

        Returns
        -------
        padans.DataFrame
        """

        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1

        if not isinstance(ids, list):
            logger.error(err_type.format("ids"))
            return -1

        if not isinstance(model_name, str):
            logger.error(err_type.format("model_name"))
            return -1

        if not self.proposals:
            logger.error("No proposals loaded")
            return -1
        elif model_name not in self.proposals:
            logger.error(err_value.format("model_name", list(self.proposals.keys())))
            return -1

        cat_id = self.get_category_id_from_name(category)
        return self.proposals[model_name].loc[(self.proposals[model_name]["category_id"] == cat_id) &
                                              (self.proposals[model_name]["id"].isin(ids))]

    def get_proposals_ids_by_category_and_property(self, category_id, property_name, property_value, model_name):
        return self.proposals[model_name].loc[(self.proposals[model_name][property_name] == property_value) &
                                              (self.proposals[model_name]["category_id"] == category_id)]["id"].values


    def _load_proposals(self, model_name, path):
        """
        Loads the proposals into memory

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the proposals
        """
        print("Loading proposals of {} model...".format(model_name))

        try:
            counter = 0
            issues = 0
            matching_parameter_issues = 0
            proposals = []
            for i, cat in self.categories.iterrows():
                c = cat["name"]
                c_id = cat["id"]
                proposal_path = os.path.join(path, c + ".txt")
                with open(proposal_path, "r") as file:
                    for line in file:
                        try:
                            if self.task_type == TaskType.INSTANCE_SEGMENTATION:
                                arr = line.split(" ")
                                match_param, confidence = arr[0], float(arr[1])
                                if confidence < 0 or confidence > 1:
                                    raise ValueError

                                # filter applied to proposals based on confidence
                                if confidence < self._ignore_proposals_threshold:
                                    continue

                                if not self.match_on_filename:
                                    match_param = int(match_param)
                                    img_id = match_param
                                else:
                                    img_id = self.get_image_id_from_image_name(match_param)
                                segmentation = [float(v) for v in arr[2:]]

                                aspect_ratio_label = self.get_aspect_ratio_label(compute_aspect_ratio_of_segmentation(segmentation))
                                img = self.get_image_from_id(img_id)
                                encoded_mask = encode_segmentation(segmentation, img["height"], img["width"])
                                area = mask.area(encoded_mask)
                                area_size_label = self.get_area_size_label(area)

                                counter += 1
                                proposals.append(
                                    {"confidence": confidence, "segmentation": segmentation, self.match_param_props: match_param,
                                     "category_id": c_id, "id": counter,
                                     "AspectRatio": aspect_ratio_label, "AreaSize": area_size_label})
                            else:
                                match_param, confidence, x1, y1, w, h = line.split(" ")
                                if not self.match_on_filename:
                                    match_param = int(match_param)
                                confidence = float(confidence)
                                if confidence < 0 or confidence > 1:
                                    raise ValueError

                                # filter applied to proposals based on confidence
                                if confidence < self._ignore_proposals_threshold:
                                    continue

                                x1, y1, w, h = float(x1), float(y1), float(w), float(h)

                                aspect_ratio_label = self.get_aspect_ratio_label((w + 1) / (h + 1))
                                area_size_label = self.get_area_size_label(w*h)

                                counter += 1
                                proposals.append(
                                    {"confidence": confidence, "bbox": [x1, y1, w, h], self.match_param_props: match_param,
                                     "category_id": c_id, "id": counter,
                                     "AspectRatio": aspect_ratio_label, "AreaSize": area_size_label})
                        except:
                            if "." in match_param and not self.match_on_filename:
                                matching_parameter_issues += 1
                            issues += 1

            if matching_parameter_issues > 0:
                raise ValueError("It seems that, for {}, the predictions refer to the ground truth by the file_name value, but the parameter match_on_filename is set to False."
                                 " Please, try to instantiate the DatasetLocalization with 'match_on_filename=True'".format(model_name))
            if counter == 0 and issues > 0:
                raise Exception

            self.__proposals_length = counter
            logger.info("Loaded {} proposals and failed with {} for {}".format(counter, issues, model_name))

        except ValueError as e:
            raise e
        except:
            raise Exception("Error loading proposals for {} model".format(model_name))

        proposals_df = pd.DataFrame(proposals)
        if self.match_on_filename:
            match = pd.merge(self.images, proposals_df, how="right", left_on="file_name", right_on=self.match_param_props)
            match.sort_values(by="id_y", ascending=True, inplace=True)
            proposals_df[self.match_param_props] = match["id_x"].values
        print("Done!")
        return proposals_df

    # -- DATA SET -- #

    def dataset_type_name(self):
        """
        Returns the name of the dataset

        Returns
        -------
        str
        """
        return self.images_set_name

    def _reset_index_gt(self):
        self.annotations.reset_index(drop=False, inplace=True)

        if self.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            self.annotations = self.annotations.drop_duplicates("id")

        self.annotations.drop("cat_id", axis=1, inplace=True)

    def update_properties(self, properties):
        self._reset_index_gt()
        self._index_gt(properties)
        self._create_properties(properties, self.annotations)
        self.load_properties_display_names()
        if len(properties) > 0:
            self._set_analyses_with_properties_available()

    def get_all_possible_properties(self):
        possible_properties = {}
        properties_loaded = self.get_property_keys()
        if properties_loaded is not None:
            for p in properties_loaded:
                possible_properties[p] = True

        not_loaded_properties = list(set(self.annotations.columns) - self.common_properties)
        for p in not_loaded_properties:
            possible_properties[p] = False

        return possible_properties

    def _are_properties_from_file_valid(self, properties):
        for p in properties:
            if p not in self.annotations.index.names:
                return False
            if "display_name" not in properties[p] or "values" not in properties[p]:
                return False
            for value in properties[p]["values"]:
                if "value" not in value or "display_name" not in value:
                    return False
        return True

    def _check_properties_file_validity(self):
        return self._is_properties_file_valid(self.get_annotations().reset_index(), self.common_properties)

    def load(self, force_loading=False, load_properties=True):
        """
        Loads the dataset, the proposals and the properties into memory

        Parameters
        ----------
        force_loading: bool, optional
            If True reload the dataset and the proposals
        """
        self.area_size_computed = False
        self.aspect_ratio = False
        try:
            if force_loading or self.annotations is None:
                print("Loading dataset...")

                file = open(self.dataset_root_param, "r")
                data = json.load(file)
                file.close()
                self.images = pd.DataFrame(data["images"])
                self.annotations = pd.DataFrame(data["annotations"])
                self.categories = pd.DataFrame(data["categories"])

                self.__is_valid_dataset_format()

                self.compute_area_size()
                self.compute_aspect_ratio()

                print("Done!")
        except:
            raise Exception("Error loading dataset")

        if not self.for_analysis:
            self._index_gt()
            self._analyses_without_properties_available = True
            return

        if self.proposals_paths is not None:
            if force_loading or not self.proposals:
                self.load_proposals()

        if not load_properties:
            self._index_gt()
            try:
                if (not os.path.exists(self.properties_filename)) or not self._is_properties_file_valid(None, None, False):
                    self._create_properties([], None)
            except:
                self._create_properties([], None)
            self.load_categories_display_names()
            self._analyses_without_properties_available = True
            return

        if is_notebook():
            self._load_or_create_properties_notebook(self.annotations,
                                                     self.common_properties,
                                                     self._set_analyses_with_properties_available)
        else:
            self._load_or_create_properties(self.annotations,
                                            self.common_properties)
            self._set_analyses_with_properties_available()

    def _index_gt(self, properties=None):
        """
        Indexes the observations DataFrame by category and properties.
        Parameters
        ----------
        properties: list, optional
            Properties to be indexed
        """
        if properties is None:
            properties = list(self.get_property_keys())
        tmp_indexes = properties
        for p in tmp_indexes:
            if self.annotations[p].isnull().values.any():
                self.annotations[p] = self.annotations[p].fillna("no value")
                logger.warning("Some observations don't have the property {}. Default value 'no value' added".format(p))
            values = list(set(self.annotations[p].values))
            if len(values) > 10 and all(isinstance(v, Number) for v in values):
                self._get_range_of_property_values(self.annotations[p].values, p)

        self.annotations = self.annotations.assign(cat_id=self.annotations["category_id"])
        self.annotations = self.annotations.assign(img_id=self.annotations["image_id"])
        indexes = ["cat_id", "img_id"]
        indexes.extend(tmp_indexes)
        self.annotations = self.annotations.set_index(list(indexes)).sort_index()

    def __is_valid_dataset_format(self):
        """
        Check the dataset format validity

        Returns
        -------
        bool
            True if the dataset format is correct, False otherwise
        """
        # check categories format
        fields = self.categories.columns.values
        if "name" not in fields:
            raise Exception(err_categories_name_dataset)
        elif self.categories["name"].isnull().values.any():
            raise Exception(err_categories_name_dataset_few)
        if "id" not in fields:
            raise Exception(err_categories_id_dataset)
        elif self.categories["id"].isnull().values.any():
            raise Exception(err_categories_id_dataset_few)

        # check images format
        fields = self.images.columns.values
        if "id" not in fields:
            raise Exception(err_images_id_dataset)
        elif self.images["id"].isnull().values.any():
            raise Exception(err_images_id_dataset_few)
        if self.match_on_filename:
            if "file_name" not in fields:
                raise Exception(err_images_filename_dataset)
            elif self.images["file_name"].isnull().values.any():
                raise Exception(err_images_filename_dataset_few)

        # check annotations format
        fields = self.annotations.columns.values
        if "image_id" not in fields:
            raise Exception(err_annotations_image_id_dataset)
        elif self.annotations["image_id"].isnull().values.any():
            raise Exception(err_annotations_image_id_dataset_few)
        if "category_id" not in fields:
            raise Exception(err_annotations_category_id_dataset)
        elif self.annotations["category_id"].isnull().values.any():
            raise Exception(err_annotations_category_id_dataset_few)
        if self.task_type == TaskType.INSTANCE_SEGMENTATION:
            if "segmentation" not in fields:
                raise Exception(err_annotations_segmentation_dataset)
            elif self.annotations["segmentation"].isnull().values.any():
                raise Exception(err_annotations_segmentation_dataset_few)
        else:
            if "bbox" not in fields:
                raise Exception(err_annotations_bbox_dataset)
            elif self.annotations["bbox"].isnull().values.any():
                raise Exception(err_annotations_bbox_dataset_few)

    def _set_range_of_property_values(self, ranges, range_labels, property_name):
        """
        Sets the already calculated value ranges for the specified property

        Parameters
        ----------
        ranges: array-like
            Value ranges
        range_labels: array-like
            Labels of the value ranges
        property_name
            Name of the property to set the ranges to
        """
        self.annotations.sort_values(by=property_name, inplace=True)
        self.annotations.reset_index(drop=True, inplace=True)
        prev_pos = 0
        for i, pos in enumerate(ranges):
            self.annotations.loc[prev_pos:pos, property_name] = range_labels[i]
            prev_pos = pos

