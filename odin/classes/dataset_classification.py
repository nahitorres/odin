import os
from numbers import Number

import pandas as pd
import numpy as np
import json

from odin.classes import DatasetInterface, TaskType
from odin.classes.strings import *
from odin.utils import *
from odin.utils.draw_utils import pie_plot, display_co_occurrence_matrix
from odin.utils.env import is_notebook
from odin.utils.lazy_dictionary import LazyDict

logger = get_root_logger()


class DatasetClassification(DatasetInterface):

    observations = None
    masks_annotations = None

    common_properties = {'index', 'id', 'file_name', 'path', 'height', 'width', 'categories', 'category'}
    supported_types = [TaskType.CLASSIFICATION_BINARY, TaskType.CLASSIFICATION_SINGLE_LABEL,
                       TaskType.CLASSIFICATION_MULTI_LABEL]

    def __init__(self,
                 dataset_gt_param,
                 task_type,
                 proposals_paths=None,
                 observations_set_name='test',
                 observations_abs_path=None,
                 result_saving_path='./results/',
                 similar_classes=None,
                 properties_file=None,
                 for_analysis=True,
                 load_properties=True,
                 match_on_filename=False,
                 save_graphs_as_png=True):
        """
        The DatasetClassification class can be used to store the ground truth and predictions for classification models.

        Parameters
        ----------
        dataset_gt_param: str
            Path of the ground truth .json file.
        task_type: TaskType
            Problem task type. It can be: TaskType.CLASSIFICATION_BINARY, TaskType.CLASSIFICATION_SINGLE_LABEL, TaskType.CLASSIFICATION_MULTI_LABEL.
        proposals_paths: list of tuple, optional
            List of couples. Each couple contains the model name and the corresponding proposals path. (default is None)
        observations_set_name: str, optional
            Name of the data set. (default is 'test')
        observations_abs_path: str, optional
            Path of the observation directory. (default is None)
        result_saving_path: str, optional
            Path used to save results. (default is './results/')
        similar_classes: list of list, optional
            List of groups of ids of categories which are similar to each other. (default is None)
        properties_file: str, optional
            The name of the file used to store the names of and values of the properties and the names of the categories. (default is 'properties.json')
        for_analysis: bool, optional
            Indicates whether the properties and the predictions have to be loaded. If False, only the ground truth is loaded. (default is True)
        match_on_filename: bool, optional
            Indicates whether the predictions refer to the ground truth by file_name (set to True) or by id (set to False). (default is False)
        load_properties: bool, optional
            Indicates whether the properties should be loaded. (default is True)
        save_graphs_as_png: bool, optional
            Indicates whether plots should be saved as .png images. (default is True)
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

        if not isinstance(observations_set_name, str):
            raise TypeError(err_type.format("observations_set_name"))

        if observations_abs_path is not None and not isinstance(observations_abs_path, str):
            raise TypeError(err_type.format("observations_abs_path"))

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

        self.observations = None
        self.masks_annotations = None

        super().__init__(dataset_gt_param, proposals_paths, task_type, observations_set_name, observations_abs_path,
                         result_saving_path, similar_classes, properties_file, match_on_filename, save_graphs_as_png)
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
                         "DatasetClassification")
            return -1

        if self.task_type != TaskType.CLASSIFICATION_MULTI_LABEL:
            logger.error(f"Analysis not supported for task type {self.task_type}")
            return -1

        cat_ids = np.array(self.get_categories_id_from_names(self.get_categories_names()))
        n_size = len(cat_ids)
        occ_matrix = np.zeros((n_size, n_size))
        for _, row in self.get_all_observations().iterrows():
            for i in range(0, len(row["categories"])):
                id_a = np.where(cat_ids == row["categories"][i])[0][0]
                for j in range(i+1, len(row["categories"])):
                    id_b = np.where(cat_ids == row["categories"][j])[0][0]
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
        elif not self.are_valid_properties([property_name]):
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

        observations = self.get_observations_from_categories([category])
        count = observations.groupby(property_name).size()
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
        output_path = os.path.join(self.result_saving_path, f"distribution_total_{str(property_name).replace('/', '_')}_{category}"
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

        observations = self.get_all_observations()
        count = observations.groupby(property_name).size()
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
                value = len(self.get_observations_from_property_category(cat_id, property_name, pv).index)
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

    def show_distribution_of_categories(self, show=True):
        """
        It provides the distribution of the categories in the data set.
        Parameters
        ----------
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not self._analyses_without_properties_available:
            logger.error("Please select the properties first or set 'for_analysis=False' when instantiate the "
                         "DatasetClassification")
            return -1

        observations = self.observations.copy()
        count = observations.groupby(level="cat_id").size()
        sizes, display_names = [], []
        results = {}
        for name in self.get_categories_names():
            display_names.append(self.get_display_name_of_category(name))
            value = count[self.get_category_id_from_name(name)]
            sizes.append(value)
            results[name] = value
        if not show:
            return results
        title = "Distribution of categories"
        output_path = os.path.join(self.result_saving_path, "distribution_categories.png")
        pie_plot(sizes, display_names, title, output_path, self._get_save_graphs_as_png())

    # -- OBSERVATIONS -- #

    def get_observation_id_from_file_name(self, filename):
        """
        Returns the observation id from the file_name
        Parameters
        ----------
        filename: str
            Filename of the observation

        Returns
        -------
        int
            observation id
        """
        if not isinstance(filename, str):
            logger.error(err_type.format("filename"))
            return -1
        elif "file_name" not in self.observations.columns:
            logger.error("'file_name' field not present in observations")
            return -1

        obs = self.observations.loc[self.observations["file_name"] == filename]["id"].tolist()

        if len(obs) > 0:
            return obs[0]

    def get_observations_from_ids(self, ids):
        """
        Returns the observations with a specific id.
        Parameters
        ----------
        ids: list
            List of the ids of the observations to be returned

        Returns
        -------
            pandas.DataFrame
        """
        if not isinstance(ids, list):
            logger.error(err_type.format("ids"))
            return -1

        return self.observations.loc[self.observations["id"].isin(ids)].drop_duplicates("id")

    def get_observations_from_categories(self, categories):
        """
        Returns all the observations belonging to the specified categories
        Parameters
        ----------
        categories: list
            Categories names

        Returns
        -------
        pandas.DataFrame

        """
        if not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1

        categories_id = self.get_categories_id_from_names(categories)
        if self.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            return self.observations.loc[self.observations.index.get_level_values("cat_id").isin(categories_id)]\
                .drop_duplicates("id")
        return self.observations.loc[self.observations.index.get_level_values("cat_id").isin(categories_id)]

    def get_observations_from_property_category(self, category_id, property_name, property_value):
        """
        Returns all the observations belonging to a specific category and a specific property value

        Parameters
        ----------
        category_id: int
            Id of the category
        property_name: str
            Name of the property that the value belongs to
        property_value: str, Number
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
        elif property_name not in self.observations.index.names:
            logger.error(err_value.format("property_name", list(self.get_property_keys())))
            return -1

        if not isinstance(property_value, str) and not isinstance(property_value, Number):
            logger.error(err_type.format("property_value"))
            return -1

        return self.observations.loc[(self.observations.index.get_level_values("cat_id") == category_id) &
                                     (self.observations.index.get_level_values(property_name) == property_value)]

    def get_observations_from_property(self, property_name, property_value):
        """
        Returns all the observations belonging to a specific property value

        Parameters
        ----------
        property_name: str
            Name of the property that the value belongs to
        property_value: str, Number
            Property value

        Returns
        -------
        pandas.DataFrame

        """
        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        elif property_name not in self.observations.index.names:
            logger.error(err_value.format("property_name", list(self.get_property_keys())))
            return -1

        if not isinstance(property_value, str) and not isinstance(property_value, Number):
            logger.error(err_type.format("property_value"))
            return -1

        return self.observations.loc[self.observations.index.get_level_values(property_name) == property_value]\
            .drop_duplicates("id")

    def get_number_of_observations(self):
        """
        Returns the total number of observations in the dataset

        Returns
        -------
        int
        """
        if self.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            return len(self.observations["id"].unique())
        return len(self.observations.index)

    def get_all_observations(self):
        """
        Returns all the observations in the dataset

        Returns
        -------
        pandas.DataFrame

        """
        return self.observations.drop_duplicates("id")

    # -- CATEGORIES -- #

    def is_similar(self, cat1, cats2):
        """
        Checks if two categories belongs to the same similar_group
        Parameters
        ----------
        cat1: int
            Category id
        cats2: int or list
            Category id or list of categories ids

        Returns
        -------
        bool
            True if cat1 belongs to the same similar_group of a category in cats2, False otherwise
        """
        if not isinstance(cat1, int):
            logger.error(err_type.format("cat1"))
            return -1

        if self.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            if not isinstance(cats2, list):
                logger.error(err_type.format("cat2"))
                return -1
        elif not isinstance(cats2, int):
            logger.error(err_type.format("cat2"))
            return -1

        for groups in self.similar_groups:
            if self.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
                for cat2 in cats2:
                    if cat1 in groups and cat2 in groups:
                        return True
            else:
                if cat1 in groups and cats2 in groups:
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
        obs = self.get_observations_from_categories([category])
        return list(obs.index.get_level_values(property_name).unique())

    # -- PREDICTIONS -- #

    def get_proposals_from_observation_id(self, obs_id, model_name):
        """
        Returns all the proposals that refer to a specific observation

        Parameters
        ----------
        obs_id: int
            Id of the observation
        model_name: str
            Name of the model used for retrieving the proposals

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(obs_id, int):
            logger.error(err_type.format("obs_id"))
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

        return self.proposals[model_name].loc[self.proposals[model_name][self.match_param_props] == obs_id]

    def get_proposals_from_observation_id_and_categories(self, obs_id, categories_ids, model_name):
        """
        Returns all the proposals of specific categories that refer to a specific observation
        Parameters
        ----------
        obs_id: int
            Id of the observation
        categories_ids: list
            List of categories ids
        model_name: str
            Name of the model used for retrieving the proposals

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(obs_id, int):
            logger.error(err_type.format("obs_id"))
            return -1

        if not isinstance(categories_ids, list) or not all(isinstance(v, int) for v in categories_ids):
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

        return self.proposals[model_name].loc[(self.proposals[model_name][self.match_param_props] == obs_id) &
                              (self.proposals[model_name]["category_id"].isin(categories_ids))]

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
                c = cat['name']
                c_id = cat['id']
                proposal_path = os.path.join(path, c + ".txt")
                if not os.path.exists(proposal_path):
                    logger.warning("Proposals file for category {} not found for {}".format(c, model_name))
                    continue
                with open(proposal_path, "r") as file:
                    for line in file:
                        try:
                            match_param, confidence = line.split(" ")

                            if not self.match_on_filename:
                                match_param = int(match_param)
                            confidence = float(confidence)
                            if confidence < 0 or confidence > 1:
                                raise ValueError
                            counter += 1
                            proposals.append(
                                {"id": counter, self.match_param_props: match_param, "confidence": confidence,
                                 "category_id": c_id}
                            )
                        except:
                            if "." in match_param and not self.match_on_filename:
                                matching_parameter_issues += 1
                            issues += 1
            if matching_parameter_issues > 0:
                raise ValueError("It seems that, for {}, the predictions refer to the ground truth by the file_name value, but the parameter match_on_filename is set to False."
                                 " Please, try to instantiate the DatasetClassification with 'match_on_filename=True'".format(model_name))
            if counter == 0:
                raise Exception

            logger.info("Loaded {} proposals and failed with {} for {}".format(counter, issues, model_name))
        except ValueError as e:
            raise e
        except:
            raise Exception("Error loading proposals for {} model".format(model_name))
        proposals_df = pd.DataFrame(proposals)
        if self.match_on_filename:
            match = pd.merge(self.get_all_observations(), proposals_df, how="right", left_on="file_name",
                             right_on=self.match_param_props)
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
        self.observations.reset_index(drop=False, inplace=True)

        if self.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            self.observations = self.observations.drop_duplicates("id")

        self.observations.drop("cat_id", axis=1, inplace=True)

    def update_properties(self, properties):
        self._reset_index_gt()
        self._index_gt(properties)
        self._create_properties(properties, self.observations)
        self.load_properties_display_names()
        if len(properties) > 0:
            self._set_analyses_with_properties_available()

    def get_all_possible_properties(self):
        possible_properties = {}
        properties_loaded = self.get_property_keys()
        if properties_loaded is not None:
            for p in properties_loaded:
                possible_properties[p] = True

        not_loaded_properties = list(set(self.observations.columns) - self.common_properties)
        for p in not_loaded_properties:
            possible_properties[p] = False

        return possible_properties

    def _are_properties_from_file_valid(self, properties):
        for p in properties:
            if p not in self.observations.index.names:
                return False
            if "display_name" not in properties[p] or "values" not in properties[p]:
                return False
            for value in properties[p]["values"]:
                if "value" not in value or "display_name" not in value:
                    return False
        return True

    def _check_properties_file_validity(self):
        return self._is_properties_file_valid(self.get_all_observations().reset_index(), self.common_properties)

    def load(self, force_loading=False, load_properties=True):
        """
        Loads the dataset, the proposals and the properties into memory

        Parameters
        ----------
        force_loading: bool, optional
            If True reload the dataset and the proposals
        """
        try:
            if force_loading or self.observations is None:
                print("Loading dataset...")

                file = open(self.dataset_root_param, "r")
                data = json.load(file)
                file.close()
                self.observations = pd.DataFrame(data['observations'])
                self.categories = pd.DataFrame(data["categories"])

                self._is_valid_dataset_format()

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
                if not os.path.exists(self.properties_filename) or not self._is_properties_file_valid(None, None, False):
                    self._create_properties([], None)
            except:
                self._create_properties([], None)
            self.load_categories_display_names()
            self._analyses_without_properties_available = True
            return

        if is_notebook():
            self._load_or_create_properties_notebook(self.observations,
                                                     self.common_properties,
                                                     self._set_analyses_with_properties_available)
        else:
            self._load_or_create_properties(self.observations,
                                            self.common_properties)
            self._set_analyses_with_properties_available()

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
        self.observations.sort_values(by=property_name, inplace=True)
        self.observations.reset_index(drop=True, inplace=True)
        prev_pos = 0
        for i, pos in enumerate(ranges):
            self.observations.loc[prev_pos:pos, property_name] = range_labels[i]
            prev_pos = pos

    def _is_valid_dataset_format(self):
        """
        Checks the dataset format validity

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

        # check observations format
        fields = self.observations.columns.values
        if "id" not in fields:
            raise Exception(err_observations_id_dataset)
        elif self.observations["id"].isnull().values.any():
            raise Exception(err_observations_id_dataset_few)
        if self.match_on_filename:
            if "file_name" not in fields:
                raise Exception(err_observations_filename_dataset)
            elif self.observations["file_name"].isnull().values.any():
                raise Exception(err_observations_filename_dataset_few)
        if self.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            if "categories" not in fields:
                raise Exception(err_observations_categories_dataset)
            elif self.observations["categories"].isnull().values.any():
                raise Exception(err_observations_categories_dataset_few)
        else:
            if "category" not in fields:
                raise Exception(err_observations_category_dataset)
            elif self.observations["category"].isnull().values.any():
                raise Exception(err_observations_category_dataset_few)

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
            if self.observations[p].isnull().values.any():
                self.observations[p] = self.observations[p].fillna("no value")
                logger.warning("Some observations don't have the property {}. Default value 'no value' added".format(p))
            values = list(set(self.observations[p].values))
            if len(values) > 10 and all(isinstance(v, Number) for v in values):
                self._get_range_of_property_values(values, p)

        if self.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            self.observations = self.observations.assign(cat_id=self.observations["categories"]).explode("cat_id")
        else:
            self.observations = self.observations.assign(cat_id=self.observations["category"])
        indexes = ["cat_id"]
        indexes.extend(tmp_indexes)
        self.observations = self.observations.set_index(list(indexes)).sort_index()
