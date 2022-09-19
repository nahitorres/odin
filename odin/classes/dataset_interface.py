import abc
import math
import os
from numbers import Number

import pandas as pd
import numpy as np
import json

from IPython.display import display
from ipywidgets import Checkbox, VBox, Button, HBox, Label, Text

from odin.classes.strings import *
from odin.utils import get_root_logger
from odin.utils.env import is_notebook
from odin.utils.lazy_dictionary import LazyDict

logger = get_root_logger()


class DatasetInterface(metaclass=abc.ABCMeta):
    dataset_root_param = ''
    images_abs_path = ''

    images_set_name = 'test'

    categories = None
    proposals = {}

    properties_filename = None

    match_param_gt = "id"
    match_param_props = "on_id"

    # match_on_filename = False

    similar_groups = []
    __properties = {}
    __categories_display_names = {}

    __SAVE_PNG_GRAPHS = True

    def __init__(self,
                 dataset_gt_param,
                 proposals_paths,
                 task_type,
                 images_set_name,
                 images_abs_path,
                 result_saving_path,
                 similar_classes,
                 properties_file,
                 match_on_filename,
                 save_graphs_as_png):

        self.dataset_root_param = dataset_gt_param
        self.proposals_paths = proposals_paths
        self.task_type = task_type
        self.images_set_name = images_set_name
        self.images_abs_path = images_abs_path
        self.match_on_filename = match_on_filename
        self.__SAVE_PNG_GRAPHS = save_graphs_as_png
        self.result_saving_path = os.path.join(result_saving_path, images_set_name)

        self._possible_properties = []

        self.proposals = {}
        self.categories = None

        if save_graphs_as_png:
            if not os.path.exists(result_saving_path):
                os.mkdir(result_saving_path)
            if not os.path.exists(self.result_saving_path):
                os.mkdir(self.result_saving_path)

        if similar_classes is not None:
            self.similar_groups = similar_classes
        self.properties_filename = "properties.json" if (properties_file is None or os.path.abspath(properties_file) == os.path.abspath(dataset_gt_param)) else properties_file

        self.match_param_gt = "id"
        self.match_param_props = "gt_id"

        self._analyses_without_properties_available = False
        self._analyses_with_properties_available = False

        self.__properties = {}
        self.__categories_display_names = {}

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

    # -- DISTRIBUTION ANALYSES -- #

    def show_distribution_of_properties(self, properties=None, show=True):
        """
        For each property, it provides the distribution of its different values and for each value shows the distribution of the categories.

        Parameters
        ----------
        properties: list, optional
            List of properties to be included in the analysis. If not specified, all the properties are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not self._analyses_with_properties_available:
            if not self._analyses_without_properties_available:
                logger.error("Please complete the properties selection first")
            else:
                logger.error("No properties available. Please make sure to load the properties")
            return -1

        if properties is None:
            properties = self.get_property_keys()
        elif not isinstance(properties, list):
            logger.error(err_type.format("properties"))
            return -1
        elif not self.are_valid_properties(properties):
            return -1
        results = {}
        for p in properties:
            results[p] = self.show_distribution_of_property(p, show=show)

        if not show:
            return results

    def show_distribution_of_property_for_categories(self, property_name, property_values=None, categories=None, show=True):
        """
        For each category, it provides the distribution of a property.
        Parameters
        ----------
        property_name: str
            Name of the property to be analyzed.
        property_values: list, optional
            List of the property values to be included in the analysis. If not specified, all the values are included. (default is None)
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
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

        if categories is None:
            categories = self.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self.are_valid_categories(categories):
            return -1

        results = {}
        for c in categories:
            results[c] = self.show_distribution_of_property_for_category(property_name, c, property_values=property_values, show=show)

        if not show:
            return results

    # -- CATEGORIES -- #

    def load_categories_display_names(self):
        """Loads into memory the names of the categories to display in the plots.
        The names are retrieved from the properties file (default is 'properties.json')
        """
        try:
            file = open(self.properties_filename, "r")
            self.__categories_display_names = json.load(file)["categories"]
            file.close()
        except:
            logger.error(err_categories_file)

    def get_display_name_of_category(self, category):
        """Returns the display name of a specific category

        Parameters
        ----------
        category: str
            Category name used to obtain the name to display

        Returns
        -------
        str or None
            Name to display of the category or None if not found

        """
        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1

        if category not in self.__categories_display_names.keys():
            return None

        return self.__categories_display_names[category]["display_name"]

    def get_display_name_of_categories(self):
        """Returns the display name of all the categories

        Returns
        -------
        dict
            dict of the categories with the corresponding names to display
        """
        return self.__categories_display_names

    def get_category_name_from_id(self, category_id):
        """Returns the category name from its id

        Parameters
        ----------
        category_id: int
            Id of the category to retrieve

        Returns
        -------
        str or None
            Name of the category or None if not found

        """
        if not isinstance(category_id, int):
            logger.error(err_type.format("category_id"))
            return -1

        names = self.categories[self.categories["id"] == category_id]["name"].tolist()
        if len(names) > 0:
            return names[0]

    def is_valid_category(self, category_name):
        """Check if the category exists in the database

        Parameters
        ----------
        category_name: str
            Name of the category

        Returns
        -------
        bool
            True if the category exists, otherwise False
        """
        if not isinstance(category_name, str):
            logger.error(err_type.format("category_name"))
            return False

        return category_name in self.categories["name"].tolist()

    def get_categories_names(self):
        """Returns the names of all the categories

        Returns
        -------
        array of str of shape (n_categories_names,)
        """
        return list(self.categories["name"])

    def get_categories_names_from_ids(self, ids):
        """Returns the names of the categories with a specific id

        Parameters
        ----------
        ids: list
            Ids of the categories to retrieve

        Returns
        -------
        array of str of shape (n_categories_names,)
            Names of the categories
        """
        if not isinstance(ids, list) or not all(isinstance(v, int) for v in ids):
            logger.error(err_type.format("ids"))
            return -1

        return [self.get_category_name_from_id(id) for id in ids]

    def get_category_id_from_name(self, category):
        """Returns the category id from the category name

        Parameters
        ----------
        category: str
            Category name used to retrieve the id

        Returns
        -------
        int or None
            Id of the category or None if not found
        """
        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1

        cat = self.categories[self.categories["name"] == category]["id"].tolist()
        if len(cat) > 0:
            return cat[0]


    def get_categories_id_from_names(self, categories):
        """Returns the ids from the categories names

        Parameters
        ----------
        categories: list
            Names of the categories used to retrieve the ids

        Returns
        -------
        list
            Ids of the categories
        """
        if not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1

        return [self.get_category_id_from_name(name) for name in categories]

    def are_valid_categories(self, categories):
        """
        Check if the categories are valid
        Parameters
        ----------
        categories: list
            Categories names to be checked

        Returns
        -------
        bool
            True if the categories names are valid, otherwise False
        """
        if not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return False
        elif len(categories) == 0:
            logger.error(f"Empty categories list.")
            return False

        for c in categories:
            if c not in self.get_categories_names():
                logger.error("Category '{}' not valid. Possible values: {}".format(c, self.get_categories_names()))
                return False

        return True

    # -- META-ANNOTATIONS -- #

    def get_property_keys(self):
        """Returns the names of the properties

        Returns
        -------
        dict_keys
        """
        return self.__properties.keys()

    def is_possible_property(self, property_name):
        """
        Indicates whether the property could be a valid property for the evaluation.

        Parameters
        ----------
        property_name: str
            Name of the property to be verified.

        Returns
        -------
            bool
        """
        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return False
        return property_name in self._possible_properties

    def get_values_for_property(self, pkey):
        """Returns all the values of a specific property

        Parameters
        ----------
        pkey: str
            Name of the property

        Returns
        -------
        list
            List of all the values of the property
        """
        if not isinstance(pkey, str):
            logger.error(err_type.format("pkey"))
            return -1

        if pkey in self.__properties.keys():
            return [v["value"] for v in self.__properties[pkey]["values"]]
        return []

    def get_display_name_of_property(self, pkey):
        """Returns the name of the property to display

        Parameters
        ----------
        pkey: str
            Name of the property

        Returns
        -------
        str or None
            Name of the property to display or None if not found
        """
        if not isinstance(pkey, str):
            logger.error(err_type.format("pkey"))
            return -1

        if pkey not in self.__properties.keys():
            return None

        return self.__properties[pkey]["display_name"]

    def get_display_name_of_property_value(self, property, property_value):
        """Returns the name to display of a specific property value

        Parameters
        ----------
        property: str
            Name of the property containing the value
        property_value: str or Number
            Name of the property value

        Returns
        -------
        str or None
            Name to display of the property value or None if not found
        """
        if not isinstance(property, str):
            logger.error(err_type.format("property"))
            return -1
        if not isinstance(property_value, str) and not isinstance(property_value, Number):
            logger.error(err_type.format("property_value"))
            return -1

        if property in self.__properties.keys():
            values = self.__properties[property]["values"]
            for value in values:
                if value["value"] == property_value:
                    return value["display_name"]
        return None

    def are_valid_properties(self, properties):
        """
        Check if the properties are valid
        Parameters
        ----------
        properties: list
            Properties names to be checked

        Returns
        -------
        bool
            True if the properties names are valid, otherwise False
        """
        if not isinstance(properties, list):
            logger.error(err_type.format("properties"))
            return False
        elif len(properties) == 0:
            logger.error(f"Empty properties list.")
            return False

        for p in properties:
            if p not in self.get_property_keys():
                if self.is_possible_property(p):
                    logger.error(err_property_not_loaded.format(p))
                    return False
                logger.error(err_value.format("property", list(self.get_property_keys())))
                return False

        return True

    def is_valid_property(self, property_name, possible_values):
        """
        Check if the properties and the corresponding values are valid
        Parameters
        ----------
        property_name: str
            Name of the property
        possible_values: list
            List of the possible values
        Returns
        -------
        bool
            True if the property and the corresponding values are valid, otherwise False
        """
        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return False

        if not isinstance(possible_values, list):
            logger.error(err_type.format("possible_values"))
            return False

        if property_name in self.get_property_keys():
            if len(possible_values) == 0:
                logger.error(f"Empty possible values list")
                return False
            for value in possible_values:
                if value not in self.get_values_for_property(property_name):
                    logger.error(f"Property value '{value}' not valid for property '{property_name}'")
                    return False
        else:
            logger.error("Property '{}' not valid. Possible values: {}".format(property_name, list(self.get_property_keys())))
            return False

        return True

    def reload_properties(self, from_file=True, ignore_file_validity=False):
        """
        Reloads the properties into memory
        Parameters
        ----------
        from_file: bool, optional
            Indicates whether to reload the properties from the '[properties].json' file directly or from the notebook selection. (default is True)
        ignore_file_validity: bool, optional
            Indicates whether the properties file format validity should not be checked. (default is False)
        """
        if from_file:
            if not ignore_file_validity and not self._check_properties_file_validity():
                logger.error("Invalid properties file format. Please modify the properties file or "
                             "set 'ignore_file_validity=True' to select again the meta-annotations")
                return -1
            self._reset_index_gt()
            self.load()
        elif not is_notebook():
            logger.error("Please set 'from_file=True' or run ODIN from Jupyter Notebook")
            return -1
        else:
            self._load_new_properties_notebook(self.get_all_possible_properties())

    @abc.abstractmethod
    def get_all_possible_properties(self):
        pass

    @abc.abstractmethod
    def _reset_index_gt(self):
        pass

    @abc.abstractmethod
    def update_properties(self, properties):
        pass

    def _load_new_properties_notebook(self, properties):
        def on_select_all(b):
            for ckb in chkboxes:
                ckb.value = True

        def on_deselect_all(b):
            for ckb in chkboxes:
                ckb.value = False

        def on_load_all_clicked(b):
            on_select_all(None)
            on_load_clicked(None)

        def on_load_clicked(b):
            properties = set([p.description for p in chkboxes if p.value])
            if not properties:
                return
            self.update_properties(properties)
            for b in buttons:
                b.disabled = True
            for b in chkboxes:
                b.disabled = True


        chkboxes = [Checkbox(properties[p], description=p) for p in properties]
        ui_properties = VBox(chkboxes)

        select_all_button = Button(description="select all")
        deselect_all_button = Button(description="deselect all")
        select_all_button.on_click(on_select_all)
        deselect_all_button.on_click(on_deselect_all)
        ui_selection_buttons = HBox([select_all_button, deselect_all_button])

        load_selected_button = Button(description='load')
        load_all_button = Button(description='load all')
        load_selected_button.on_click(on_load_clicked)
        load_all_button.on_click(on_load_all_clicked)
        ui_buttons = HBox([load_all_button, load_selected_button])

        buttons = [select_all_button, deselect_all_button, load_all_button, load_selected_button]

        display(VBox([ui_selection_buttons, ui_properties, ui_buttons]))

    def load_properties_display_names(self, check_validity=True):
        """Loads into memory the properties and their values.
        The properties are retrieved from the properties file (default is 'properties.json')
        """
        try:
            print("Loading properties...")

            file = open(self.properties_filename, "r")
            properties = json.load(file)["properties"]
            file.close()

            if check_validity:
                if not self._are_properties_from_file_valid(properties):
                    raise Exception

            self.__properties = properties

            print("Done!")
        except :
            logger.error(err_properties_file)

    @abc.abstractmethod
    def _are_properties_from_file_valid(self, properties):
        pass

    def _load_or_create_properties(self, properties_dataset, common_properties):
        """Method to create and/or load the 'properties.json' (default name) file if ODIN is executed from CLI

        Parameters
        ----------
        properties_dataset: pandas.DataFrame
            DataFrame containing the properties. For DatasetLocalization it represents the annotations,
            for DatasetClassification it represents the observations
        common_properties: set
            Represents the common properties to not consider
        """
        props = set(properties_dataset.columns)
        self._possible_properties = list(props - common_properties)

        if (not os.path.exists(self.properties_filename)) or (not self._is_properties_file_valid(properties_dataset,
                                                                                                 common_properties)):
            all_properties = props - common_properties
            properties = self.__select_properties_terminal(all_properties)
            self._index_gt(properties)
            self._create_properties(properties, properties_dataset)
            self.load_properties_display_names()
            self.load_categories_display_names()
        else:
            self.load_properties_display_names(check_validity=False)
            self.load_categories_display_names()
            self._index_gt()

    def _load_or_create_properties_notebook(self, properties_dataset, common_properties, optional_function=None):
        """Method to create and/or load the 'properties.json' (default name) file if ODIN is executed from Notebook

        Parameters
        ----------
        properties_dataset: pandas.DataFrame
            DataFrame containing the properties. For DatasetLocalization it represents the annotations,
            for DatasetClassification it represents the observations
        common_properties: set
            Represents the common properties to not consider
        """
        def on_load_selected_clicked(b):
            properties = set([p.description for p in checkboxes if p.value])
            if not properties:
                return
            self._index_gt(properties)
            self._create_properties(properties, properties_dataset)
            self.load_properties_display_names()
            self.load_categories_display_names()
            for b in buttons:
                b.disabled = True
            for b in checkboxes:
                b.disabled = True
            if optional_function is not None:
                optional_function()

        def on_load_all_clicked(b):
            on_select_all(None)
            on_load_selected_clicked(None)

        def on_select_all(b):
            for ckb in checkboxes:
                ckb.value = True

        def on_deselect_all(b):
            for ckb in checkboxes:
                ckb.value = False

        props = set(properties_dataset.columns)
        self._possible_properties = list(props - common_properties)
        if (not os.path.exists(self.properties_filename)) or (not self._is_properties_file_valid(properties_dataset,
                                                                                                 common_properties)):
            all_properties = props - common_properties
            if len(all_properties) == 0:
                self._create_properties({}, properties_dataset)
                self.load_properties_display_names()
                self.load_categories_display_names()
            else:
                checkboxes = [Checkbox(False, description=p) for p in all_properties]
                ui_boxes = VBox(checkboxes)

                select_all_button = Button(description="select all")
                deselect_all_button = Button(description="deselect all")
                select_all_button.on_click(on_select_all)
                deselect_all_button.on_click(on_deselect_all)
                ui_selection_buttons = HBox([select_all_button, deselect_all_button])

                load_selected_button = Button(description='load')
                load_all_button = Button(description='load all')
                load_selected_button.on_click(on_load_selected_clicked)
                load_all_button.on_click(on_load_all_clicked)
                ui_buttons = HBox([load_all_button, load_selected_button])

                buttons = [select_all_button, deselect_all_button, load_selected_button, load_all_button]

                print("Select at least one property to load:")
                display(ui_selection_buttons, ui_boxes, ui_buttons)

        else:
            self.load_properties_display_names(check_validity=False)
            self.load_categories_display_names()
            self._index_gt()
            if optional_function is not None:
                optional_function()

    def _get_range_of_property_values(self, values, property_name):
        """Creates ranges of values for properties with a large number of different numerical values

        Parameters
        ----------
        values: array-like
            All different numerical values
        property_name: str
            Name of the property
        """
        sorted_values = np.sort(values)

        labels = ["XL", "L", "M", "H", "XH"]
        var_values = np.asarray([1 / 10, 3 / 10, 7 / 10, 9 / 10, 1], dtype=np.float) * len(sorted_values)
        positions = np.around(var_values).astype(np.int)
        self._set_range_of_property_values(positions, labels, property_name)

    def _create_properties(self, properties, properties_dataset):
        """Method to create the 'properties.json' (default name) file

        Parameters
        ----------
        properties: set
            Properties to include
        properties_dataset: pandas.Dataframe
            All possible properties

        """
        print("Creating properties file...")
        if properties_dataset is not None:
            properties_dataset = properties_dataset.reset_index()
        p_values = {}
        values = {}
        for p in properties:
            p_values[p] = list(set(properties_dataset[p].values.tolist()))
            values_array = []
            for value in p_values[p]:
                display_name = self.get_display_name_of_property_value(p, value)
                if display_name is None:
                    display_name = value
                values_array.append({"value": value, "display_name": display_name})
            display_name = self.get_display_name_of_property(p)
            if display_name is None:
                display_name = p
            values[p] = {"values": values_array, "display_name": display_name}
        categories = {}
        cats_names = self.get_categories_names()
        for cat_name in cats_names:
            display_name = self.get_display_name_of_category(cat_name)
            if display_name is None:
                display_name = cat_name
            categories[cat_name] = {"display_name": display_name}

        file = open(self.properties_filename, "w")
        json.dump({"properties": values, "categories": categories}, file, indent=4)
        file.close()

        print("Done!")

    def __select_properties_terminal(self, all_properties):
        """Method that allows the user to select the properties to load

        Parameters
        ----------
        all_properties: set
            All available properties names

        Returns
        -------
        set
            Properties selected
        """
        if len(all_properties) == 0:
            return {}
        all_properties_np = np.array(list(all_properties))
        text_selection = ""
        for i, p in enumerate(all_properties_np):
            text_selection = text_selection + f"[{i + 1}] - {p}\n"
        indexes_selectable = np.array(range(1, all_properties_np.size + 1))
        while True:
            user_input = input(f"Found {all_properties_np.size} properties:\n{text_selection}"
                               f"Select properties to load.\nDefault {indexes_selectable}: ")
            try:
                properties_selected = np.unique(list(map(int, user_input.split())))
            except:
                print("\nPlease select the properties from the list below.")
                continue
            indexes_selected = []
            if properties_selected.size == 0:
                indexes_selected = np.array(indexes_selectable) - 1
            else:
                try:
                    for p_index in properties_selected:
                        if p_index in indexes_selectable:
                            indexes_selected.append(p_index - 1)
                        else:
                            raise IndexError()
                except IndexError:
                    print("\nPlease select the properties from the list below.")
                    continue

            user_input = input(f"Are you sure you want to load these properties: \n"
                               f"{all_properties_np[indexes_selected]}? [Y/n]: ")
            if not user_input or user_input.lower() == 'y':
                properties = set(all_properties_np[indexes_selected])
                return properties

    def _is_properties_file_valid(self, properties_dataset, common_properties, check_properties=True):
        """Check if the values in the existing 'properties.json' file (default filename) are valid

        Parameters
        ----------
        properties_dataset: pandas.DatFrame
            DataFrame containing the properties as columns
        common_properties: set
            Common properties to exclude

        Returns
        -------
        bool
            True if the file is valid, otherwise False
        """
        try:
            file = open(self.properties_filename, "r")
            file_data = json.load(file)
            file.close()

            # check categories validity
            file_categories = file_data["categories"]
            for cat in file_categories.keys():
                if cat not in self.get_categories_names():
                    return False

            if not check_properties:
                return True

            file_properties = file_data["properties"]
            props = set(properties_dataset.columns)
            all_properties = props - common_properties

            if len(file_properties) == 0 and len(all_properties) > 0:
                return False

            # check first only properties validity
            for p in file_properties.keys():
                if p not in all_properties:
                    return False

            # load all possible properties values
            properties_values = {}

            for p in all_properties:
                properties_values[p] = list(set(properties_dataset[p].values.tolist()))
                if len(properties_values[p]) > 10 and all(isinstance(v, Number) for v in properties_values[p]):
                    properties_values[p] = ["XL", "L", "M", "H", "XH"]

            for property in file_properties.keys():
                for value in file_properties[property]['values']:
                    if value['value'] not in properties_values[property]:
                        if value['value'] == "no value" and any(math.isnan(x) if isinstance(x, float) else False for x in properties_values[property]):
                            continue
                        return False

            return True
        except:
            return False

    # -- PREDICTIONS -- #

    def load_proposals(self):
        tmp_dict = {}
        for model_name, path in self.proposals_paths:
            tmp_dict[model_name] = (self._load_proposals, model_name, path)
        self.proposals = LazyDict(tmp_dict)

    def get_proposals(self, model_name):
        """Returns all the proposals

        Parameters
        ----------
        model_name: str
            Name of the model used for retrieving the proposals

        Returns
        -------
        pandas.DataFrame
            DataFrame containing all the proposals
        """

        if not self.proposals:
            logger.error("No proposals loaded")
            return -1
        elif model_name not in self.proposals:
            logger.error(err_value.format("model_name", list(self.proposals.keys())))
            return -1

        return self.proposals[model_name]

    def get_proposals_of_category(self, category, model_name):
        """Returns all the proposals of a specific category

        Parameters
        ----------
        category: str
            Category name used to retrieve the proposals
        model_name: str
            Name of the model used for retrieving the proposals

        Returns
        -------
        pandas.DataFrame
            DataFrame containing all the proposals of the category specified
        """

        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1

        if not self.proposals:
            logger.error("No proposals loaded")
            return -1
        elif model_name not in self.proposals:
            logger.error(err_value.format("model_name", list(self.proposals.keys())))
            return -1

        cat_id = self.get_category_id_from_name(category)
        return self.proposals[model_name][self.proposals[model_name]["category_id"] == cat_id]

    def are_analyses_without_properties_available(self):
        return self._analyses_without_properties_available

    def are_analyses_with_properties_available(self):
        return self._analyses_with_properties_available

    def _set_analyses_with_properties_available(self):
        self._analyses_with_properties_available = True
        self._analyses_without_properties_available = True

    def _get_save_graphs_as_png(self):
        """
        Returns a bool indicating if the graphs have to be saved
        Returns
        -------
        bool
        """
        return self.__SAVE_PNG_GRAPHS

    def _index_gt(self, properties=None):
        pass

    @property
    @abc.abstractmethod
    def dataset_type_name(self):
        pass

    @abc.abstractmethod
    def show_distribution_of_property_for_category(self, property_name, category, property_values=None, show=True):
        pass

    @abc.abstractmethod
    def show_distribution_of_property(self, property_name, property_values=None, show=True):
        pass

    @abc.abstractmethod
    def show_distribution_of_categories(self):
        pass

    @abc.abstractmethod
    def get_property_values_for_category(self, property_name, category):
        pass

    @abc.abstractmethod
    def _load_proposals(self, model_name, path):
        pass

    @abc.abstractmethod
    def _set_range_of_property_values(self, ranges, range_labels, property_name):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load') and
                callable(subclass.load) or
                NotImplemented)

    @abc.abstractmethod
    def _check_properties_file_validity(self):
        pass

    @abc.abstractmethod
    def load(self):
        """Method to load dataset into memory"""
        raise NotImplementedError
