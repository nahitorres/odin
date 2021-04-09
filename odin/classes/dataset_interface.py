import abc

import os
import numpy as np
import json
from IPython.display import display
from ipywidgets import Checkbox, VBox, Button, HBox
from collections import defaultdict

from odin.utils import get_root_logger

logger = get_root_logger()


class DatasetInterface(metaclass=abc.ABCMeta):
    dataset_root_param = ''
    images_abs_path = ''

    images_set_name = 'test'

    coco_lib = None

    categories = None
    proposals = None

    properties_filename = None

    analysis_available = False

    match_param_gt = "id"
    match_param_props = "on_id"

    match_on_filename = False

    similar_groups = []
    __properties = {}

    def __init__(self, dataset_gt_param, proposal_path, images_set_name='test', images_abs_path=None,
                 similar_classes=None, property_names=None, terminal_env=False, properties_file=None, match_on_filename=False):
        self.dataset_root_param = dataset_gt_param
        self.proposal_path = proposal_path
        self.images_set_name = images_set_name
        self.images_abs_path = images_abs_path
        self.proposals_length = 0
        self.terminal_env = terminal_env
        self.property_names = property_names
        self.match_on_filename = match_on_filename
        if similar_classes is not None:
            self.similar_groups = similar_classes
        if properties_file is None:
            self.properties_filename = "properties.json"
        else:
            self.properties_filename = properties_file

        if not self.match_on_filename:
            self.match_param_gt = "id"
            self.match_param_props = "on_id"
        else:
            self.match_param_gt = "file_name"
            self.match_param_props = "on_file_name"

    def _select_properties_terminal(self, all_properties):
        if len(all_properties) == 0:
            return set([])
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

    def create_properties(self, properties, properties_dataset):
        p_values = defaultdict(list)
        for i, data in properties_dataset.iterrows():
            properties = properties.intersection(data.keys())
            for p in properties:
                p_values[p].append(data[p])
        values = {}
        for p in properties:
            p_values[p] = set(p_values[p])
            if len(p_values[p]) < 10:
                values_array = []
                for value in p_values[p]:
                    try:
                        value = value.item()
                    except AttributeError:
                        pass
                    values_array.append({"value": value, "display_name": value})
                values[p] = {"values": values_array, "display_name": p}
            else:
                logger.warn(f"Too many values for property '{p}'")
        json.dump(values, open(self.properties_filename, "w"), indent=4)

    def load_properties(self):
        self.__properties = json.load(open(self.properties_filename, "r"))

    def load_or_create_properties(self, properties_dataset, common_properties):
        if (not os.path.exists(self.properties_filename)) or (not self.is_properties_file_valid(properties_dataset,
                                                                                                common_properties)):
            if self.property_names is None:
                all_properties = set(properties_dataset.columns.values) - common_properties
                properties = self._select_properties_terminal(all_properties)
            else:
                properties = set(self.property_names)
                all_properties = set(properties_dataset.columns.values)
                for p in properties:
                    if p not in all_properties:
                        logger.warn(f"Property '{p}' not found in the dataset")
            self.create_properties(properties, properties_dataset)
        self.load_properties()

    def load_or_create_properties_notebook(self, properties_dataset, common_properties):
        def on_load_selected_clicked(b):
            properties = set([p.description for p in checkboxes if p.value])
            if not properties:
                return
            self.create_properties(properties, properties_dataset)
            self.load_properties()
            load_selected_button.disabled = True
            load_all_button.disabled = True
            print('Properties successfully loaded!')
            for b in checkboxes:
                b.disabled = True

        def on_load_all_clicked(b):
            properties = set(all_properties)
            self.create_properties(properties, properties_dataset)
            self.load_properties()
            load_selected_button.disabled = True
            load_all_button.disabled = True
            print("Properties successfully loaded!")
            for b in checkboxes:
                b.value = True
                b.disabled = True

        if (not os.path.exists(self.properties_filename)) or (not self.is_properties_file_valid(properties_dataset,
                                                                                                common_properties)):
            all_properties = set(properties_dataset.columns.values) - common_properties
            if len(all_properties) == 0:
                self.create_properties(set([]), properties_dataset)
                self.load_properties()
            else:
                checkboxes = [Checkbox(False, description=p) for p in all_properties]
                ui_boxes = VBox(checkboxes)
                print("Select at least one property to load:")
                display(ui_boxes)
                load_selected_button = Button(description='load')
                load_all_button = Button(description='load all')
                ui_buttons = HBox([load_all_button, load_selected_button])
                display(ui_buttons)
                load_selected_button.on_click(on_load_selected_clicked)
                load_all_button.on_click(on_load_all_clicked)
        else:
            self.load_properties()

    def is_properties_file_valid(self, properties_dataset, common_properties):
        file_properties = json.load(open(self.properties_filename, "r"))
        all_properties = set(properties_dataset.columns.values) - common_properties

        if len(file_properties) == 0 and len(all_properties) > 0:
            return False

        # check first only properties validity
        for p in file_properties.keys():
            if p not in all_properties:
                return False

        # load all possible properties values
        properties_values = defaultdict(list)
        for i, data in properties_dataset.iterrows():
            for property in data.keys():
                if data[property] not in properties_values[property]:
                    properties_values[property].append(data[property])

        for property in file_properties.keys():
            for value in file_properties[property]['values']:
                if value['value'] not in properties_values[property]:
                    return False

        return True

    def __load_proposals(self):
        pass

    def get_category_name_from_id(self, category_id):
        return self.categories[self.categories["id"] == category_id]["name"].values[0]

    def is_valid_category(self, category_name):
        return category_name in self.categories["name"].values

    def get_categories_names(self):
        return self.categories["name"].values

    def get_categories_names_from_ids(self, ids):
        return self.categories[self.categories["id"].isin(ids)]["name"].values

    def get_category_id_from_name(self, category):

        cat = self.categories[self.categories["name"] == category]["id"]
        return cat.values[0]

    def get_categories_id_from_names(self, categories):
        return self.categories[self.categories["name"].isin(categories)]["id"].values

    def get_property_keys(self):
        return self.__properties.keys()

    def get_values_for_property(self, pkey):
        if pkey in self.__properties.keys():
            return [v["value"] for v in self.__properties[pkey]["values"]]
        return []

    def get_display_name_of_property(self, pkey):
        if pkey in self.__properties.keys():
            return self.__properties[pkey]["display_name"]
        else:
            return None

    def get_proposals_of_category(self, category):
        if self.proposals is None:
            self.__load_proposals()
        cat_id = self.get_category_id_from_name(category)
        return self.proposals[self.proposals["category_id"] == cat_id]

    def get_proposals(self):
        if self.proposals is None:
            self.__load_proposals()
        return self.proposals

    def get_display_name_of_property_value(self, property, property_value):
        if property in self.__properties.keys():
            values = self.__properties[property]["values"]
            for value in values:
                if value["value"] == property_value:
                    return value["display_name"]
        return None



    def get_image_id_from_image_name(self, filename):
        pass

    @property
    @abc.abstractmethod
    def dataset_type_name(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load') and
                callable(subclass.load) or
                NotImplemented)

    @abc.abstractmethod
    def load(self):
        """Method to load dataset into memory"""
        raise NotImplementedError
