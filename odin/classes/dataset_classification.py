import os
import pandas as pd
import json
from abc import ABC


from odin.classes import DatasetInterface, TaskType
from odin.utils import *

logger = get_root_logger()


class DatasetClassification(DatasetInterface, ABC):

    observations = None

    common_properties = {'id', 'file_name', 'height', 'width', 'categories', 'category'}
    supported_types = [TaskType.CLASSIFICATION_BINARY, TaskType.CLASSIFICATION_SINGLE_LABEL, TaskType.CLASSIFICATION_MULTI_LABEL]

    def __init__(self, dataset_gt_param, task_type, proposal_path=None, observations_set_name='test',
                 observations_abs_path=None, similar_classes=None, property_names=None, terminal_env=False,
                 properties_file=None, for_analysis=True, match_on_filename=False):

        if task_type not in self.supported_types:
            logger.error(f"Unsupported classification type: {task_type}")
            return

        super().__init__(dataset_gt_param, proposal_path, observations_set_name, observations_abs_path, similar_classes,
                         property_names, terminal_env, properties_file, match_on_filename)
        self.classification_type = task_type
        self.property_names = property_names
        self.for_analysis = for_analysis
        
        self.load()
        if similar_classes is not None:
            self.similar_groups = similar_classes

    def dataset_type_name(self):
        return self.images_set_name

    def __load_proposals(self):
        counter = 0
        issues = 0
        proposals = []
        for i, cat in self.categories.iterrows():
            c = cat['name']
            c_id = cat['id']
            proposal_path = os.path.join(self.proposal_path, c + ".txt")
            with open(proposal_path, "r") as file:
                for line in file:
                    try:
                        match_param, confidence = line.split(" ")
                        if not self.match_on_filename:
                            match_param = int(match_param)
                        confidence = float(confidence)
                        counter += 1
                        proposals.append(
                            {"id": counter, self.match_param_props: match_param, "confidence": confidence,
                             "category_id": c_id}
                        )
                    except:
                        issues += 1
        self.__proposals_length = counter
        logger.info("Loaded {} proposals and failed with {}".format(counter, issues))
        return pd.DataFrame(proposals)

    def load(self, force_loading=False):
        try:
            if force_loading or self.observations is None:
                data = json.load(open(self.dataset_root_param, "r"))
                self.observations = pd.DataFrame(data['observations'])
                self.categories = pd.DataFrame(data["categories"])
        except:
            logger.error("Error loading dataset.")
            return
        if not self.for_analysis:
            return
        try:
            if force_loading or self.proposals is None:
                self.proposals = self.__load_proposals()
        except:
            logger.error("Error loading proposals.")
            return
        if self.terminal_env or self.property_names is not None:
            self.load_or_create_properties(self.observations,
                                           self.common_properties)
        else:
            self.load_or_create_properties_notebook(self.observations,
                                                    self.common_properties)

    def is_similar_classification(self, cat1, cats2):
        for groups in self.similar_groups:
            if self.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
                for cat2 in cats2:
                    if cat1 in groups and cat2 in groups:
                        return True
            else:
                if cat1 in groups and cats2 in groups:
                    return True
        return False

    def get_observation_id_from_file_name(self, filename):
        return self.observations[self.observations["file_name"] == filename]["id"].values[0]


    def get_observations_from_categories(self, categories):

        categories_id = self.get_categories_id_from_names(categories)

        if self.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            return self.observations[self.observations["categories"].apply(lambda x: any(cat_id in x for cat_id in
                                                                                         categories_id))]
        else:
            return self.observations[self.observations["category"].isin(categories_id)]

    def get_observations_from_property_category(self, category_id, property_name, property_value):
        if self.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            return self.observations[(self.observations[property_name] == property_value) &
                                     (self.observations["categories"].apply(lambda x: category_id in x))]
        else:
            return self.observations[(self.observations[property_name] == property_value) &
                                     (self.observations["category"] == category_id)]

    def get_observations_from_property(self, property_name, property_value):
        return self.observations[self.observations[property_name] == property_value]

    def get_number_of_observations(self):
        return len(self.observations.index)

    def get_all_observations(self):
        return self.observations
