import os
from abc import ABC

import json
import numpy as np
import pandas as pd

from odin.classes import DatasetInterface, TaskType
from odin.utils import *
from odin.utils.utils import encode_segmentation, compute_aspect_ratio_of_segmentation
from pycocotools import mask
from pycocotools import coco

logger = get_root_logger()


class DatasetLocalization(DatasetInterface, ABC):

    annotations = None
    images = None

    possible_analysis = set()
    area_size_computed = False
    aspect_ratio = False

    common_properties = {'area', 'bbox', 'category_id', 'id', 'image_id', 'iscrowd', 'segmentation'}
    supported_types = [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]

    def __init__(self, dataset_gt_param, task_type, proposal_path=None, images_set_name='test',
                 images_abs_path=None, similar_classes=None, property_names=None, terminal_env=False,
                 properties_file=None, for_analysis=True, match_on_filename=False):

        if task_type not in self.supported_types:
            logger.error(f"Task not supported: {task_type}")

        super().__init__(dataset_gt_param, proposal_path, images_set_name, images_abs_path, similar_classes,
                         property_names, terminal_env, properties_file, match_on_filename)
        self.objnames_TP_graphs = []  # clear. the TRUE POSITIVE that can be drawn
        self.possible_analysis = set()  # clear. It would be updated according to the dataset provided
        self.is_segmentation = task_type == TaskType.INSTANCE_SEGMENTATION
        self.similar_classes = similar_classes
        self.for_analysis = for_analysis
        self.load()

    def dataset_type_name(self):
        return self.images_set_name

    def get_annotations_from_class_list(self, classes_to_classify):
        classes_id_filter = self.get_categories_id_from_names(classes_to_classify)
        anns_filtered = self.annotations[self.annotations["category_id"].isin(classes_id_filter)]
        return anns_filtered.to_dict("records")

    def is_segmentation_ds(self):
        return self.is_segmentation

    def __load_proposals(self):
        counter = 0
        issues = 0
        proposals = []
        for i, cat in self.categories.iterrows():
            c = cat["name"]
            c_id = cat["id"]
            proposal_path = os.path.join(self.proposal_path, c + ".txt")
            with open(proposal_path, "r") as file:
                for line in file:
                    if self.is_segmentation:
                        try:
                            arr = line.split(" ")
                            match_param, confidence = arr[0], float(arr[1])
                            if not self.match_on_filename:
                                match_param = int(match_param)
                            try:
                                segmentation = [float(v) for v in arr[2:]]
                            except:
                                segmentation = []
                            counter += 1
                            proposals.append(
                                {"confidence": confidence, "segmentation": segmentation, self.match_param_props: match_param,
                                 "category_id": c_id, "id": counter})
                        except:
                            issues += 1
                    else:
                        try:
                            match_param, confidence, x1, y1, x2, y2 = line.split(" ")
                            if not self.match_on_filename:
                                match_param = int(match_param)
                            confidence = float(confidence)
                            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                            counter += 1
                            proposals.append(
                                {"confidence": confidence, "bbox": [x1, y1, x2, y2], self.match_param_props: match_param,
                                 "category_id": c_id, "id": counter})
                        except:
                            issues += 1
        self.__proposals_length = counter
        logger.info("Loaded {} proposals and failed with {}".format(counter, issues))
        return pd.DataFrame(proposals)

    def load(self, force_loading=False):
        self.area_size_computed = False
        self.aspect_ratio = False
        try:
            if force_loading or self.coco_lib is None or self.annotations is None:
                self.coco_lib = coco.COCO(self.dataset_root_param)

                data = json.load(open(self.dataset_root_param, "r"))
                self.images = pd.DataFrame(data["images"])
                self.annotations = pd.DataFrame(data["annotations"])
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
            self.load_or_create_properties(self.annotations,
                                           self.common_properties)
        else:
            self.load_or_create_properties_notebook(self.annotations,
                                                    self.common_properties)
        self.compute_area_size()
        self.compute_aspect_ratio()

    def get_anns_for_category(self, category_id):
        return self.annotations[self.annotations["category_id"] == category_id]

    def get_number_of_annotations(self):
        return len(self.annotations.index)

    def get_num_images_with_category(self, category_id):
        return len(self.annotations[self.annotations["category_id"] == category_id].index)

    def get_annotations_of_class_with_property(self, category_id, property_name, value):
        if not property_name in self.annotations.keys():
            return []
        else:
            return self.annotations[(self.annotations[property_name] == value) &
                                    (self.annotations["category_id"] == category_id)]

    def get_annotations_with_property(self, property_name, value):
        if not property_name in self.annotations.keys():
            return []
        else:
            return self.annotations[self.annotations[property_name] == value]

    def get_image_with_id(self, image_id):
        return self.images[self.images["id"] == image_id]

    def compute_area_size(self):
        if self.area_size_computed:
            return
        areas = []
        anns = self.annotations.to_dict("records")
        for index, annotation in enumerate(anns):
            if self.is_segmentation:
                img = self.get_image_with_id(annotation["image_id"])
                encoded_mask = encode_segmentation(annotation["segmentation"][0], img["height"].values[0],
                                                   img["width"].values[0])
                area = mask.area(encoded_mask)
            else:
                _, _, w, h = annotation['bbox']
                area = w * h
            areas.append([area, index])

        labels = ["XS", "S", "M", "L", "XL"]

        # The areas are sorted and then the labels are assigned: 0.1%-XS, 0.2%-S, 0.4%-M, 0.2%-L, 0.1%-XL
        var_values = np.asarray([1 / 10, 3 / 10, 7 / 10, 9 / 10, 1], dtype=np.float) * len(areas)
        positions = np.around(var_values).astype(np.int)

        areas = sorted(areas, key=lambda x: x[0])
        prev = 0
        for i, pos in enumerate(positions):
            for j in range(prev, pos):
                real_index = areas[j][1]
                anns[real_index]["AreaSize"] = labels[i]
            prev = pos
        self.area_size_computed = True
        self.annotations = pd.DataFrame(anns)

    def compute_aspect_ratio(self):
        if self.aspect_ratio:
            return
        aspect_ratios = []
        anns = self.annotations.to_dict("records")
        for index, annotation in enumerate(anns):

            if self.is_segmentation:
                aspect_ratio = compute_aspect_ratio_of_segmentation(annotation["segmentation"][0])

            else:
                _, _, w, h = annotation['bbox']
                aspect_ratio = (w + 1) / (h + 1)
            aspect_ratios.append([aspect_ratio, index])

        labels = ['XT', 'T', 'M', 'W', 'XW']

        # The areas are sorted and then the labels are assigned: 0.1%-XT, 0.2%-T, 0.4%-M, 0.2%-W, 0.1%-XW
        var_values = np.asarray([1 / 10, 3 / 10, 7 / 10, 9 / 10, 1], dtype=np.float) * len(aspect_ratios)
        positions = np.around(var_values).astype(np.int)

        areas = sorted(aspect_ratios, key=lambda x: x[0])
        prev = 0
        for i, pos in enumerate(positions):
            for j in range(prev, pos):
                real_index = areas[j][1]
                anns[real_index]["AspectRatio"] = labels[i]
            prev = pos
        self.aspect_ratio = True
        self.annotations = pd.DataFrame(anns)

    def is_similar(self, cat1, cat2):
        for groups in self.similar_groups:
            if cat1 in groups and cat2 in groups:
                return True
        return False

    def get_annotations(self):
        return self.annotations

    def get_annotations_from_image(self, image_id):
        return self.annotations[self.annotations["image_id"] == image_id]

    def get_image_id_from_image_name(self, filename):
        filename = filename + ".png"
        return self.images[self.images["file_name"] == filename]["id"].values[0]

    def get_images_from_categories(self, categories):
        cat_ids = self.get_categories_id_from_names(categories)
        img_ids = self.annotations[self.annotations["category_id"].isin(cat_ids)]["image_id"].unique()
        return self.images[self.images["id"].isin(img_ids)]

    def get_height_width_from_image(self, image_id):
        image = self.get_image_with_id(image_id)
        return image["height"].values[0], image["width"].values[0]

    def get_proposals_with_ids(self, ids):
        return self.proposals[self.proposals["id"].isin(ids)]

    # rename to add for_category
    def get_proposals_by_ids(self, category, ids):
        cat_id = self.get_category_id_from_name(category)
        return self.proposals[(self.proposals["category_id"] == cat_id) & (self.proposals["id"].isin(ids))]

    def get_images_id_with_path(self):
        imgs_with_path = self.images.copy(deep=True)
        imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
        return imgs_with_path[["path", "id"]].to_dict("records")

    def get_images_id_with_path_for_category_with_property_value(self, category, property_name, property_value):

        category_id = self.get_category_id_from_name(category)
        images = self.get_annotations_of_class_with_property(category_id,
                                                              property_name, property_value)

        if len(images)> 0:
            img_ids = images["image_id"].unique()
            imgs_with_path = self.images.copy(deep=True)
            imgs_with_path = imgs_with_path[imgs_with_path["id"].isin(img_ids)]
            imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
            return imgs_with_path[["path", "id"]].to_dict("records")
        else:
            return []
    def get_images_id_with_path_for_category(self, category):
        category_id = self.get_category_id_from_name(category)
        img_ids = self.get_anns_for_category(category_id)["image_id"].unique()
        imgs_with_path = self.images.copy(deep=True)
        imgs_with_path = imgs_with_path[imgs_with_path["id"].isin(img_ids)]
        imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
        return imgs_with_path[["path", "id"]].to_dict("records")

    def get_images_id_with_path_with_property_value(self, property, property_value):
        images = self.get_annotations_with_property(property, property_value)
        if len(images) > 0:
            img_ids = images["image_id"].unique()
            imgs_with_path = self.images.copy(deep=True)
            imgs_with_path = imgs_with_path[imgs_with_path["id"].isin(img_ids)]
            imgs_with_path["path"] = imgs_with_path["file_name"].apply(lambda x: os.path.join(self.images_abs_path, x))
            return imgs_with_path[["path", "id"]].to_dict("records")
        else:
            return []
    def get_number_of_images(self):
        return len(self.images.index)