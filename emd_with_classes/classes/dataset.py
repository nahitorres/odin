import os
from abc import ABC
import json
import numpy as np
from .dataset_interface import DatasetInterface
from emd_with_classes.utils import *
from emd_with_classes.utils.utils import encode_segmentation, compute_aspect_ratio_of_segmentation
from pycocotools import mask
from pycocotools import coco
from collections import defaultdict

logger = get_root_logger()


class Dataset(DatasetInterface, ABC):
    proposals_path = ''
    dataset_root_param = ''
    images_abs_path = ''
    categories = []
    similar_classes = []

    images_set_name = 'test'

    possible_analysis = set()
    coco_lib = None
    area_size_computed = False
    aspect_ratio = False

    dataset = None
    proposals = None

    similar_groups = []
    __properties = {}

    def __init__(self, dataset_gt_param, proposal_path, is_segmentation, images_set_name='test',
                 images_abs_path=None, similar_classes=None):

        self.dataset_root_param = dataset_gt_param
        self.proposal_path = proposal_path
        self.images_set_name = images_set_name
        self.objnames_TP_graphs = []  # clear. the TRUE POSITIVE that can be drawn
        self.possible_analysis = set()  # clear. It would be updated according to the dataset provided
        self.images_abs_path = images_abs_path
        self.is_segmentation = is_segmentation
        self.similar_classes = similar_classes
        self.__proposals_length = 0
        self.load()
        self.compute_area_size()
        self.compute_aspect_ratio()
        if not similar_classes is None:
            self.similar_groups = similar_classes

    def load_or_create_properties(self):
        if not os.path.exists("properties.json"):
            common_properties = {'area', 'bbox', 'category_id', 'id', 'image_id', 'iscrowd', 'segmentation'}
            anns = self.coco_lib.loadAnns(ids=self.coco_lib.getAnnIds())
            properties = set(anns[0].keys()) - common_properties
            p_values = defaultdict(list)

            for ann in anns:
                properties = properties.intersection(ann.keys())
                for p in properties:
                    p_values[p].append(ann[p])

            values = {}
            for p in properties:
                p_values[p] = set(p_values[p])
                if 1 < len(p_values[p]) < 10:
                    values_array = []
                    for value in p_values[p]:
                        values_array.append({"value": value, "display_name": value})
                    values[p] = {"values": values_array, "display_name": p}

            json.dump(values, open("properties.json", "w"), indent=4)
        return json.load(open("properties.json", "r"))

    def get_categories_names(self):
        return [c["name"] for c in self.coco_lib.loadCats(ids=self.coco_lib.getCatIds())]


    def get_annotations_from_class_list(self, classes_to_classify):
        all_annotations = []
        classes_id_filter = self.get_categories_id_from_names(classes_to_classify)

        for id_ann, ann in self.coco_lib.anns.items():
            if ann['category_id'] in classes_id_filter:
                all_annotations.append(ann)
        return all_annotations

    def is_segmentation_ds(self):
        return self.is_segmentation

    def __load_proposals(self):
        pdict = {}
        counter = 0
        issues = 0
        for category in self.coco_lib.loadCats(self.coco_lib.getCatIds()):
            c = category["name"]
            c_id = category["id"]
            proposals_dict = defaultdict(list)
            proposal_path = os.path.join(self.proposal_path, c + ".txt")
            if not os.path.exists(proposal_path):
                pdict[c] = proposals_dict
            with open(proposal_path, "r") as file:
                for line in file:

                    if self.is_segmentation:
                        try:
                            arr = line.split(" ")
                            filename, confidence = arr[0], float(arr[1])
                            try:
                                segmentation = [float(v) for v in arr[2:]]
                            except:
                                segmentation = []
                            counter += 1
                            proposals_dict[filename].append(
                                {"confidence": confidence, "segmentation": segmentation, "image_name": filename, "category_id": c_id, "id": counter})
                        except:
                            issues += 1
                    else:
                        try:
                            filename, confidence, x1, y1, x2, y2 = line.split(" ")
                            confidence = float(confidence)
                            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                            counter += 1
                            proposals_dict[filename].append(
                                {"confidence": confidence, "bbox": [x1, y1, x2, y2], "image_name": filename, "category_id": c_id, "id": counter})
                        except:
                            issues+=1
            for filename in proposals_dict.keys():
                if len(proposals_dict[filename]) > 1:
                    proposals_dict[filename] = sorted(proposals_dict[filename], key=lambda k: k['confidence'],
                                                      reverse=True)

            pdict[c] = proposals_dict
        self.__proposals_length = counter
        logger.info("Loaded {} proposals and failed with {}".format(counter, issues))
        return pdict

    def load(self, force_loading=False):
        self.area_size_computed = False
        self.aspect_ratio = False
        if force_loading or self.coco_lib is None:
            self.coco_lib = coco.COCO(self.dataset_root_param)
        if force_loading or self.proposals is None:
            self.proposals = self.__load_proposals()
        self.__properties = self.load_or_create_properties()
        self.compute_area_size()
        self.compute_aspect_ratio()

    def dataset_type_name(self):
        return self.images_set_name

    def get_cant_anns_for_category(self, category_id):
        return len(self.coco_lib.getAnnIds(catIds=[category_id]))

    def get_anns_for_category(self, category_id):
        return self.coco_lib.loadAnns(self.coco_lib.getAnnIds(catIds=[category_id]))

    def get_number_of_images(self):
        return len(self.coco_lib.getImgIds())

    def get_number_of_annotations(self):
        return len(self.coco_lib.getAnnIds())

    def get_category_id_from_name(self, category):
        return [c["id"] for c in self.coco_lib.loadCats(ids=self.coco_lib.getCatIds()) if c["name"] == category][0]

    def get_categories_id_from_names(self, categories):
        return [c["id"] for c in self.coco_lib.loadCats(ids=self.coco_lib.getCatIds()) if c["name"] in categories]

    def get_num_images_with_category(self, category_id):
        return len(self.coco_lib.getImgIds(catIds=[category_id]))

    def get_category_name_from_id(self, category_id):

        return [c["name"] for c in self.coco_lib.loadCats(ids=[category_id])][0]

    def get_proposals_dict(self):
        if self.proposals == None:
            self.__load_proposals()
        return self.proposals

    def get_annotations_of_class_with_property(self, category_id, property_name, value):
        return [ann for ann in self.coco_lib.loadAnns(self.coco_lib.getAnnIds(catIds=[category_id])) if
                property_name in ann.keys() and ann[property_name] == value]

    def get_annotations_with_property(self,  property_name, value):
        return [ann for ann in self.coco_lib.loadAnns(self.coco_lib.getAnnIds()) if
                property_name in ann.keys() and ann[property_name] == value]

    def get_image_with_id(self, image_id):
        return self.coco_lib.loadImgs(ids=[image_id])[0]

    def get_images_with_id(self, image_id):
        return self.coco_lib.loadImgs(ids=[image_id])

    def compute_area_size(self):
        if self.area_size_computed:
            return
        areas = []
        anns = self.coco_lib.loadAnns(ids=self.coco_lib.getAnnIds())
        for index, annotation in enumerate(anns):
            if self.is_segmentation:
                img = self.get_image_with_id(annotation["image_id"])
                encoded_mask = encode_segmentation(annotation["segmentation"][0], img["height"], img["width"])
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

    def compute_aspect_ratio(self):
        if self.aspect_ratio:
            return
        aspect_ratios = []
        anns = self.coco_lib.loadAnns(ids=self.coco_lib.getAnnIds())
        for index, annotation in enumerate(anns):

            if self.is_segmentation:
                aspect_ratio = compute_aspect_ratio_of_segmentation(annotation["segmentation"][0])

            else:
                _, _, w, h = annotation['bbox']
                aspect_ratio = (w + 1) / (h + 1)  # TODO: check +1 for non zero division
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

    def is_similar(self, cat1, cat2):
        for groups in self.similar_groups:
            if cat1 in groups and cat2 in groups:
                return True
        return False

    def get_annotations(self):
        return  self.coco_lib.loadAnns(ids=self.coco_lib.getAnnIds())

    def get_annotations_from_image(self, image_id):
        # TODO: check
        return self.coco_lib.loadAnns(ids=self.coco_lib.getAnnIds(imgIds=[image_id]))

    def get_image_id_from_image_name(self, filename):

        return [im["id"] for im in self.coco_lib.loadImgs(ids=self.coco_lib.getImgIds()) if
                im["file_name"].split(".")[0] == filename][0]

    def get_images_from_categories(self, categories):
        categories_id = self.get_categories_id_from_names(categories)
        images = {}
        for i, v in self.coco_lib.catToImgs.items():
            if i in categories_id:
                for img in self.get_images_with_id(v):
                    images[img['id']] = img
        return list(images.values())

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

    def get_height_with_from_image(self, image_id):
        image = self.get_image_with_id(image_id)
        return image["height"], image["width"]

    def get_proposals_of_category(self, category):
        if self.proposals == None:
            self.__load_proposals()
        detections = []
        for k in self.proposals[category]:
            detections.extend(self.proposals[category][k])
        return detections

    def get_proposals(self):
        if self.proposals == None:
            self.__load_proposals()
        detections = []
        for category in self.get_categories_names():
            for k in self.proposals[category]:
                detections.extend(self.proposals[category][k])
        return detections

    def get_proposals_with_ids(self, ids):
        if self.proposals == None:
            self.__load_proposals()
        detections = []
        for category in self.get_categories_names():
            for k in self.proposals[category]:
                for d in self.proposals[category][k]:
                    if d["id"] in ids:
                        detections.append(d)
        return detections

    #rename to add for_category
    def get_proposals_by_ids(self, category, ids):
        if self.proposals == None:
            self.__load_proposals()
        detections = []
        for k in self.proposals[category]:
            for d in self.proposals[category][k]:
                if d["id"] in ids:
                    detections.append(d)
        return detections



    def get_display_name_of_property_value(self, property, property_value):

        if property in self.__properties.keys():
            values = self.__properties[property]["values"]
            for value in values:
                if value["value"] == property_value:
                    return value["display_name"]
        return None

    def get_images_id_with_path(self):
        imgs_with_path = []
        for img in self.coco_lib.loadImgs(ids=self.coco_lib.getImgIds()):
            imgs_with_path.append({"path": os.path.join(self.images_abs_path, img["file_name"]), "id":img["id"]})
        return imgs_with_path

    def get_images_id_with_path_for_category_with_property_value(self, category, property_name, property_value):
        category_id = self.get_category_id_from_name(category)
        anns = self.get_annotations_of_class_with_property(category_id, property_name, property_value)
        img_ids = set([ann["image_id"] for ann in anns])

        imgs_with_path = []
        for img in self.coco_lib.loadImgs(ids=img_ids):
            imgs_with_path.append({"path": os.path.join(self.images_abs_path, img["file_name"]), "id": img["id"]})
        return imgs_with_path

    def get_images_id_with_path_for_category(self, category):
        category_id = self.get_category_id_from_name(category)
        anns = self.get_anns_for_category(category_id)
        img_ids = set([ann["image_id"] for ann in anns])

        imgs_with_path = []
        for img in self.coco_lib.loadImgs(ids=img_ids):
            imgs_with_path.append({"path": os.path.join(self.images_abs_path, img["file_name"]), "id": img["id"]})
        return imgs_with_path

    def get_images_id_with_path_with_property_value(self, property, property_value):

        anns = self.get_annotations_with_property( property, property_value)
        img_ids = set([ann["image_id"] for ann in anns])

        imgs_with_path = []
        for img in self.coco_lib.loadImgs(ids=img_ids):
            imgs_with_path.append({"path": os.path.join(self.images_abs_path, img["file_name"]), "id": img["id"]})
        return imgs_with_path