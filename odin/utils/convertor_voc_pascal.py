import os
import json
import xmltodict
from tqdm import tqdm, tqdm_notebook
from odin.utils.env import is_notebook


class VOCtoCoco:

    def __init__(self, images, images_path, annotations_path, output_path):
        self.images = images
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.output_path = output_path
        self.__tqdm = tqdm_notebook if is_notebook() else tqdm

    def __get_category_id_from_name(self, coco_ds, category_name):
        index = 1
        for c in coco_ds["categories"]:
            index += 1
            if c["name"] == category_name:
                return c["id"]
        new_cat = {"id": index, "name": category_name, "supercategory": category_name}
        coco_ds["categories"].append(new_cat)
        return index

    def convert_and_save(self):
        coco_ds = {}
        coco_ds["categories"] = []
        coco_ds["annotations"] = []
        coco_ds["images"] = []
        imgs_counter = 1
        anns_counter = 1
        for im in self.__tqdm(self.images):
            ann_path = os.path.join(self.annotations_path, im + ".xml")
            img_path = os.path.join(self.images_path, im + ".jpg")

            if os.path.exists(ann_path) and os.path.exists(img_path):
                ann_srt = ""
                with open(ann_path, "r") as file_content:
                    for line in file_content:
                        ann_srt += line.replace("\n", "")
                ann = xmltodict.parse(ann_srt)

                if "segmented" in ann.keys() and ann["segmented"]:
                    continue

                image_info = {
                    'file_name': ann["annotation"]["filename"],
                    'height': ann["annotation"]["size"]["height"],
                    'width': ann["annotation"]["size"]["width"],
                    'id': imgs_counter
                }
                coco_ds["images"].append(image_info)
                if type(ann["annotation"]["object"]) == type([]):
                    obj_to_iterate = ann["annotation"]["object"]
                else:
                    obj_to_iterate = [ann["annotation"]["object"]]

                for obj in obj_to_iterate:

                    base_keys = ["bndbox", "name"]

                    coco_ann = {}
                    xmin, xmax, ymin, ymax = float(obj["bndbox"]["xmin"]), float(obj["bndbox"]["xmax"]), float(
                        obj["bndbox"]["ymin"]), float(obj["bndbox"]["ymax"])
                    coco_ann["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
                    coco_ann["category_id"] = self.__get_category_id_from_name(coco_ds, obj["name"])
                    coco_ann["id"] = anns_counter
                    coco_ann["image_id"] = imgs_counter
                    for key in obj:
                        if not key in base_keys:
                            coco_ann[key] = obj[key]
                    coco_ds["annotations"].append(coco_ann)
                    anns_counter += 1

                imgs_counter += 1
        json.dump(coco_ds, open(self.output_path, "w"), indent=4)
        return coco_ds
