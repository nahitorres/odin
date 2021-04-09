import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from .dataset_localization import DatasetLocalization
from odin.utils import Iterator
from .visulizer_interface import VisualizerInterface
from odin.classes import strings as labels_str

class VisualizerLocalization(VisualizerInterface):

    def __init__(self, dataset: DatasetLocalization):
        self.dataset = dataset

        self.__colors = {}
        
        category_ids = []
        for c in  self.dataset.get_categories_names():
            self.__colors[c] = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            category_ids.append(self.dataset.get_category_id_from_name(c))



    def __show_image(self, image_path, index):
            im_id = self.__current_images[index]["id"]

            print("Image with id:{}".format(im_id))
            if not os.path.exists(image_path):
                print("Image path does not exist: " + image_path )
            else:
                plt.figure(figsize=(10, 10))
                img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                
                if self.__current_category is None:
                    annIds = self.dataset.coco_lib.getAnnIds(imgIds=[im_id])
                elif type(self.__current_category) is int:
                    annIds = self.dataset.coco_lib.getAnnIds(imgIds=[im_id], catIds=[self.__current_category])
                else:
                    annIds = self.dataset.coco_lib.getAnnIds(imgIds=[im_id], catIds=self.__current_category)


                if self.__current_meta_anotation != None and self.__meta_annotation_value != None:
                    anns = [ann for ann in self.dataset.coco_lib.loadAnns(annIds) if
                            ann[self.__current_meta_anotation] == self.__meta_annotation_value]
                else:
                    anns = [ann for ann in self.dataset.coco_lib.loadAnns(annIds)]

                if len(anns) == 0:
                    plt.show()
                    return 0

                if self.dataset.is_segmentation and 'segmentation' in anns[0]:
                    # TODO: move to another function
                    ax = plt.gca()
                    for ann in anns:
                        cat = self.dataset.get_category_name_from_id(ann['category_id'])
                        color = self.__colors[cat]
                        seg_points = ann["segmentation"]
                        for pol in seg_points:
                            poly = [[float(pol[i]), float(pol[i+1])] for i in range(0, len(pol), 2)]
                            np_poly = np.array(poly)

                        ax.add_patch(
                                Polygon(np_poly, linestyle='--', fill=False, facecolor='none', edgecolor=color, linewidth=2))
                        ax.text(x=seg_points[0][0], y=seg_points[0][1], s=ann['category_id'], color='white', fontsize=9, horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor=color))


                    plt.imshow(img)

                    plt.axis('off')
                    plt.show()
                else:
                    # TODO: move to another function
                    ax = plt.gca()
                    for ann in anns:
                        cat = self.dataset.get_category_name_from_id(ann['category_id'])
                        color = self.__colors[cat]
                        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                                [bbox_x + bbox_w, bbox_y]]
                        np_poly = np.array(poly).reshape((4, 2))

                        ax.add_patch(Polygon(np_poly, linestyle='--', facecolor='none', edgecolor=color, linewidth=3))
                        ax.text(x=bbox_x, y=bbox_y, s=ann['category_id'], color='white', fontsize=9, horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor=color))
                    plt.axis('off')
                plt.show()
            
    def visualize_annotations(self, categories=None):
        categories_ds = self.dataset.get_categories_names()
        if categories is None:
            categories = categories_ds
            images = self.dataset.get_images_id_with_path()
        else:
            images = []
            for cat in categories:
                if cat in categories_ds:
                    ii = self.dataset.get_images_id_with_path_for_category(cat)
                    images.extend(ii)
                else:
                    print(labels_str.warn_incorrect_class)
        
        category_ids = [self.dataset.get_category_id_from_name(c) for c in categories]

        self.__start_iterator( images, category=category_ids)

    def visualize_annotations_for_property(self, meta_annotation, meta_annotation_value):
        
        images = self.dataset.get_images_id_with_path_with_property_value(meta_annotation,
                                                                                     meta_annotation_value)
        
        self.__start_iterator(images, meta_annotation= meta_annotation, meta_annotation_value=meta_annotation_value)

    def visualize_annotations_for_class_for_property(self, category, meta_annotation, meta_annotation_value):
       
        if self.dataset.is_valid_category(category):

            images = self.dataset.get_images_id_with_path_for_category_with_property_value(category, meta_annotation,
                                                                                         meta_annotation_value)
            category_id = self.dataset.get_category_id_from_name(category)
            self.__start_iterator(images, category=category_id, meta_annotation=meta_annotation, meta_annotation_value=meta_annotation_value)

        else:
            print(labels_str.warn_incorrect_class)
            
    def __start_iterator(self,  images, category=None, meta_annotation=None, meta_annotation_value=None):
        self.__current_category = category
        self.__current_images = images
        self.__current_meta_anotation = meta_annotation
        self.__meta_annotation_value = meta_annotation_value
        paths = [img["path"] for img in images]
        if len(paths) == 0:
            print(labels_str.warn_no_images_criteria)
        else:
            iterator = Iterator(paths, show_name=False, image_display_function=self.__show_image)
            iterator.start_iteration()