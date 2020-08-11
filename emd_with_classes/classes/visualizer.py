from emd_with_classes.utils import Iterator
from matplotlib import pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Polygon
from .dataset import Dataset


class Visualizer:

    def __init__(self, dataset_coco: Dataset):
        self.dataset = dataset_coco

    
    def visualize_annotations(self, categories=None):

        def show_image(image_path, index):
            im_id = images[index]["id"]
            print("Image with id:{}".format(im_id))
            plt.figure(figsize=(10, 10))
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            annIds = self.dataset.coco_lib.getAnnIds(imgIds=[im_id])
            anns = self.dataset.coco_lib.loadAnns(annIds)
            anns = [ann for ann in self.dataset.coco_lib.loadAnns(annIds) if
                    ann["category_id"] in category_ids]
            
 
            if len(anns) == 0:
                plt.show()
                return 0

            if self.dataset.is_segmentation and 'segmentation' in anns[0]:
                ax = plt.gca()
                for ann in anns:
                    cat = self.dataset.get_category_name_from_id(ann['category_id'])
                    color = colors[cat]
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
                ax = plt.gca()
                for ann in anns:
                    cat = self.dataset.get_category_name_from_id(ann['category_id'])
                    color = colors[cat]
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                            [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

                    ax.add_patch(Polygon(np_poly, linestyle='--', facecolor='none', edgecolor=color, linewidth=3))

            plt.show()
        

        if categories is None:
            categories = self.dataset.get_categories_names()
            images = self.dataset.get_images_id_with_path()
        else:
            images = []
            for cat in categories:
                ii = self.dataset.get_images_id_with_path_for_category(cat)
                images.extend(ii)
        paths = []
        for img in images:
            if not img["path"] in paths:
                paths.append(img["path"])
        colors = {}
        category_ids = []
        for c in  categories:
            colors[c] = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            category_ids.append(self.dataset.get_category_id_from_name(c))
        iterator = Iterator(paths, show_name=False, image_display_function=show_image)
        iterator.start_iteration()

    def visualize_annotations_for_property(self, meta_annotation, meta_annotation_value):
        def show_image(image_path, index):
            im_id = images[index]["id"]
            print("Image with id:{}".format(im_id))
            plt.figure(figsize=(10, 10))
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            annIds = self.dataset.coco_lib.getAnnIds(imgIds=[im_id])
            anns = self.dataset.coco_lib.loadAnns(annIds)
            anns = [ann for ann in self.dataset.coco_lib.loadAnns(annIds) if
                    ann[meta_annotation] == meta_annotation_value]
            
            if len(anns) == 0:
                plt.show()
                return 0

            if self.dataset.is_segmentation and 'segmentation' in anns[0]:
                ax = plt.gca()
                for ann in anns:
                    cat = self.dataset.get_category_name_from_id(ann['category_id'])
                    color = colors[cat]
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
                ax = plt.gca()
                for ann in anns:
                    cat = self.dataset.get_category_name_from_id(ann['category_id'])
                    color = colors[cat]
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                            [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

                    ax.add_patch(Polygon(np_poly, linestyle='--', facecolor='none', edgecolor=color, linewidth=3))

            plt.show()
            
        images = self.dataset.get_images_id_with_path_with_property_value(meta_annotation,
                                                                                     meta_annotation_value)
        colors = {}
        category_ids = []
        categories = self.dataset.get_categories_names()
        for c in  categories:
            colors[c] = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            category_ids.append(self.dataset.get_category_id_from_name(c))
        paths = [img["path"] for img in images]
        iterator = Iterator(paths, show_name=False, image_display_function=show_image)
        iterator.start_iteration()
        
    def visualize_annotations_for_class_for_property(self, category, meta_annotation, meta_annotation_value):
        def show_image(image_path, index):
            im_id = images[index]["id"]
            
            print("Image with id:{}".format(im_id))
            plt.figure(figsize=(10, 10))
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            category_id = self.dataset.get_category_id_from_name(category)
            annIds = self.dataset.coco_lib.getAnnIds(imgIds=[im_id], catIds=[category_id])
            anns = [ann for ann in self.dataset.coco_lib.loadAnns(annIds) if
                    ann[meta_annotation] == meta_annotation_value]

            if len(anns) == 0:
                plt.show()
                return 0

            if self.dataset.is_segmentation and 'segmentation' in anns[0]:
                ax = plt.gca()
                for ann in anns:
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
                ax = plt.gca()
                for ann in anns:
                    bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                            [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    
                    
                    ax.add_patch(Polygon(np_poly, linestyle='--', facecolor='none', edgecolor=color, linewidth=3))
                    ax.text(x=bbox_x, y=bbox_y, s=ann['category_id'], color='white', fontsize=9, horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor=color))
                plt.axis('off')
            plt.show()

        images = self.dataset.get_images_id_with_path_for_category_with_property_value(category, meta_annotation,
                                                                                     meta_annotation_value)
        color = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        paths = [img["path"] for img in images]
        iterator = Iterator(paths, show_name=False, image_display_function=show_image)
        iterator.start_iteration()