import os
import json
from PIL import Image
from matplotlib import pyplot as plt
from tabulate import tabulate
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import numpy as np
from odin.classes.safe_writer import SafeWriter
from odin.annotator import AnnotatorInterface, MetaPropertiesType
from odin.classes import strings as labels_str


class AnnotatorLocalization(AnnotatorInterface):

    def set_objects(self):
        self.objects = self.dataset_orig.get_annotations_from_class_list(self.classes_to_annotate)
        self.max_pos = len(self.objects) - 1
        self.count_images = len(self.dataset_annotated["images"])
        
    def set_display_function(self, custom_display_function):
        if custom_display_function is None:
            self.custom_display_function = self.show_image
        else:
            self.custom_display_function = custom_display_function

    def add_annotation_to_mapping(self, ann):
        if self.validate:
            if not self.validate_function(ann):
                return
        else:
            for k_name, v in self.properties_and_values.items():
                if k_name not in ann:
                    return

        cat_name = self.dataset_orig.get_category_name_from_id(ann['category_id'])
        if ann['id'] not in self.mapping['annotated_ids']:  # only adds category counter if it was not annotate
            self.mapping['categories_counter'][cat_name] += 1
        self.mapping['annotated_ids'].add(ann['id'])

    def update_mapping_from_whole_dataset(self):
        for ann in self.objects:
            self.add_annotation_to_mapping(ann)
        self.updated = True

    def update_annotation_counter_and_current_pos(self, dataset_annotated):
        #prop_names = set(self.properties_and_values.keys())
        #objects = self.dataset_orig.get_annotations_from_class_list(self.classes_to_annotate)
        objects = dataset_annotated['annotations']
        last_ann_id = objects[0]['id']
        '''for ann in dataset_annotated['annotations']:
            if prop_names.issubset(
                    set(ann.keys())):  # we consider that it was annotated when all the props are subset of keys
                self.add_annotation_to_mapping(ann)
                last_ann_id = ann['id']  # to get the last index in self.object so we can update the current pos in the last one anotated
            else:
                break'''
        self.current_pos = next(i for i, a in enumerate(objects) if a['id'] == last_ann_id)

    def checkbox_changed(self, b):
        if b['owner'].value is None or b['name'] != 'value':
            return

        class_name = b['owner'].description
        value = b['owner'].value
        annotation_name = b['owner']._dom_classes[0]

        ann_id = self.objects[self.current_pos]['id']
        # image_id = self.objects[self.current_pos]['image_id']
        for ann in self.dataset_annotated['annotations']:
            if ann['id'] == ann_id:
                break
        if self.properties_and_values[annotation_name][0].value in [MetaPropertiesType.COMPOUND.value]:
            if annotation_name not in ann.keys():
                ann[annotation_name] = {p: False for p in self.properties_and_values[annotation_name][1]}
            ann[annotation_name][class_name] = value
        else:  # UNIQUE VALUE
            ann[annotation_name] = value

        self.execute_validation(ann)

        if self.current_pos == self.max_pos:
            self.save_state()

    def show_name_func(self, image_record, path_img):
        print(os.path.basename(path_img) + '. Class: {} [ID={}]'.format(
            self.dataset_orig.get_category_name_from_id(self.objects[self.current_pos]['category_id']),
            self.objects[self.current_pos]['category_id']))

    def show_image(self, image_record, ann_key):
        #   read img from path and show it
        path_img = os.path.join(self.dataset_orig.images_abs_path, image_record['file_name'])
        img = Image.open(path_img)
        if self.show_name:
           self.show_name_func()
        plt.figure(figsize=self.fig_size)

        if not self.show_axis:
            plt.axis('off')
        plt.imshow(img)

        # draw the bbox from the object onum
        ax = plt.gca()
        class_colors = cm.rainbow(np.linspace(0, 1, len(self.dataset_orig.get_categories_names())))

        annotation = self.dataset_orig.coco_lib.anns[ann_key]
        object_class_name = self.dataset_orig.get_category_name_from_id(annotation['category_id'])
        c = class_colors[list(self.dataset_orig.get_categories_names()).index(object_class_name)]
        if not self.dataset_orig.is_segmentation:
            [bbox_x1, bbox_y1, diff_x, diff_y] = annotation['bbox']
            bbox_x2 = bbox_x1 + diff_x
            bbox_y2 = bbox_y1 + diff_y
            poly = [[bbox_x1, bbox_y1], [bbox_x1, bbox_y2], [bbox_x2, bbox_y2],
                    [bbox_x2, bbox_y1]]
            np_poly = np.array(poly).reshape((4, 2))

            # draws the bbox
            ax.add_patch(
                Polygon(np_poly, linestyle='-', facecolor=(c[0], c[1], c[2], 0.0),
                        edgecolor=(c[0], c[1], c[2], 1.0), linewidth=2))
        else:
            seg_points = annotation['segmentation']
            for pol in seg_points:
                poly = [[float(pol[i]), float(pol[i + 1])] for i in range(0, len(pol), 2)]
                np_poly = np.array(poly)  # .reshape((len(pol), 2))
                ax.add_patch(
                    Polygon(np_poly, linestyle='-', facecolor=(c[0], c[1], c[2], 0.25),
                            edgecolor=(c[0], c[1], c[2], 1.0), linewidth=2))
            # set the first XY point for printing the text
            bbox_x1 = seg_points[0][0];
            bbox_y1 = seg_points[0][1]

        #  write the class name in bbox
        ax.text(x=bbox_x1, y=bbox_y1, s=object_class_name, color='white', fontsize=9, horizontalalignment='left',
                verticalalignment='top',
                bbox=dict(facecolor=(c[0], c[1], c[2], 0.5)))
        plt.show()

    def save_state(self):
        w = SafeWriter(os.path.join(self.file_path_for_json), "w")
        w.write(json.dumps(self.dataset_annotated, indent=4))
        w.close()
        self.add_annotation_to_mapping(next(ann_dat for ann_dat in self.dataset_annotated['annotations']
                                              if ann_dat['id'] == self.objects[self.current_pos]['id']))

    def perform_action(self):
        self.next_button.disabled = (self.current_pos == self.max_pos)
        self.previous_button.disabled = (self.current_pos == 0)

        # current_class_id = self.objects[self.current_pos]['category_id']
        current_ann = next(
            a for a in self.dataset_annotated['annotations'] if a['id'] == self.objects[self.current_pos]['id'])
        # current_gt_id = self.objects[self.current_pos]['gt_id']
        self.change_check_radio_boxes_value(current_ann)

        self.execute_validation(current_ann)

        with self.out:
            self.out.clear_output()
            image_record, ann_key = self.get_image_record()
            self.custom_display_function(image_record, ann_key)

        self.text_index.unobserve(self.selected_index)
        self.text_index.value = self.current_pos + 1
        self.text_index.observe(self.selected_index)

    def get_image_record(self):
        # current_class_id = self.objects[self.current_pos]['category_id']
        current_image_id = self.objects[self.current_pos]['image_id']

        ann_key = self.objects[self.current_pos]['id']
        img_record = self.dataset_orig.coco_lib.imgs[current_image_id]
        return img_record, ann_key

    def on_reset_clicked(self, b):
        current_ann = next(a for a in self.dataset_annotated['annotations'] if a['id'] == self.objects[self.current_pos]['id'])
        for m_k, m_v in self.properties_and_values.items():
            if m_k in current_ann:
                del current_ann[m_k]

        self.change_check_radio_boxes_value(current_ann)
        self.execute_validation(current_ann)

    def on_save_clicked(self, b):
        self.save_state()
        image_record, _ = self.get_image_record()
        path_img = os.path.join(self.output_directory, 'JPEGImages', image_record['file_name'])
        self.save_function(path_img)

    def print_statistics(self):
        if not self.updated:
            self.update_mapping_from_whole_dataset()
        table = []
        total = 0
        for c_k, c_number in self.mapping["categories_counter"].items():
            table.append([c_k, c_number])
            total += c_number
        table = sorted(table, key=lambda x: x[0])
        table.append(['Total', '{}/{}'.format(total, len(self.objects))])
        print(tabulate(table, headers=[labels_str.info_class_name, labels_str.info_ann_objects]))

    def print_results(self):
        complete, incomplete = 0, 0
        incomplete_srt = ""
        if self.validate:
            for index, record in enumerate(self.dataset_annotated['annotations']):
                if self.validate_function(record):
                    complete += 1
                else:
                    incomplete += 1
                    incomplete_srt += f" {index + 1}"
        print(f"{labels_str.info_completed_obj} {complete}")
        print(f"{labels_str.info_incomplete_obj} {incomplete}")
        if incomplete > 0:
            print(f"{labels_str.info_positions} {incomplete_srt}")
