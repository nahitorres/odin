import os
import json
from PIL import Image
from matplotlib import pyplot as plt
from IPython.display import display
from tabulate import tabulate
from odin.classes.safe_writer import SafeWriter
from odin.classes import strings as labels_str
from odin.annotator import AnnotatorInterface
from odin.annotator import MetaPropertiesType
from odin.classes import TaskType
from odin.utils.leaflet_zoom_utils import get_image_container_zoom, show_new_image


class AnnotatorClassification(AnnotatorInterface):

    def set_display_function(self, custom_display_function):
        self.image_container = None
        if custom_display_function is None or type(custom_display_function) is str:
            # if custom_display_function == "zoom_js":
            #    self.custom_display_function = self.__show_image_js_zoom
            if custom_display_function == "zoom_leaflet":
                self.image_container = get_image_container_zoom()
                with self.out:
                    display(self.image_container)
                self.custom_display_function = self.show_image_leaflet
            elif custom_display_function == "default":
                self.custom_display_function = self.show_image
            else:
                raise NotImplementedError(f"Function {custom_display_function} not implemented!")
        else:
            self.custom_display_function = custom_display_function

    def set_objects(self):
        self.objects = self.dataset_annotated["observations"]
        self.max_pos = len(self.objects) - 1
        self.count_images = len(self.dataset_annotated["observations"])

    def add_annotation_to_mapping(self, ann):
        if self.validate:
            if not self.validate_function(ann):
                return
        else:
            for k_name, v in self.properties_and_values.items():
                if k_name not in ann:
                    return

        classification_type = self.dataset_orig.classification_type

        if classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            cat_names = [self.dataset_orig.get_category_name_from_id(c_id) for c_id in ann['categories']]
        elif classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL or classification_type == TaskType.CLASSIFICATION_BINARY:
            cat_names = [self.dataset_orig.get_category_name_from_id(ann['category'])]
        if ann['id'] not in self.mapping['annotated_ids']:  # only adds category counter if it was not annotate
            for cat_name in cat_names:
                self.mapping['categories_counter'][cat_name] += 1
        self.mapping['annotated_ids'].add(ann['id'])

    def update_mapping_from_whole_dataset(self):
        for ann in self.objects:
            self.add_annotation_to_mapping(ann)
        self.updated = True

    def update_annotation_counter_and_current_pos(self, dataset_annotated):
        #prop_names = set(self.properties_and_values.keys())
        last_ann_id = dataset_annotated["observations"][0]['id']
        self.current_pos = next(i for i, a in enumerate(dataset_annotated["observations"]) if a['id'] == last_ann_id)

    def checkbox_changed(self, b):

        if b['owner'].value is None or b['name'] != 'value':
            return

        class_name = b['owner'].description
        value = b['owner'].value
        annotation_name = b['owner']._dom_classes[0]

        ann =  self.get_image_record()
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
        if self.show_name:
            classification_type = self.dataset_orig.classification_type

            if classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
                categories = image_record['categories']
                str_output = os.path.basename(path_img) + ' - {}: '.format(labels_str.info_class)
                str_output +=  ','.join(['{} [id={}]'.format(self.dataset_orig.get_category_name_from_id(category),
                    category) for category in categories])
                print(str_output)

            elif classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL or classification_type == TaskType.CLASSIFICATION_BINARY:
                category = image_record['category']
                print(os.path.basename(path_img) + ' - {}: {} [id={}]'.format(labels_str.info_class,
                    self.dataset_orig.get_category_name_from_id(category),
                    category))

    def show_image_leaflet(self, image_record):
        path_img = os.path.join(self.dataset_orig.images_abs_path, image_record['file_name'])
        self.show_name_func(image_record, path_img)
        if not os.path.exists(path_img):
            print(f"{labels_str.info_missing} {path_img}")
            return

        show_new_image(self.image_container, path_img)

    def show_image(self, image_record):
        path_img = os.path.join(self.dataset_orig.images_abs_path, image_record['file_name'])
        self.show_name_func(image_record, path_img)
        if not os.path.exists(path_img):
            print(f"{labels_str.info_missing} {path_img}")
            return

        #   read img from path and show it
        img = Image.open(path_img)
        plt.figure(figsize=self.fig_size)     
        if not self.show_axis:
            plt.axis('off')
        plt.imshow(img)
        plt.show()

    def save_state(self):
        w = SafeWriter(os.path.join(self.file_path_for_json), "w")
        w.write(json.dumps(self.dataset_annotated, indent=4))
        w.close()
        self.add_annotation_to_mapping(self.dataset_annotated['observations'][self.current_pos])

    def perform_action(self):
        self.next_button.disabled = (self.current_pos == self.max_pos)
        self.previous_button.disabled = (self.current_pos == 0)

        current_ann = self.objects[self.current_pos]

        self.change_check_radio_boxes_value(current_ann)

        image_record = self.get_image_record()
        self.execute_validation(image_record)
        with self.out:
            if self.image_container is None: #is leaflet display
                self.out.clear_output()
            
            self.custom_display_function(image_record)

        self.text_index.unobserve(self.selected_index)
        self.text_index.value = self.current_pos + 1
        self.text_index.observe(self.selected_index)

    def get_image_record(self):
        img_record = self.objects[self.current_pos]
        return img_record

    def on_save_clicked(self, b):
        self.save_state()
        image_record = self.get_image_record()
        path_img = os.path.join(self.output_directory, 'JPEGImages', image_record['file_name'])
        self.save_function(path_img)

    def on_reset_clicked(self, b):
        current_ann = self.objects[self.current_pos]
        for m_k, m_v in self.properties_and_values.items():
            if m_k in current_ann:
                del current_ann[m_k]
        self.change_check_radio_boxes_value(current_ann)
        self.execute_validation(current_ann)

    def print_statistics(self):
        if not self.updated:
            self.update_mapping_from_whole_dataset()
        table = []
        total = 0
        for c_k, c_number in self.mapping["categories_counter"].items():
            table.append([c_k, c_number])
            total += c_number
        table = sorted(table, key=lambda x: x[0])

        classification_type = self.dataset_orig.classification_type
        # show total images only in binary/single-label
        if classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            table.append([labels_str.info_total, '{}'.format(total)])
        elif classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL or classification_type == TaskType.CLASSIFICATION_BINARY:
            table.append([labels_str.info_total, '{}/{}'.format(total, self.count_images)])
        print(tabulate(table, headers=[labels_str.info_class_name, labels_str.info_ann_objects]))
    
    def print_results(self):
        complete, incomplete = 0, 0
        incomplete_srt = ""
        if self.validate:
            for index, record in enumerate(self.objects):
                if self.validate_function(record):
                    complete += 1
                else:
                    incomplete += 1
                    incomplete_srt += f" {index+1}"
        print(f"{labels_str.info_completed} {complete}")
        print(f"{labels_str.info_incomplete} {incomplete}")
        if incomplete > 0:
            print(f"{labels_str.info_positions} {incomplete_srt}")
