import os
import json
from PIL import Image
from matplotlib import pyplot as plt
from ipywidgets import Button, Output, HBox, VBox, Label, BoundedIntText, Checkbox, HTML, RadioButtons, BoundedFloatText
from IPython.display import Javascript, display
from collections import defaultdict
from tabulate import tabulate
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import numpy as np
import copy
from emd_with_classes.classes.dataset import Dataset
from enum import Enum
from emd_with_classes.classes.safe_writer import SafeWriter


class MetaPropertiesTypes(Enum):
    META_PROP_UNIQUE = 0
    META_PROP_COMPOUND = 1
    META_PROP_CONTINUE = 2


class Annotator:

    def __init__(self,
                 dataset,
                 metrics_and_values,  # properties{ name: ([possible values], just_one_value = True)}
                 output_path=None,
                 show_name=True,
                 show_axis=False,
                 fig_size=(10, 10),
                 buttons_vertical=False,
                 image_display_function=None,
                 classes_to_annotate=None
                 ):

        self.dataset_orig = dataset
        self.metrics_and_values = metrics_and_values
        self.show_axis = show_axis
        self.name = (self.dataset_orig.dataset_root_param.split('/')[-1]).split('.')[0]  # get_original_file_name
        self.show_name = show_name
        if output_path is None:
            splitted_array = self.dataset_orig.dataset_root_param.split('/')
            n = len(splitted_array)
            self.output_directory = os.path.join(*(splitted_array[0:n - 1]))
        else:
            self.output_directory = output_path

        if classes_to_annotate is None:  # if classes_to_annotate is None, all the classes would be annotated
            self.classes_to_annotate = self.dataset_orig.get_categories_names()  # otherwise, the only the classes in the list

        self.file_path_for_json = os.path.join("/", self.output_directory, self.name + "_ANNOTATED.json")
        print("New dataset with meta_annotations {}".format(self.file_path_for_json))
        self.mapping = None
        self.objects = self.dataset_orig.get_annotations_from_class_list(self.classes_to_annotate)
        self.max_pos = len(self.objects) - 1
        self.current_pos = 0
        self.mapping, self.dataset_annotated = self.__create_results_dict(self.file_path_for_json)

        self.fig_size = fig_size
        self.buttons_vertical = buttons_vertical

        if image_display_function is None:
            self.image_display_function = self.__show_image
        else:
            self.image_display_function = image_display_function

        # create buttons
        self.previous_button = self.__create_button("Previous", (self.current_pos == 0), self.__on_previous_clicked)
        self.next_button = self.__create_button("Next", (self.current_pos == self.max_pos), self.__on_next_clicked)
        self.save_button = self.__create_button("Download", False, self.__on_save_clicked)
        self.save_function = self.__save_function  # save_function
        self.current_image = {}
        buttons = [self.previous_button, self.next_button]
        buttons.append(self.save_button)

        label_total = Label(value='/ {}'.format(len(self.objects)))
        self.text_index = BoundedIntText(value=1, min=1, max=len(self.objects))
        self.text_index.layout.width = '80px'
        self.text_index.layout.height = '35px'
        self.text_index.observe(self.__selected_index)
        self.out = Output()
        self.out.add_class("my_canvas_class")

        self.checkboxes = {}
        self.radiobuttons = {}
        self.bounded_text = {}

        output_layout = []
        for k_name, v in self.metrics_and_values.items():
            if MetaPropertiesTypes.META_PROP_UNIQUE == v[1]:  # radiobutton
                self.radiobuttons[k_name] = RadioButtons(name=k_name, options=v[0],
                                                         disabled=False,
                                                         indent=False)
            elif MetaPropertiesTypes.META_PROP_COMPOUND == v[1]:  # checkbox
                self.checkboxes[k_name] = [Checkbox(False, description='{}'.format(prop_name),
                                                    indent=False, name=k_name) for prop_name in v[0]]
            elif MetaPropertiesTypes.META_PROP_CONTINUE == v[1]:
                self.bounded_text[k_name] = BoundedFloatText(value=v[0][0], min=v[0][0], max=v[0][1])

        self.check_radio_boxes_layout = {}

        for rb_k, rb_v in self.radiobuttons.items():
            rb_v.layout.width = '180px'
            rb_v.observe(self.__checkbox_changed)
            rb_v.add_class(rb_k)
            html_title = HTML(value="<b>" + rb_k + "</b>")
            self.check_radio_boxes_layout[rb_k] = VBox([rb_v])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout[rb_k]]))

        for cb_k, cb_i in self.checkboxes.items():
            for cb in cb_i:
                cb.layout.width = '180px'
                cb.observe(self.__checkbox_changed)
                cb.add_class(cb_k)
            html_title = HTML(value="<b>" + cb_k + "</b>")
            self.check_radio_boxes_layout[cb_k] = VBox(children=[cb for cb in cb_i])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout[cb_k]]))

        for bf_k, bf in self.bounded_text.items():
            bf.layout.width = '80px'
            bf.layout.height = '35px'
            bf.observe(self.__checkbox_changed)
            bf.add_class(bf_k)
            html_title = HTML(value="<b>" + bf_k + "</b>")
            self.check_radio_boxes_layout[bf_k] = VBox([bf])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout[bf_k]]))

        self.all_widgets = VBox(children=
                                [HBox([self.text_index, label_total]),
                                 HBox(buttons),
                                 HBox([self.out,
                                 VBox(output_layout)])])

        ## loading js library to perform html screenshots
        j_code = """
                require.config({
                    paths: {
                        html2canvas: "https://html2canvas.hertzen.com/dist/html2canvas.min"
                    }
                });
            """
        display(Javascript(j_code))

    def __add_annotation_to_mapping(self, ann):
        cat_name = self.dataset_orig.get_category_name_from_id(ann['category_id'])
        if ann['id'] not in self.mapping['annotated_ids']:  # o nly adds category counter if it was not annotate
            self.mapping['categories_counter'][cat_name] += 1

        self.mapping['annotated_ids'].add(ann['id'])

    def __create_results_dict(self, file_path):
        mapping = {}
        mapping["annotated_ids"] = set()  # key: object_id from dataset, values=[(annotations done)]
        mapping["categories_counter"] = dict.fromkeys([c for c in self.dataset_orig.get_categories_names()], 0)
        self.mapping = mapping

        if not os.path.exists("/" + file_path): #it does exist __ANNOTATED in the output directory
            with open(self.dataset_orig.dataset_root_param, 'r') as input_json_file:
                dataset_annotated = json.load(input_json_file)
                input_json_file.close()
            #take the same metaproperties already in file if it's not empty (like a new file)
            meta_prop = dataset_annotated['meta_properties'] if 'meta_properties' in dataset_annotated.keys() else []

            # adds the new annotations categories to dataset if it doesn't exist
            for k_name, v in self.metrics_and_values.items():
                new_mp_to_append = {
                    "name": k_name,
                    "type": v[1].value,
                    "values": [p for p in v[0]].sort()
                }

                names_prop_in_file = {m_p['name']: m_p for m_i, m_p in enumerate(dataset_annotated[
                                                                                     'meta_properties'])} if 'meta_properties' in dataset_annotated.keys() else None

                if 'meta_properties' not in dataset_annotated.keys():  # it is a new file
                    meta_prop.append(new_mp_to_append)
                    dataset_annotated['meta_properties'] = []

                elif names_prop_in_file is not None and k_name not in names_prop_in_file.keys():
                    # if there is a property with the same in meta_properties, it must be the same structure as the one proposed
                    meta_prop.append(new_mp_to_append)
                    self.__update_annotation_counter_and_current_pos(dataset_annotated)

                elif names_prop_in_file is not None and k_name in names_prop_in_file.keys() and \
                        names_prop_in_file[k_name] == new_mp_to_append:
                    #we don't append because it's already there
                    self.__update_annotation_counter_and_current_pos(dataset_annotated)

                else:
                    raise NameError("An annotation with the same name {} "
                                    "already exist in dataset {}, and it has different structure. Check properties.".format(
                        k_name, self.dataset_orig.dataset_root_param))

                #if k_name is in name_props_in_file and it's the same structure. No update is done.
            dataset_annotated['meta_properties'] = dataset_annotated['meta_properties'] + meta_prop
        else:
            with open(file_path, 'r') as classification_file:
                dataset_annotated = json.load(classification_file)
                classification_file.close()

            self.__update_annotation_counter_and_current_pos(dataset_annotated)
        return mapping, dataset_annotated

    def __update_annotation_counter_and_current_pos(self, dataset_annotated):
        prop_names = set(self.metrics_and_values.keys())
        last_ann_id = self.objects[0]['id']
        for ann in dataset_annotated['annotations']:
            if prop_names.issubset(
                    set(ann.keys())):  # we consider that it was annotated when all the props are subset of keys
                self.__add_annotation_to_mapping(ann)
                last_ann_id = ann['id']  # to get the last index in self.object so we can update the current pos in the last one anotated
            else:
                break
        self.current_pos = next(i for i, a in enumerate(self.objects) if a['id'] == last_ann_id)

    def __checkbox_changed(self, b):
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
        if self.metrics_and_values[annotation_name][1] in [MetaPropertiesTypes.META_PROP_COMPOUND]:
            if annotation_name not in ann.keys():
                ann[annotation_name] = {p: 0 for p in self.metrics_and_values[annotation_name][0]}
            ann[annotation_name][class_name] = int(value)
        else:  # UNIQUE VALUE
            ann[annotation_name] = value

    def __create_button(self, description, disabled, function):
        button = Button(description=description)
        button.disabled = disabled
        button.on_click(function)
        return button

    def __show_image(self, image_record, ann_key):
        #   read img from path and show it
        path_img = os.path.join(self.dataset_orig.images_abs_path, image_record['file_name'])
        img = Image.open(path_img)
        if self.show_name:
            print(os.path.basename(path_img) + '. Class: {} [class_id={}]'.format(
                self.dataset_orig.get_category_name_from_id(self.objects[self.current_pos]['category_id']),
                self.objects[self.current_pos]['category_id']))
        plt.figure(figsize=self.fig_size)

        if not self.show_axis:
            plt.axis('off')
        plt.imshow(img)

        # draw the bbox from the object onum
        ax = plt.gca()
        class_colors = cm.rainbow(np.linspace(0, 1, len(self.dataset_orig.get_categories_names())))

        annotation = self.dataset_orig.coco_lib.anns[ann_key]
        object_class_name = self.dataset_orig.get_category_name_from_id(annotation['category_id'])
        c = class_colors[self.dataset_orig.get_categories_names().index(object_class_name)]
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
        w.write(json.dumps(self.dataset_annotated))
        w.close()
        self.__add_annotation_to_mapping(next(ann_dat for ann_dat in self.dataset_annotated['annotations']
                                              if ann_dat['id'] == self.objects[self.current_pos]['id']))

    def __save_function(self, image_path):
        img_name = os.path.basename(image_path).split('.')[0]
        j_code = """
            require(["html2canvas"], function(html2canvas) {
                var element = $(".p-Widget.jupyter-widgets-output-area.output_wrapper.$it_name$")[0];
                console.log(element);
                 html2canvas(element).then(function (canvas) { 
                    var myImage = canvas.toDataURL(); 
                    var a = document.createElement("a"); 
                    a.href = myImage; 
                    a.download = "$img_name$.png"; 
                    a.click(); 
                    a.remove(); 
                });
            });
            """
        j_code = j_code.replace('$it_name$', "my_canvas_class")
        j_code = j_code.replace('$img_name$', img_name)
        tmp_out = Output()
        with tmp_out:
            display(Javascript(j_code))
            tmp_out.clear_output()

    def __perform_action(self):
        self.next_button.disabled = (self.current_pos == self.max_pos)
        self.previous_button.disabled = (self.current_pos == 0)

        current_class_id = self.objects[self.current_pos]['category_id']
        current_ann = next(
            a for a in self.dataset_annotated['annotations'] if a['id'] == self.objects[self.current_pos]['id'])
        # current_gt_id = self.objects[self.current_pos]['gt_id']

        for m_k, m_v in self.metrics_and_values.items():
            if m_v[1] == MetaPropertiesTypes.META_PROP_UNIQUE:  # radiobutton
                self.radiobuttons[m_k].unobserve(self.__checkbox_changed)
                self.radiobuttons[m_k].value = current_ann[m_k] if m_k in current_ann.keys() else None
                self.radiobuttons[m_k].observe(self.__checkbox_changed)
            elif m_v[1] == MetaPropertiesTypes.META_PROP_COMPOUND:  # checkbox
                for cb_i, cb_v in enumerate(self.checkboxes[m_k]):
                    cb_v.unobserve(self.__checkbox_changed)
                    cb_v.value = bool(current_ann[m_k][cb_v.description]) if m_k in current_ann.keys() else False
                    cb_v.observe(self.__checkbox_changed)
            elif m_v[1] == MetaPropertiesTypes.META_PROP_CONTINUE:  # textbound
                self.bounded_text[m_k].unobserve(self.__checkbox_changed)
                self.bounded_text[m_k].value = float(current_ann[m_k]) if m_k in current_ann.keys() else \
                    self.bounded_text[m_k].min
                self.bounded_text[m_k].observe(self.__checkbox_changed)

        with self.out:
            self.out.clear_output()
            image_record, ann_key = self.__get_image_record()
            self.image_display_function(image_record, ann_key)

        self.text_index.unobserve(self.__selected_index)
        self.text_index.value = self.current_pos + 1
        self.text_index.observe(self.__selected_index)

    def __get_image_record(self):
        # current_class_id = self.objects[self.current_pos]['category_id']
        current_image_id = self.objects[self.current_pos]['image_id']

        ann_key = self.objects[self.current_pos]['id']
        img_record = self.dataset_orig.coco_lib.imgs[current_image_id]
        return img_record, ann_key

    def __on_previous_clicked(self, b):
        self.save_state()
        self.current_pos -= 1
        self.__perform_action()

    def __on_next_clicked(self, b):
        self.save_state()
        self.current_pos += 1
        self.__perform_action()

    def __on_save_clicked(self, b):
        self.save_state()
        image_record, _ = self.__get_image_record()
        path_img = os.path.join(self.output_directory, 'JPEGImages', image_record['file_name'])
        self.save_function(path_img)

    def __selected_index(self, t):
        if t['owner'].value is None or t['name'] != 'value':
            return
        self.current_pos = t['new'] - 1
        self.__perform_action()

    def start_annotation(self):
        if self.max_pos < self.current_pos:
            print("No available images")
            return
        display(self.all_widgets)
        self.__perform_action()

    def print_statistics(self):
        table = []
        total = 0
        for c_k, c_number in self.mapping["categories_counter"].items():
            table.append([c_k, c_number])
            total += c_number
        table = sorted(table, key=lambda x: x[0])
        table.append(['Total', '{}/{}'.format(total, len(self.objects))])
        print(tabulate(table, headers=['Class name', 'Annotated objects']))
