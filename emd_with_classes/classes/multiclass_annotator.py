import os
import json
import glob
from PIL import Image
from matplotlib import pyplot as plt
from ipywidgets import Button, Output, HBox, VBox, Label, BoundedIntText, Checkbox, HTML, RadioButtons
from IPython.display import Javascript, display
from collections import defaultdict
from tabulate import tabulate
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import numpy as np


class MultiClassAnnotator:
    def __init__(self,
                 dataset_voc,
                 output_path_statistic,
                 name,
                 metrics,  # this is an array of the following ['occ', 'truncated', 'side', 'part']
                 show_name=True,
                 show_axis=False,
                 fig_size=(10, 10),
                 buttons_vertical=False,
                 image_display_function=None,
                 classes_to_annotate=None
                 ):

        if dataset_voc.annotations_gt is None:  # in case that dataset_voc has not been called
            dataset_voc.load()

        self.dataset_voc = dataset_voc
        self.metrics = metrics
        self.show_axis = show_axis
        self.name = name
        self.show_name = show_name
        if output_path_statistic is None:
            output_path_statistic = self.dataset_voc.dataset_root_param

        if classes_to_annotate is None:  # if classes_to_annotate is None, all the classes would be annotated
            self.classes_to_annotate = self.dataset_voc.objnames_all  # otherwise, the only the classes in the list

        self.output_path = output_path_statistic
        self.file_path = os.path.join(self.output_path, self.name + ".json")
        self.mapping, self.dataset = self.__create_results_dict(self.file_path, metrics)

        self.objects = dataset_voc.get_objects_index(self.classes_to_annotate)
        self.current_pos = 0

        self.max_pos = len(self.objects) - 1

        self.fig_size = fig_size
        self.buttons_vertical = buttons_vertical

        if image_display_function is None:
            self.image_display_function = self.__show_image
        else:
            self.image_display_function = image_display_function

        # create buttons
        self.previous_button = self.__create_button("Previous", (self.current_pos == 0), self.__on_previous_clicked)
        self.next_button = self.__create_button("Next", (self.current_pos == self.max_pos), self.__on_next_clicked)
        self.save_button = self.__create_button("Save", False, self.__on_save_clicked)
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
        self.out.add_class(name)

        metrics_labels = self.dataset_voc.read_label_metrics_name_from_file()
        metrics_labels['truncated'] = {'0': 'False', '1': 'True'}
        self.metrics_labels = metrics_labels

        self.checkboxes = {}
        self.radiobuttons = {}

        output_layout = []
        for m_i, m_n in enumerate(self.metrics):
            if 'parts' == m_n:  # Special case
                continue

            if m_n in ['truncated', 'occ']:  # radiobutton
                self.radiobuttons[m_n] = RadioButtons(options=[i for i in metrics_labels[m_n].values()],
                                                      disabled=False,
                                                      indent=False)
            else:  # checkbox
                self.checkboxes[m_n] = [Checkbox(False, description='{}'.format(metrics_labels[m_n][i]),
                                                 indent=False) for i in metrics_labels[m_n].keys()]

        self.check_radio_boxes_layout = {}
        for cb_k, cb_i in self.checkboxes.items():
            for cb in cb_i:
                cb.layout.width = '180px'
                cb.observe(self.__checkbox_changed)
            html_title = HTML(value="<b>" + cb_k + "</b>")
            self.check_radio_boxes_layout[cb_k] = VBox(children=[cb for cb in cb_i])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout[cb_k]]))

        for rb_k, rb_v in self.radiobuttons.items():
            rb_v.layout.width = '180px'
            rb_v.observe(self.__checkbox_changed)
            html_title = HTML(value="<b>" + rb_k + "</b>")
            self.check_radio_boxes_layout[rb_k] = VBox([rb_v])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout[rb_k]]))

        #create an output for the future dynamic SIDES_PARTS attributes
        self.dynamic_output_for_parts = Output()
        html_title = HTML(value="<b>" + "Parts" + "</b>")
        output_layout.append(VBox([html_title, self.dynamic_output_for_parts]))

        self.all_widgets = VBox(children=
                                [HBox([self.text_index, label_total]),
                                 HBox(buttons),
                                 HBox(output_layout),
                                 self.out])

        ## loading js library to perform html screenshots
        j_code = """
                require.config({
                    paths: {
                        html2canvas: "https://html2canvas.hertzen.com/dist/html2canvas.min"
                    }
                });
            """
        display(Javascript(j_code))

    def __create_results_dict(self, file_path, cc):
        mapping = {}
        # mapping["categories_id"] = {}
        # mapping["categories_name"] = {}
        # mapping["objects"] = {}
        #
        # if not os.path.exists(file_path):
        #     dataset = {}
        #     dataset['categories'] = []
        #     dataset["images"] = []
        #     for index, c in enumerate(cc):
        #         category = {}
        #         category["supercategory"] = c
        #         category["name"] = c
        #         category["id"] = index
        #         dataset["categories"].append(category)
        # else:
        #     with open(file_path, 'r') as classification_file:
        #         dataset = json.load(classification_file)
        #     for index, img in enumerate(dataset['images']):
        #         mapping['images'][img["path"]] = index
        #
        # for index, c in enumerate(dataset['categories']):
        #     mapping['categories_id'][c["id"]] = c["name"]
        #     mapping['categories_name'][c["name"]] = c["id"]
        # index_categories = len(dataset['categories']) - 1
        #
        # for c in cc:
        #     if not c in mapping['categories_name'].keys():
        #         mapping['categories_id'][index_categories] = c
        #         mapping['categories_name'][c] = index_categories
        #         category = {}
        #         category["supercategory"] = c
        #         category["name"] = c
        #         category["id"] = index_categories
        #         dataset["categories"].append(category)
        #         index_categories += 1

        return {}, {}

    def __checkbox_changed(self, b):
        if b['owner'].value is None or b['name'] != 'value':
            return

        class_name = b['owner'].description
        value = b['owner'].value

        current_index = self.mapping["images"][self.objects[self.current_pos]]
        class_index = self.mapping["categories_name"][class_name]
        if not class_index in self.dataset["images"][current_index]["categories"] and value:
            self.dataset["images"][current_index]["categories"].append(class_index)
        if class_index in self.dataset["images"][current_index]["categories"] and not value:
            self.dataset["images"][current_index]["categories"].remove(class_index)

    def __create_button(self, description, disabled, function):
        button = Button(description=description)
        button.disabled = disabled
        button.on_click(function)
        return button

    def __show_image(self, image_record, obj_num):
        #   read img from path and show it
        path_img = os.path.join(self.output_path, 'JPEGImages', image_record['filename'])
        img = Image.open(path_img)
        if self.show_name:
            print(os.path.basename(path_img) + '. Class: {} [class_id={}]'.format(
                self.objects[self.current_pos]['class_name'],
                self.objects[self.current_pos]['class_id']))
        plt.figure(figsize=self.fig_size)

        if not self.show_axis:
            plt.axis('off')
        plt.imshow(img)

        # draw the bbox from the object onum
        ax = plt.gca()
        class_colors = cm.rainbow(np.linspace(0, 1, len(self.dataset_voc.objnames_all)))

        [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = image_record['objects'][obj_num]['bbox']
        poly = [[bbox_x1, bbox_y1], [bbox_x1, bbox_y2], [bbox_x2, bbox_y2],
                [bbox_x2, bbox_y1]]
        np_poly = np.array(poly).reshape((4, 2))

        object_class_name = image_record['objects'][obj_num]['class']
        c = class_colors[self.dataset_voc.objnames_all.index(object_class_name)]

        # draws the bbox
        ax.add_patch(
            Polygon(np_poly, linestyle='-', facecolor=(c[0], c[1], c[2], 0.0),
                    edgecolor=(c[0], c[1], c[2], 1.0), linewidth=2))

        #  write the class name in bbox
        ax.text(x=bbox_x1, y=bbox_y1, s=object_class_name, color='white', fontsize=9, horizontalalignment='left',
                verticalalignment='top',
                bbox=dict(facecolor=(c[0], c[1], c[2], 0.5)))
        plt.show()

    def save_state(self):
        with open(self.file_path, 'w') as output_file:
            json.dump(self.dataset, output_file, indent=4)

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
        j_code = j_code.replace('$it_name$', self.name)
        j_code = j_code.replace('$img_name$', img_name)
        tmp_out = Output()
        with tmp_out:
            display(Javascript(j_code))
            tmp_out.clear_output()

    def __perform_action(self):
        # generates statistic saved in json file
        # print(self.objects[self.current_class])
        # if not self.objects[self.current_class][self.current_pos]['path'] in self.mapping["images"].keys():
        #     image = {}
        #     image["path"] = self.objects[self.current_class][self.current_pos]
        #     image["id"] = len(self.mapping["images"]) + 1
        #     image["categories"] = []
        #     self.dataset["images"].append(image)
        #     self.mapping["images"][image["path"]] = len(self.dataset["images"]) - 1
        # current_index = self.mapping["images"][self.objects[self.current_class][self.current_pos]]
        self.next_button.disabled = (self.current_pos == self.max_pos)
        self.previous_button.disabled = (self.current_pos == 0)

        current_class_id = self.objects[self.current_pos]['class_id']
        current_gt_id = self.objects[self.current_pos]['gt_id']

        # start to check for each type of metric
        if 'occ' in self.radiobuttons.keys():
            cb = self.radiobuttons['occ']
            rb_options = self.radiobuttons['occ'].options
            cb.unobserve(self.__checkbox_changed)
            cb.value = None #clear the current value
            if self.dataset_voc.annotations_gt['gt'][current_class_id]['details'][
                current_gt_id]:  # check if it is empty
                occ_level = self.dataset_voc.annotations_gt['gt'][current_class_id]['details'][current_gt_id][
                    'occ_level']
                cb.value = rb_options[occ_level - 1]
            cb.observe(self.__checkbox_changed)

        if 'truncated' in self.radiobuttons.keys(): #since this works for PASCAL VOC there's always a truncation value
            cb = self.radiobuttons['truncated']
            rb_options = self.radiobuttons['truncated'].options
            cb.unobserve(self.__checkbox_changed)
            cb.value = rb_options[
                int(self.dataset_voc.annotations_gt['gt'][current_class_id]['istrunc'][current_gt_id] == True)]
            cb.observe(self.__checkbox_changed)

        if 'views' in self.checkboxes.keys():
            for cb_i, cb in enumerate(self.checkboxes['views']):
                cb.unobserve(self.__checkbox_changed)
                cb.value = False #clear the value
                if self.dataset_voc.annotations_gt['gt'][current_class_id]['details'][current_gt_id]:
                    # check if it is empty
                    cb.value = bool(self.dataset_voc.annotations_gt['gt'][current_class_id]['details'][current_gt_id][
                                        'side_visible'][cb.description])
                cb.observe(self.__checkbox_changed)

        #need to create the output first for the buttons
        with self.dynamic_output_for_parts:
            self.dynamic_output_for_parts.clear_output()
            if self.objects[self.current_pos]['class_name'] in self.metrics_labels['parts']:
                self.cb_parts = [Checkbox(False, description='{}'.format(i), indent=False) for i in self.metrics_labels['parts'][self.objects[self.current_pos]['class_name']]]
            else:
                self.cb_parts = [HTML(value="No PARTS defined in Conf file")]
            display(VBox(children=[cb for cb in self.cb_parts]))

        with self.out:
            self.out.clear_output()
            image_record, obj_num = self.__get_image_record()
            self.image_display_function(image_record, obj_num)

        self.text_index.unobserve(self.__selected_index)
        self.text_index.value = self.current_pos + 1
        self.text_index.observe(self.__selected_index)

    def __get_image_record(self):
        current_class_id = self.objects[self.current_pos]['class_id']
        current_gt_id = self.objects[self.current_pos]['gt_id']

        obj_num = self.dataset_voc.annotations_gt['gt'][current_class_id]['onum'][current_gt_id]
        index_row = self.dataset_voc.annotations_gt['gt'][current_class_id]['rnum'][current_gt_id]
        r = self.dataset_voc.annotations_gt['rec'][index_row]
        return r, obj_num

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
        path_img = os.path.join(self.output_path, 'JPEGImages', image_record['filename'])
        self.save_function(path_img)

    def __selected_index(self, t):
        if t['owner'].value is None or t['name'] != 'value':
            return
        self.current_pos = t['new'] - 1
        self.__perform_action()

    def start_classification(self):
        if self.max_pos < self.current_pos:
            print("No available images")
            return
        display(self.all_widgets)
        self.__perform_action()

    def print_statistics(self):
        counter = defaultdict(int)
        for record in self.dataset["images"]:
            for c in record["categories"]:
                counter[c] += 1
        table = []
        for c in counter:
            table.append([self.mapping["categories_id"][c], counter[c]])
        table = sorted(table, key=lambda x: x[0])
        print(tabulate(table, headers=['Class name', 'Annotated images']))
