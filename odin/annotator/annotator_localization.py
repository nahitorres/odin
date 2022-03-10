import json
import os

import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as Poly
import matplotlib.cm as cm
import numpy as np
from IPython.core.display import display
from ipyleaflet import Map, projections, LocalTileLayer, DrawControl, Rectangle, ImageOverlay, Polygon, WidgetControl
from ipywidgets import HTML, Label, VBox, HBox, RadioButtons, Checkbox, BoundedFloatText, Textarea, Output, Button, \
    Layout
from tabulate import tabulate

from odin.annotator import MetaPropertiesType
from odin.classes import TaskType, DatasetLocalization
from odin.classes import strings as labels_str
from odin.classes.strings import err_type
from odin.annotator import AnnotatorInterface


class AnnotatorLocalization(AnnotatorInterface):
    supported_types = [TaskType.OBJECT_DETECTION,
                       TaskType.INSTANCE_SEGMENTATION]

    __mandatory_params_no_dataset = {'task_type', 'images', 'classes_to_annotate', 'output_path', 'ds_name'}

    def __init__(self,
                 dataset=None,
                 task_type=None,
                 images=None,
                 classes_to_annotate=None,
                 output_path=None,
                 ds_name=None,
                 properties_and_values=None,
                 show_name=True,
                 show_axis=False,
                 fig_size=(10, 10),
                 buttons_vertical=False,
                 custom_display_function=None,
                 validate_function=None,
                 show_reset=True):
        """
        The AnnotatorLocalization class can be used to annotate data sets for localization tasks, such as object detection
        (it is possible to draw and annotate bounding boxes) and instance segmentation (it is possible to draw and annotate segmentation masks).

        Parameters
        ----------
        dataset: DatasetLocalization, optional
            Dataset to be modified with annotations. If not specified, a new one is created. (default is None)
        task_type: TaskType, optional
            Problem task_type. If the dataset is not specified, it is a mandatory parameter. (default is None)
        images: list, optional
            List of images path. If the dataset is not specified, it is a mandatory parameter. (default is None)
        classes_to_annotate: list, optional
            List of categories to be annotated. If the dataset is not specified, it is a mandatory parameter. (default is None)
        output_path: str, optional
            The path where the annotated data set will be saved. If the dataset is not specified, it is a mandatory parameter. (default is None)
        ds_name: str, optional
            Name of the data set. If the dataset is not specified, it is a mandatory parameter. (default is None)
        properties_and_values: dict, optional
            Meta-annotations and corresponding values used to annotate the data set. If not specified, only the categeories will be annotated.
            (default is None)Example: {'property_name': (property_type, ['value1', 'value2', '...', 'valuen'])}
            The meta-annotations can be of the following types: MetaPropertiesType.UNIQUE, MetaPropertiesType.TEXT, MetaPropertiesType.CONTINUE
        show_name: bool, optional
            Indicates whether to show the name of the file. (default is True)
        show_axis: bool, optional
            Indicates whether to show the axis of the plot. (default is False)
        fig_size: pair, optional
            Indicates the size of the figure visualized during the annotation process. (default is (10, 10))
        buttons_vertical: bool, optional
            Indicates whether to display the buttons vertically. (default is False)
        custom_display_function: function, optional
            User custom display visualization. (default is None)
        validate_function: function, optional
            Annotation process validation function. (default is None)
        show_reset: bool, optional
            Indicates whether to show the 'reset annotation' button. (default is True)
        """
        if dataset is None:  # create new dataset

            if task_type is None or images is None or classes_to_annotate is None or output_path is None or ds_name is None:
                raise Exception(
                    f"Invalid parameters. Please be sure to specify the following parameters: {self.__mandatory_params_no_dataset}")

            if not isinstance(task_type, TaskType):
                raise TypeError(err_type.format("task_type"))
            elif task_type not in self.supported_types:
                raise Exception(labels_str.warn_task_not_supported)

            if not isinstance(images, list):
                raise TypeError(err_type.format("images"))
            elif len(images) == 0:
                raise Exception(labels_str.warn_no_images)

            if not isinstance(classes_to_annotate, list):
                raise TypeError(err_type.format("classes_to_annotate"))
            elif len(classes_to_annotate) <= 1:
                raise Exception(labels_str.warn_little_classes)

            self.images = images


        else:

            if type(dataset) is not DatasetLocalization:
                raise TypeError(f"Invalid dataset type: {type(dataset)}. Use DatasetLocalization.")

            if properties_and_values is None:
                raise Exception("Please be sure to specify the properties and the values")
            elif not isinstance(properties_and_values, dict):
                raise TypeError(err_type.format("properties_and_values"))

        self.objects = []
        self.current_img_id = 0
        self.selected_ann_id = None
        self.free_ann_ids = []
        self.last_ann_id = 0

        super().__init__(dataset, task_type, classes_to_annotate, output_path, ds_name, properties_and_values,
                         show_name,
                         show_axis, fig_size, buttons_vertical, custom_display_function, validate_function, show_reset)

        self.title_lbl = Label(value="title")

        if self.annotate_classes:
            # create drawing map
            self.__create_map()
            self.__disable_check_radio_boxes_value()
            self.all_widgets = VBox(
                [HBox([self.text_index, self.label_total]),
                 HBox(self.buttons),
                 self.title_lbl,
                 HBox([self.map,
                       VBox(self.output_layout)])])
        else:
            self.out = Output()
            self.out.add_class(self.name)
            self.all_widgets = VBox(
                [HBox([self.text_index, self.label_total]),
                 HBox(self.buttons),
                 self.validation_show,
                 self.title_lbl,
                 HBox([self.out,
                       VBox(self.output_layout)])])
            self._load_js()

    def _set_display_function(self, custom_display_function):
        self.image_display_function = self._show_image if custom_display_function is None else custom_display_function

    def _create_results_dict(self):
        """
        Create a dictionary to mapping the categories and annotations and create a dict for the dataset to be created
        Returns
        -------
            mapping, dataset
        """

        mapping = {"categories_id": {}, "categories_name": {}, "annotations": {}, "images": {}}
        dataset = {"categories": [], "annotations": [], "images": [], "meta_properties": []}

        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as classification_file:
                dataset = json.load(classification_file)
                classification_file.close()
        elif self.dataset_orig is not None:
            with open(self.dataset_orig.dataset_root_param, 'r') as input_json_file:
                dataset = json.load(input_json_file)
                input_json_file.close()

        max_img_id = 0
        for img in dataset["images"]:
            mapping["images"][img["file_name"]] = img["id"]
            mapping["images"][img["id"]] = img
            if img["id"] > max_img_id:
                max_img_id = img["id"]

        if self.annotate_classes:
            for path in self.images:
                filename = os.path.basename(path)
                if filename not in mapping["images"].keys():
                    if os.path.exists(path):
                        im = cv2.imread(path)
                        h, w, _ = im.shape
                        img = {"id": max_img_id + 1, "file_name": filename, "width": w, "height": h}
                    else:
                        img = {"id": max_img_id + 1, "file_name": filename}
                    mapping["images"][filename] = max_img_id + 1
                    dataset["images"].append(img)
                    max_img_id += 1

        for c in dataset['categories']:
            mapping['categories_id'][c["id"]] = c["name"]
            mapping['categories_name'][c["name"]] = c["id"]
        index_categories = len(dataset['categories']) + 1
        if self.annotate_classes:
            for c in self.classes_to_annotate:
                if c not in mapping['categories_name'].keys():
                    mapping['categories_id'][index_categories] = c
                    mapping['categories_name'][c] = index_categories
                    category = {"supercategory": c, "name": c, "id": index_categories}
                    dataset["categories"].append(category)
                    index_categories += 1
            self.objects = self.images
        else:
            self.objects = dataset["annotations"]

        if self.annotate_meta_properties:
            meta_prop = dataset['meta_properties'] if 'meta_properties' in dataset.keys() else []
            for k_name, v in self.properties_and_values.items():
                new_mp = {"name": k_name, "type": v[0].value}
                if len(v) > 1:
                    new_mp["values"] = sorted([p for p in v[1]])

                names_prop_in_file = {m_p['name']: m_p for m_i, m_p in enumerate(
                    dataset['meta_properties'])} if 'meta_properties' in dataset.keys() else None

                if 'meta_properties' not in dataset.keys():  # it is a new file
                    meta_prop.append(new_mp)
                    dataset['meta_properties'] = []

                elif names_prop_in_file is not None and k_name not in names_prop_in_file.keys():
                    # if there is a property with the same in meta_properties, it must be the same structure as the one proposed
                    meta_prop.append(new_mp)
                elif names_prop_in_file is not None and k_name in names_prop_in_file.keys() and \
                        names_prop_in_file[k_name] == new_mp:
                    # we don't append because it's already there
                    pass

                else:
                    raise NameError("An annotation with the same name {} "
                                    "already exist in dataset {}, and it has different structure. Check properties.".format(
                        k_name, self.dataset_orig.dataset_root_param))
            dataset['meta_properties'] = meta_prop

        return mapping, dataset

    def _create_buttons(self):
        """
        Create the widgets for the buttons
        Returns
        -------
            buttons
        """

        self.previous_button = self._create_button(labels_str.str_btn_prev, (self.current_pos == 0),
                                                   self._on_previous_clicked)
        self.next_button = self._create_button(labels_str.str_btn_next, (self.current_pos == self.max_pos),
                                               self._on_next_clicked)
        if not self.annotate_classes:
            self.save_button = self._create_button(labels_str.str_btn_download, False, self._on_save_clicked)
            return [self.previous_button, self.next_button, self.save_button]
        self.delete_button = self._create_button(labels_str.str_btn_delete_bbox, True, self.__on_delete_clicked)
        if self.show_reset:
            self.reset_button = self._create_button(labels_str.str_btn_reset, False, self._on_reset_clicked)
            return [self.previous_button, self.next_button, self.delete_button, self.reset_button]
        return [self.previous_button, self.next_button, self.delete_button]

    def _on_save_clicked(self, b):
        """
        Function called when the 'download' button is clicked
        """

        self.save_state()
        self.save_function(self.mapping["images"][self.current_img_id]["file_name"], self.current_pos)

    def _on_reset_clicked(self, b):
        """
        Function called when the 'reset' button is clicked
        """
        if self.annotate_classes:
            for ann_id in list(self.mapping["annotations"].keys()):
                self.map.remove_layer(self.mapping["annotations"][ann_id]["polygon"])
                del self.mapping["annotations"][ann_id]
                self.free_ann_ids.append(ann_id)
            self.selected_ann_id = None
            self.delete_button.disabled = True
            self.__disable_check_radio_boxes_value()
        else:
            for m_k, m_v in self.properties_and_values.items():
                if m_k in self.mapping["annotations"][self.selected_ann_id]:
                    del self.mapping["annotations"][self.selected_ann_id][m_k]
            self.__change_check_radio_boxes_value()
            self._execute_validation()

    def _create_check_radio_boxes(self):
        """
        Create the widgets for the annotation
        """

        labels = dict()

        if self.annotate_classes:
            labels["categories"] = self.classes_to_annotate
            self.radiobuttons["categories"] = RadioButtons(name="categories", options=self.classes_to_annotate,
                                                           disabled=False, indent=False)

        if self.annotate_meta_properties:
            for k_name, v in self.properties_and_values.items():
                if len(v) == 3:
                    label = v[2]
                elif MetaPropertiesType.TEXT.value == v[0].value and len(v) == 2:
                    label = v[1]
                else:
                    label = k_name

                labels[k_name] = label

                if MetaPropertiesType.UNIQUE.value == v[0].value:  # radiobutton
                    self.radiobuttons[k_name] = RadioButtons(name=k_name, options=v[1],
                                                             disabled=False,
                                                             indent=False)
                elif MetaPropertiesType.COMPOUND.value == v[0].value:  # checkbox
                    self.checkboxes[k_name] = [Checkbox(False, indent=False, name=k_name, disabled=False,
                                                        description=prop_name) for prop_name in v[1]]
                elif MetaPropertiesType.CONTINUE.value == v[0].value:
                    self.bounded_text[k_name] = BoundedFloatText(disabled=False, value=v[1][0], min=v[1][0],
                                                                 max=v[1][1])

                elif MetaPropertiesType.TEXT.value == v[0].value:
                    self.box_text[k_name] = Textarea(disabled=False)
        return labels

    def _set_check_radio_boxes_layout(self, labels):
        """
        Initialize the widget for the annotations
        Parameters
        ----------
        labels
            meta-properties names

        Returns
        -------
            output_layout
        """

        output_layout = []

        if self.annotate_classes:
            self.radiobuttons["categories"].layout.width = '180px'
            self.radiobuttons["categories"].observe(self._checkbox_changed)
            self.radiobuttons["categories"].add_class("categories")
            html_title = HTML(value="<b>Categories</b>")
            self.check_radio_boxes_layout["categories"] = VBox([self.radiobuttons["categories"]])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout["categories"]]))

        if self.annotate_meta_properties:
            for rb_k, rb_v in self.radiobuttons.items():
                if rb_k == "categories":
                    continue
                rb_v.layout.width = '180px'
                rb_v.observe(self._checkbox_changed)
                rb_v.add_class(rb_k)
                html_title = HTML(value="<b>" + labels[rb_k] + "</b>")
                self.check_radio_boxes_layout[rb_k] = VBox([rb_v])
                output_layout.append(VBox([html_title, self.check_radio_boxes_layout[rb_k]]))

            for cb_k, cb_i in self.checkboxes.items():
                if cb_k == "categories":
                    continue
                for cb in cb_i:
                    cb.layout.width = '180px'
                    cb.observe(self._checkbox_changed)
                    cb.add_class(cb_k)
                html_title = HTML(value="<b>" + labels[cb_k] + "</b>")
                self.check_radio_boxes_layout[cb_k] = VBox(children=[cb for cb in cb_i])
                output_layout.append(VBox([html_title, self.check_radio_boxes_layout[cb_k]]))

            for bf_k, bf in self.bounded_text.items():
                bf.layout.width = '80px'
                bf.layout.height = '35px'
                bf.observe(self._checkbox_changed)
                bf.add_class(bf_k)
                html_title = HTML(value="<b>" + labels[bf_k] + "</b>")
                self.check_radio_boxes_layout[bf_k] = VBox([bf])
                output_layout.append(VBox([html_title, self.check_radio_boxes_layout[bf_k]]))

            for tb_k, tb_i in self.box_text.items():
                tb_i.layout.width = '500px'
                tb_i.observe(self._checkbox_changed)
                tb_i.add_class(tb_k)
                html_title = HTML(value="<b>" + labels[tb_k] + "</b>")
                self.check_radio_boxes_layout[tb_k] = VBox([tb_i])
                output_layout.append(VBox([html_title, self.check_radio_boxes_layout[tb_k]]))

        return output_layout

    def __change_check_radio_boxes_value(self):
        """
        update the values of the widget based on the curent annotation
        Parameters
        ----------
        current_ann: dict
            current annotation
        """

        if self.selected_ann_id is None:
            return
        current_ann = self.mapping["annotations"][self.selected_ann_id]

        if self.annotate_classes:
            self.delete_button.disabled = False
            self.radiobuttons["categories"].unobserve(self._checkbox_changed)
            self.radiobuttons["categories"].disabled = False
            self.radiobuttons["categories"].value = self.mapping["categories_id"][
                current_ann["category_id"]] if "category_id" in current_ann.keys() else None
            self.radiobuttons["categories"].observe(self._checkbox_changed)
        if self.annotate_meta_properties:
            for m_k, m_v in self.properties_and_values.items():
                if m_v[0].value == MetaPropertiesType.UNIQUE.value:  # radiobutton
                    self.radiobuttons[m_k].unobserve(self._checkbox_changed)
                    self.radiobuttons[m_k].value = current_ann[m_k] if m_k in current_ann.keys() else None
                    self.radiobuttons[m_k].disabled = False
                    self.radiobuttons[m_k].observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.COMPOUND.value:  # checkbox
                    for cb_i, cb_v in enumerate(self.checkboxes[m_k]):
                        cb_v.unobserve(self._checkbox_changed)
                        if m_k in current_ann.keys():
                            if cb_v.description in current_ann[m_k].keys():
                                cb_v.value = current_ann[m_k][cb_v.description]
                            else:
                                cb_v.value = False
                        else:
                            cb_v.value = False
                        cb_v.disabled = False
                        cb_v.observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.CONTINUE.value:  # textbound
                    self.bounded_text[m_k].unobserve(self._checkbox_changed)
                    self.bounded_text[m_k].value = float(current_ann[m_k]) if m_k in current_ann.keys() else \
                        self.bounded_text[m_k].min
                    self.bounded_text[m_k].disabled = False
                    self.bounded_text[m_k].observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.TEXT.value:  # text
                    self.box_text[m_k].unobserve(self._checkbox_changed)
                    self.box_text[m_k].value = current_ann[m_k] if m_k in current_ann.keys() else ""
                    self.box_text[m_k].disabled = False
                    self.box_text[m_k].observe(self._checkbox_changed)

    def _checkbox_changed(self, b):
        """
        Function called when the widget value change
        """

        if b['owner'].value is None or b['name'] != 'value':
            return

        class_name = b['owner'].description
        value = b['owner'].value
        annotation_name = b['owner']._dom_classes[0]

        if annotation_name == "categories":
            class_index = self.mapping["categories_name"][value]
            self.mapping["annotations"][self.selected_ann_id]["category_id"] = class_index
        else:
            if self.properties_and_values[annotation_name][0].value in [MetaPropertiesType.COMPOUND.value]:
                if annotation_name not in self.mapping["annotations"][self.selected_ann_id].keys():
                    self.mapping["annotations"][self.selected_ann_id][annotation_name] = {p: False for p in
                                                                                          self.properties_and_values[
                                                                                              annotation_name][1]}
                self.mapping["annotations"][self.selected_ann_id][annotation_name][class_name] = value
            else:
                self.mapping["annotations"][self.selected_ann_id][annotation_name] = value
        self._execute_validation()
        if self.current_pos == self.max_pos:
            self.save_state()

    def save_state(self):
        """
        Create the json file for the dataset
        """
        if self.dataset_orig is None:
            self.__save_state_class_annotations()
        else:
            self.__save_state_meta_annotations()

    def __save_state_meta_annotations(self):
        """
        Create the json file for the dataset when only meta-annotations are added
        """
        self.dataset_annotated["annotations"][self.current_pos] = self.mapping["annotations"][self.selected_ann_id]
        with open(self.file_path, 'w') as f:
            json.dump(self.dataset_annotated, f, indent=4)

    def _show_image(self, image_record, ann_key):
        """
        Display the selected image
        Parameters
        ----------
        image_record: dict
            image to display
        """

        #   read img from path and show it
        path_img = os.path.join(self.dataset_orig.images_abs_path, image_record['file_name'])
        img = Image.open(path_img)
        if self.show_name:
            self.title_lbl.value = image_record['file_name'].split('.')[0]
        plt.figure(figsize=self.fig_size)

        if not self.show_axis:
            plt.axis('off')
        plt.imshow(img)

        # draw the bbox from the object onum
        ax = plt.gca()
        class_colors = cm.rainbow(np.linspace(0, 1, len(self.dataset_orig.get_categories_names())))

        annotation = self.dataset_orig.get_annotation_from_id(ann_key)
        object_class_name = self.dataset_orig.get_category_name_from_id(int(annotation['category_id']))
        c = class_colors[list(self.dataset_orig.get_categories_names()).index(object_class_name)]
        if not self.dataset_orig.task_type == TaskType.INSTANCE_SEGMENTATION:
            [bbox_x1, bbox_y1, diff_x, diff_y] = annotation['bbox']
            bbox_x2 = bbox_x1 + diff_x
            bbox_y2 = bbox_y1 + diff_y
            poly = [[bbox_x1, bbox_y1], [bbox_x1, bbox_y2], [bbox_x2, bbox_y2],
                    [bbox_x2, bbox_y1]]
            np_poly = np.array(poly).reshape((4, 2))

            # draws the bbox
            ax.add_patch(
                Poly(np_poly, linestyle='-', facecolor=(c[0], c[1], c[2], 0.0),
                     edgecolor=(c[0], c[1], c[2], 1.0), linewidth=2))
        else:
            seg_points = annotation['segmentation']
            for pol in seg_points:
                poly = [[float(pol[i]), float(pol[i + 1])] for i in range(0, len(pol), 2)]
                np_poly = np.array(poly)  # .reshape((len(pol), 2))
                ax.add_patch(
                    Poly(np_poly, linestyle='-', facecolor=(c[0], c[1], c[2], 0.25),
                         edgecolor=(c[0], c[1], c[2], 1.0), linewidth=2))
            # set the first XY point for printing the text
            bbox_x1 = seg_points[0][0]
            bbox_y1 = seg_points[0][1]

        #  write the class name in bbox
        ax.text(x=bbox_x1, y=bbox_y1, s=object_class_name, color='white', fontsize=9, horizontalalignment='left',
                verticalalignment='top',
                bbox=dict(facecolor=(c[0], c[1], c[2], 0.5)))
        plt.show()

    def _perform_action(self):
        """
        Update the dataset and the mapping paramenters based on the annotation and clear the output to display a new
        image
        """

        self.next_button.disabled = (self.current_pos == self.max_pos)
        self.previous_button.disabled = (self.current_pos == 0)

        self.mapping["annotations"] = {}
        self.selected_ann_id = None

        if self.annotate_classes:
            self.__clear_map()
            self.__show_image_to_annotate()
            self.__disable_check_radio_boxes_value()
            self.delete_button.disabled = True
        else:
            with self.out:
                self.out.clear_output()
                self.selected_ann_id = self.dataset_annotated["annotations"][self.current_pos]["id"]
                self.current_img_id = self.dataset_annotated["annotations"][self.current_pos]["image_id"]
                self.image_display_function(self.mapping["images"][self.current_img_id],
                                            self.selected_ann_id)
                self.mapping["annotations"][self.selected_ann_id] = self.dataset_annotated["annotations"][
                    self.current_pos]
                self.__change_check_radio_boxes_value()
                self._execute_validation()

        self.text_index.unobserve(self._selected_index)
        self.text_index.value = self.current_pos + 1
        self.text_index.observe(self._selected_index)

    def _execute_validation(self):
        """
        Validate the actual status of the annotation
        """

        if self.validate:
            if self.validate_function(self.mapping["annotations"][self.selected_ann_id]):
                self.validation_show.value = labels_str.srt_validation_not_ok
            else:
                self.validation_show.value = labels_str.srt_validation_ok

    def __create_map(self):
        """
        Create a layer Map to display the image in order to be able to draw annotations
        """

        self.map = Map(center=(50, 50), zoom=2, crs=projections.Simple, dragging=True,
                       zoom_control=True, double_click_zoom=True,
                       layers=[LocalTileLayer(path='white.png')], layout=dict(width='600px', height='600px'))

        button = Button(
            disabled=False,
            button_style='',
            icon='arrows-alt',
            layout=Layout(width='25px', height='25px')
        )

        def function(b):
            self.map.center = center = (50, 50)
            self.map.zoom = 2

        self.__create_draw_control()
        button.on_click(function)
        recenter_control = WidgetControl(widget=button, position='topleft')
        self.map.add_control(recenter_control)

    def __on_delete_clicked(self, b):
        """
        Function called when the 'delete' button si clicked

        """
        if self.selected_ann_id in self.mapping["annotations"].keys():
            self.map.remove_layer(self.mapping["annotations"][self.selected_ann_id]["polygon"])
            del self.mapping["annotations"][self.selected_ann_id]
        self.free_ann_ids.append(self.selected_ann_id)
        self.selected_ann_id = None
        self.delete_button.disabled = True
        self.__disable_check_radio_boxes_value()

    def __clear_map(self, keep_img_overlay=False):
        """
        Remove all the layer from the map
        Parameters
        ----------
        keep_img_overlay: bool, optional
            if True, keep the image
        """
        starting_layer = 2 if keep_img_overlay else 1
        for l in self.map.layers[starting_layer:]:
            self.map.remove_layer(l)

    def __create_draw_control(self):
        """
        Create the control to draw the annotations for the object detection and instance segmentation tasks

        """
        if self.task_type == TaskType.OBJECT_DETECTION:
            dc = DrawControl(rectangle={'shapeOptions': {'color': '#0000FF'}}, circle={}, circlemarker={}, polyline={},
                             marker={}, polygon={})
        else:
            dc = DrawControl(rectangle={}, circle={}, circlemarker={}, polyline={},
                             marker={}, polygon={'shapeOptions': {'color': '#0000FF'}})
        dc.edit = False
        dc.remove = False

        def handle_bbox_draw(target, action, geo_json):
            if action == "created" and geo_json["geometry"]["type"] == "Polygon":
                coordinates = geo_json['geometry']['coordinates'][0]
                hs = [c[1] for c in coordinates]
                ws = [c[0] for c in coordinates]
                min_h, max_h = min(hs), max(hs)
                min_w, max_w = min(ws), max(ws)

                # coordinates only inside image
                hh, ww, offset_h, offset_w = self.img_coords[2:]
                max_h = max(0, min(hh + offset_h, max_h))
                max_w = max(0, min(ww + offset_w, max_w))
                min_h = max(offset_h, min(hh + offset_h, min_h))
                min_w = max(offset_w, min(ww + offset_w, min_w))

                # remove draw
                dc.clear()

                if max_h - min_h < 1 or max_w - min_w < 1:
                    print(labels_str.warn_skip_wrong)
                    return
                rectangle = self.__create_rectangle(((min_h, min_w), (max_h, max_w)))
                ann_id = self.__get_available_id()
                ann = {"id": ann_id,
                       "image_id": self.current_img_id,
                       "polygon": rectangle}

                self.mapping["annotations"][ann["id"]] = ann
                self.map.add_layer(rectangle)

                # automatically select last annotation
                self.selected_ann_id = ann_id
                self.__reset_colors_annotations()
                self.__change_check_radio_boxes_value()

        def handle_segmentation_draw(target, action, geo_json):
            if action == "created" and geo_json["geometry"]["type"] == "Polygon":
                coordinates = geo_json['geometry']['coordinates'][0]
                hs = [c[1] for c in coordinates]
                ws = [c[0] for c in coordinates]
                min_h, max_h = min(hs), max(hs)
                min_w, max_w = min(ws), max(ws)
                # coordinates only inside image
                hh, ww, offset_h, offset_w = self.img_coords[2:]
                max_h = max(0, min(hh + offset_h, max_h))
                max_w = max(0, min(ww + offset_w, max_w))
                min_h = max(offset_h, min(hh + offset_h, min_h))
                min_w = max(offset_w, min(ww + offset_w, min_w))
                # remove draw
                dc.clear()

                if max_h - min_h < 1 or max_w - min_w < 1:
                    print(labels_str.warn_skip_wrong)
                    return

                coor = []
                for c in coordinates:
                    w_c, h_c = c
                    # h_c = max(0, min(hh + offset_h, h_c))
                    h_c = max(offset_h, min(hh + offset_h, h_c))
                    # w_c = max(0, min(ww + offset_w, w_c))
                    w_c = max(offset_w, min(ww + offset_w, w_c))
                    coor.append((h_c, w_c))
                segmentation = self.__create_segmentation(coor)
                ann_id = self.__get_available_id()
                ann = {"id": ann_id,
                       "image_id": self.current_img_id,
                       "polygon": segmentation}

                self.mapping["annotations"][ann["id"]] = ann
                self.map.add_layer(segmentation)
                # automatically select last annotation
                self.selected_ann_id = ann_id
                self.__reset_colors_annotations()
                self.__change_check_radio_boxes_value()

        if self.task_type == TaskType.OBJECT_DETECTION:
            dc.on_draw(handle_bbox_draw)
        else:
            dc.on_draw(handle_segmentation_draw)
        self.map.add_control(dc)

    def __disable_check_radio_boxes_value(self):
        """
        Disable all the widgets
        """
        if self.annotate_classes:
            self.radiobuttons["categories"].unobserve(self._checkbox_changed)
            self.radiobuttons["categories"].value = None
            self.radiobuttons["categories"].disabled = True
            self.radiobuttons["categories"].observe(self._checkbox_changed)
        if self.annotate_meta_properties:
            for m_k, m_v in self.properties_and_values.items():
                if m_v[0].value == MetaPropertiesType.UNIQUE.value:  # radiobutton
                    self.radiobuttons[m_k].unobserve(self._checkbox_changed)
                    self.radiobuttons[m_k].value = None
                    self.radiobuttons[m_k].disabled = True
                    self.radiobuttons[m_k].observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.COMPOUND.value:  # checkbox
                    for cb_i, cb_v in enumerate(self.checkboxes[m_k]):
                        cb_v.unobserve(self._checkbox_changed)
                        cb_v.value = None
                        cb_v.disabled = True
                        cb_v.observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.CONTINUE.value:  # textbound
                    self.bounded_text[m_k].unobserve(self._checkbox_changed)
                    self.bounded_text[m_k].value = self.bounded_text[m_k].min
                    self.bounded_text[m_k].disabled = True
                    self.bounded_text[m_k].observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.TEXT.value:  # text
                    self.box_text[m_k].unobserve(self._checkbox_changed)
                    self.box_text[m_k].value = ""
                    self.box_text[m_k].disabled = True
                    self.box_text[m_k].observe(self._checkbox_changed)

    def __create_rectangle(self, bounds):
        """
        Create a Rectangle to be displayed on the image
        Parameters
        ----------
        bounds:
            Rectange bounds

        Returns
        -------
            Rectangle

        """
        rectangle = Rectangle(bounds=bounds, color="red", fill_color="red")
        rectangle.on_click(self.__handle_click)
        return rectangle

    def __create_segmentation(self, bounds):
        """
        Create a polygon to be displayed on the image
        Parameters
        ----------
        bounds:
            Polygon bounds

        Returns
        -------
            Polygon
        """
        segmentation = Polygon(locations=bounds, color="red", fill_color="red")
        segmentation.on_click(self.__handle_click)
        return segmentation

    def __handle_click(self, **kwargs):
        """
        Function to handle the user click on an annotation
        """
        if kwargs.get('type') == 'click':
            click_coords = kwargs.get('coordinates')
            clicked_ann = None
            # find clicked annotations (can be more than 1 if overlapping)
            clicked_size = None
            for idx, ann in enumerate(self.mapping["annotations"].values()):
                if self.task_type == TaskType.OBJECT_DETECTION:
                    coordinates = ann["polygon"].bounds
                else:
                    coordinates = ann["polygon"].locations
                    # rectangle opposite coordinates wrt geojson
                hs = [c[0] for c in coordinates]
                ws = [c[1] for c in coordinates]
                min_h, max_h = min(hs), max(hs)
                min_w, max_w = min(ws), max(ws)
                # don't break so if two rectangles are overlapping I take only the last drawed
                if min_h <= click_coords[0] <= max_h and min_w <= click_coords[1] <= max_w:
                    curr_size = (max_h - min_h) * (max_w - min_w)
                    if clicked_size is None or curr_size < clicked_size:
                        clicked_size = curr_size
                        clicked_ann = ann["id"]
            if clicked_ann is not None:
                self.selected_ann_id = clicked_ann
                self.__reset_colors_annotations()
                self.__change_check_radio_boxes_value()

    def __get_available_id(self):
        """
        Returns an available annotation id
        Returns
        -------
        int
        """
        if len(self.free_ann_ids) > 0:
            ann_id = self.free_ann_ids.pop(0)
        else:
            ann_id = self.last_ann_id + 1
            self.last_ann_id += 1
        return ann_id

    def __reset_colors_annotations(self):
        """
        Reset the colors of the annotations on the image
        """
        for idx, ann_id in enumerate(self.mapping["annotations"].keys()):
            if self.selected_ann_id is not None and self.selected_ann_id == ann_id:
                self.map.layers[idx + 2].color = "blue"
                self.map.layers[idx + 2].fill_color = "blue"
            elif "category_id" not in self.mapping["annotations"][ann_id].keys():
                self.map.layers[idx + 2].color = "red"
                self.map.layers[idx + 2].fill_color = "red"
            else:
                self.map.layers[idx + 2].color = "green"
                self.map.layers[idx + 2].fill_color = "green"

    def __get_img_overlay(self, img_path):
        """
        Returns the paramenter of the image shown
        Parameters
        ----------
        img_path: str
            image path

        Returns
        -------
        img_ov: ImageOverlay
        h: image height
        w: image width
        hh
        ww
        offset_h
        offset_w
        """
        # We need to create an ImageOverlay for each image to show,
        # and set the appropriate bounds based  on the image size

        if not os.path.exists(img_path):
            print(labels_str.warn_img_path_not_exits + img_path)

        im = cv2.imread(img_path)
        h, w, _ = im.shape

        max_v = 100

        offset_h = -25
        offset_w = -25
        hh = max_v - offset_h * 2
        ww = max_v - offset_w * 2

        if h > w:
            ww = int(w * hh / h)
            offset_w = (max_v - ww) / 2
        elif w > h:
            hh = int(h * ww / w)
            offset_h = (max_v - hh) / 2

        img_ov = ImageOverlay(url=img_path, bounds=((offset_h, offset_w), (hh + offset_h, ww + offset_w)))
        return img_ov, h, w, hh, ww, offset_h, offset_w

    def __show_image_to_annotate(self):
        """
        Show current image
        """
        img_ov, h, w, hh, ww, off_h, off_w = self.__get_img_overlay(self.images[self.current_pos])
        self.img_coords = (h, w, hh, ww, off_h, off_w)
        self.last_ann_id = max(self.dataset_annotated["annotations"], key=lambda x: x['id'])['id'] if len(
            self.dataset_annotated["annotations"]) > 0 else 0
        self.current_img_id = self.mapping["images"][os.path.basename(self.images[self.current_pos])]
        self.map.add_layer(img_ov)
        self.__show_existing_annotations()
        if self.show_name:
            self.title_lbl.value = os.path.basename(self.images[self.current_pos]).split(".")[0]

    def __show_existing_annotations(self):
        """
        Show the existing annotations on the image
        """
        # resize coords to fit in map
        h_i, w_i, hh, ww, offset_h, offset_w = self.img_coords[:]

        for ann in self.dataset_annotated["annotations"]:
            if ann["image_id"] == self.current_img_id:
                if self.task_type == TaskType.OBJECT_DETECTION and "bbox" in ann:
                    coordinates = self.__get_coordinates_from_image_bbox(ann["bbox"], h_i, w_i, hh, ww, offset_h,
                                                                         offset_w)
                    rectangle = self.__create_rectangle(coordinates)
                    rectangle.color = "green"
                    rectangle.fill_color = "green"
                    ann["polygon"] = rectangle

                    self.map.add_layer(rectangle)

                elif self.task_type == TaskType.INSTANCE_SEGMENTATION and "segmentation" in ann:
                    coordinates = self.__get_coordinates_from_image_segmentation(ann["segmentation"][0], h_i, w_i, hh,
                                                                                 ww, offset_h, offset_w)
                    polygon = self.__create_segmentation(coordinates)
                    polygon.color = "green"
                    polygon.fill_color = "green"
                    ann["polygon"] = polygon

                    self.map.add_layer(polygon)

                if "polygon" in ann.keys():
                    self.mapping["annotations"][ann["id"]] = ann

    def __save_state_class_annotations(self):
        """
        Create the json dataset file
        """
        self.dataset_annotated["annotations"] = list(
            filter(lambda x: x['image_id'] != self.current_img_id, self.dataset_annotated["annotations"]))
        self.dataset_annotated["images"] = list(
            filter(lambda x: x['id'] != self.current_img_id, self.dataset_annotated["images"]))

        h, w, hh, ww, off_h, off_w = self.img_coords[:]
        for ann_id in self.mapping["annotations"].keys():
            ann = self.mapping["annotations"][ann_id]
            coordinates = ann["polygon"].bounds if self.task_type == TaskType.OBJECT_DETECTION else ann[
                "polygon"].locations

            ann_info = {}
            for k in ann.keys():
                if k == "polygon":
                    if self.task_type == TaskType.OBJECT_DETECTION:
                        bbox = self.__get_image_bbox_from_coordinates(coordinates, h, w, hh, ww, off_h, off_w)
                        ann_info["bbox"] = bbox
                    else:
                        segmentation = self.__get_image_segmentation_from_coordinates(coordinates, h, w, hh, ww, off_h,
                                                                                      off_w)
                        ann_info["segmentation"] = [segmentation]
                else:
                    ann_info[k] = ann[k]
            self.dataset_annotated["annotations"].append(ann_info)

        img_info = {
            "id": self.current_img_id,
            "file_name": os.path.basename(self.images[self.current_pos]),
            "width": w,
            "height": h}
        self.dataset_annotated["images"].append(img_info)

        with open(self.file_path, 'w') as f:
            json.dump(self.dataset_annotated, f, indent=4)

    def __get_coordinates_from_image_bbox(self, bbox, h_i, w_i, hh, ww, offset_h, offset_w):
        """
        Get the coordinates of the rectangle from the bounding box

        Returns
        -------
            coordinates
        """
        # create rectangle layer
        min_w, min_h, w, h = bbox
        max_w = min_w + w
        max_h = min_h + h

        h_ratio = hh / h_i
        w_ratio = ww / w_i

        min_h = h_i - min_h
        max_h = h_i - max_h
        # coords to map coords
        min_h = min_h * h_ratio + offset_h
        max_h = max_h * h_ratio + offset_h
        min_w = min_w * w_ratio + offset_w
        max_w = max_w * w_ratio + offset_w

        coordinates = [(min_h, min_w), (max_h, max_w)]
        return coordinates

    def __get_coordinates_from_image_segmentation(self, segmentation, h_i, w_i, hh, ww, offset_h, offset_w):
        """
        Get the coordinates of the polygon from the segmentation points
        Returns
        -------
            coordinates
        """
        coordinates = []
        for i in range(0, len(segmentation), 2):
            s_w = segmentation[i]
            s_h = segmentation[i + 1]

            h_ratio = hh / h_i
            w_ratio = ww / w_i

            s_h = h_i - s_h
            s_h = s_h * h_ratio + offset_h
            s_w = s_w * w_ratio + offset_w

            coordinates.append([s_h, s_w])
        return coordinates

    def __get_image_bbox_from_coordinates(self, coordinates, h, w, hh, ww, off_h, off_w):
        """
        Get the bounding box from the coordinates of the rectangle
        Returns
        -------
            bbox
        """
        # rectangle opposite coordinates wrt geojson
        hs = [c[0] for c in coordinates]
        ws = [c[1] for c in coordinates]
        min_h, max_h = max(hs), min(hs)
        min_w, max_w = min(ws), max(ws)

        h_ratio = h / hh
        w_ratio = w / ww
        # map coords to img coords
        min_h = (min_h - off_h) * h_ratio
        max_h = (max_h - off_h) * h_ratio
        min_w = (min_w - off_w) * w_ratio
        max_w = (max_w - off_w) * w_ratio

        min_h = h - min_h
        max_h = h - max_h

        bbox = [min_w, min_h, max_w - min_w, max_h - min_h]
        return bbox

    def __get_image_segmentation_from_coordinates(self, coordinates, h, w, hh, ww, off_h, off_w):
        """
        Get the segmentation from the coordinates of the polygon
        Returns
        -------
            segmentation
        """
        segmentation = []
        for c in coordinates:
            c_h, c_w = c
            h_ratio = h / hh
            w_ratio = w / ww

            c_h = (c_h - off_h) * h_ratio
            c_w = (c_w - off_w) * w_ratio

            c_h = h - c_h

            segmentation.extend([c_w, c_h])
        return segmentation

    def start_annotation(self):
        """
        Start the annotation process
        """
        if self.max_pos < self.current_pos:
            print("No available observation")
            return
        display(self.all_widgets)
        self._perform_action()

    def print_statistics(self):
        """
        Print the statistic related to the annotations
        """
        if not self.annotate_classes:
            counter = dict()
            for c in self.mapping["categories_id"]:
                counter[c] = 0
            total = 0
            for record in self.dataset_annotated["annotations"]:
                if "category_id" in record:
                    skip = False
                    for k_name, v in self.properties_and_values.items():
                        if k_name not in record:
                            skip = True
                            break
                    if not skip:
                        counter[record["category_id"]] += 1
                        total += 1

            table = []
            for c in counter:
                table.append([self.mapping["categories_id"][c], counter[c]])
            table = sorted(table, key=lambda x: x[0])
            table.append(['Total', '{}/{}'.format(total, len(self.dataset_annotated["annotations"]))])

            print(tabulate(table, headers=[labels_str.info_class_name, labels_str.info_ann_objects]))

    def print_results(self):
        """
        Print the actual status of the annotations
        """
        if not self.annotate_classes:
            complete, incomplete = 0, 0
            incomplete_srt = ""
            if self.validate:
                for index, record in enumerate(self.dataset_annotated["annotations"]):
                    if self.validate_function(record):
                        complete += 1
                    else:
                        incomplete += 1
                        incomplete_srt += f" {index + 1}"
            print(f"{labels_str.info_completed_obj} {complete}")
            print(f"{labels_str.info_incomplete_obj} {incomplete}")
            if incomplete > 0:
                print(f"{labels_str.info_positions} {incomplete_srt}")










