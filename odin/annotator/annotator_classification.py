import json
import os
from PIL import Image
from tabulate import tabulate
from matplotlib import pyplot as plt
from IPython.core.display import display
from ipywidgets import Output, Checkbox, RadioButtons, BoundedFloatText, Textarea, HTML, VBox, HBox, Label

from odin.annotator import MetaPropertiesType
from odin.classes import DatasetClassification, TaskType
from odin.classes import strings as labels_str
from odin.classes.strings import err_type
from odin.annotator.annotator_interface import AnnotatorInterface
from odin.utils.leaflet_zoom_utils import get_image_container_zoom, show_new_image

class AnnotatorClassification(AnnotatorInterface):

    supported_types = [TaskType.CLASSIFICATION_BINARY,
                       TaskType.CLASSIFICATION_SINGLE_LABEL,
                       TaskType.CLASSIFICATION_MULTI_LABEL]

    __mandatory_params_no_dataset = {'task_type', 'observations', 'classes_to_annotate', 'output_path', 'ds_name'}

    def __init__(self,
                 dataset=None,
                 task_type=None,
                 observations=None,
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
                 show_reset=True,
                 is_image=True):
        """
        The AnnotatorClassification class can be used to annotate data sets for classification tasks.

        Parameters
        ----------
        dataset: DatasetClassification, optional
            Dataset to be modified with annotations. If not specified, a new one is created. (default is None)
        task_type: TaskType, optional
            Problem task_type. If the dataset is not specified, it is a mandatory parameter. (default is None)
        observations: list, optional
            List of observations path. If the dataset is not specified, it is a mandatory parameter. (default is None)
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
        is_image: bool, optional
            Indicates whether the observations represent images. If False, the 'custom_display_function' parameter is mandatory. (default is True)
        """
        if dataset is None:  # create new dataset

            # check mandatory parameters validity
            if task_type is None or observations is None or classes_to_annotate is None or output_path is None or ds_name is None:
                raise Exception(f"Invalid parameters. Please be sure to specify the following parameters: {self.__mandatory_params_no_dataset}")

            if not isinstance(task_type, TaskType):
                raise TypeError(err_type.format("task_type"))
            elif task_type not in self.supported_types:
                raise Exception(labels_str.warn_task_not_supported)

            if not isinstance(observations, list):
                raise TypeError(err_type.format("observations"))
            elif len(observations) == 0:
                raise Exception(labels_str.warn_no_images)

            if not isinstance(classes_to_annotate, list):
                raise TypeError(err_type.format("classes_to_annotate"))
            elif len(classes_to_annotate) <= 1:
                raise Exception(labels_str.warn_little_classes)
            elif len(classes_to_annotate) > 2 and task_type.value == TaskType.CLASSIFICATION_BINARY.value:
                raise Exception(labels_str.warn_binary_only_two)

        else:  # dataset already exists

            if type(dataset) is not DatasetClassification:
                raise TypeError(f"Invalid dataset type: {type(dataset)}. Use DatasetClassification.")

            if properties_and_values is None:
                raise Exception("Please be sure to specify the properties and the values")
            elif not isinstance(properties_and_values, dict):
                raise TypeError(err_type.format("properties_and_values"))

        if not is_image and custom_display_function is None:
            raise Exception(labels_str.warn_display_function_needed)
        self.is_image = is_image
        self.key = "path" if self.is_image else "observation"

        self.objects = observations

        self.out = None
        self.image_container = None

        super().__init__(dataset, task_type, classes_to_annotate, output_path, ds_name, properties_and_values, show_name,
                       show_axis, fig_size, buttons_vertical, custom_display_function, validate_function, show_reset)

        if self.out is not None:
            self.out.add_class("self.name")
            self.all_widgets = VBox(children=
                                    [HBox([self.text_index, self.label_total]),
                                     HBox(self.buttons),
                                     self.validation_show,
                                     HBox([self.out,
                                           VBox(self.output_layout)])])
        else:
            self.title_lbl = Label(value="title")
            self.all_widgets = VBox(children=
                                    [HBox([self.text_index, self.label_total]),
                                     HBox(self.buttons),
                                     self.title_lbl,
                                     self.validation_show,
                                     HBox([self.image_container,
                                           VBox(self.output_layout)])])
        self._load_js()

    def _set_display_function(self, custom_display_function):
        if custom_display_function is None:
            self.out = Output()
            self.image_display_function = self._show_image
        elif type(custom_display_function) is str:
            if custom_display_function == "zoom_leaflet":
                self.image_container = get_image_container_zoom()
                self.image_display_function = self._show_image_leaflet
            else:
                raise NotImplementedError(f"Function {custom_display_function} not implemented!")
        else:
            self.out = Output()
            self.image_display_function = custom_display_function

    def _perform_action(self):
        """
        Update the dataset and the mapping paramenters based on the annotation and clear the output to display a new
        image
        """
        self.next_button.disabled = (self.current_pos == self.max_pos)
        self.previous_button.disabled = (self.current_pos == 0)

        if not self.objects[self.current_pos] in self.mapping["observations"].keys():
            observation = {self.key: self.objects[self.current_pos]}
            if self.is_image:
                observation["file_name"] = os.path.basename(self.objects[self.current_pos])

            observation["id"] = len(self.mapping["observations"]) + 1
            if self.__is_multilabel():
                observation["categories"] = []
            self.dataset_annotated["observations"].append(observation)

            self.mapping["observations"][observation[self.key]] = len(self.dataset_annotated["observations"]) - 1

        current_index = self.mapping["observations"][self.objects[self.current_pos]]
        self.__change_check_radio_boxes_value(self.dataset_annotated["observations"][current_index])

        if self.out is not None:
            with self.out:
                self.out.clear_output()
                self.image_display_function(self.dataset_annotated["observations"][current_index])
        else:
            self.image_display_function(self.dataset_annotated["observations"][current_index])

        self._execute_validation(self.dataset_annotated["observations"][current_index])

        self.text_index.unobserve(self._selected_index)
        self.text_index.value = self.current_pos + 1
        self.text_index.observe(self._selected_index)

    def _show_name_func(self, image_record, path_img):
        if self.show_name:
            self.title_lbl.value = os.path.basename(path_img)

    def _show_image_leaflet(self, image_record):
        if self.dataset_orig is None:
            path_img = image_record['path']
        else:
            path_img = os.path.join(self.dataset_orig.images_abs_path, image_record['file_name'])
        self._show_name_func(image_record, path_img)
        if not os.path.exists(path_img):
            print(f"{labels_str.info_missing} {path_img}")
            return

        show_new_image(self.image_container, path_img)

    def _show_image(self, image_record):
        """
        Display the selected image
        Parameters
        ----------
        image_record: dict
            image to display
        """
        if self.dataset_orig is None:
            path_img = image_record['path']
        else:
            path_img = os.path.join(self.dataset_orig.images_abs_path, image_record['file_name'])
        if not os.path.exists(path_img):
            print("Image cannot be load" + path_img)
            return
        img = Image.open(path_img)
        if self.show_name:
            print(os.path.basename(path_img))
        plt.figure(figsize=self.fig_size)
        if not self.show_axis:
            plt.axis('off')
        plt.imshow(img)
        plt.show()

    def save_state(self):
        """
        Create the json file for the dataset
        """
        with open(self.file_path, 'w') as output_file:
            json.dump(self.dataset_annotated, output_file, indent=4)

    def _create_results_dict(self):
        """
        Create a dictionary to mapping the categories and observations and create a dict for the dataset to be created
        Returns
        -------
            mapping, dataset
        """
        mapping = {"categories_id": {}, "categories_name": {}, "observations": {}}
        dataset = {'categories': [], "observations": [], "meta_properties": []}

        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as classification_file:
                dataset = json.load(classification_file)
                classification_file.close()
        elif self.dataset_orig is not None:
            with open(self.dataset_orig.dataset_root_param, 'r') as input_json_file:
                dataset = json.load(input_json_file)
                input_json_file.close()
        if self.objects is None:
            if "path" not in dataset["observations"]:
                self.objects = [os.path.join(self.dataset_orig.images_abs_path, obs['file_name']) for obs in dataset['observations']]
            else:
                self.objects = [obs['path'] for obs in dataset['observations']]

        for index, img in enumerate(dataset['observations']):
            if "path" not in img and self.is_image:
                key = os.path.join(self.dataset_orig.images_abs_path, img['file_name'])
                mapping['observations'][key] = index
            else:
                mapping['observations'][img[self.key]] = index
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

        if self.annotate_meta_properties:
            meta_prop = dataset['meta_properties'] if 'meta_properties' in dataset.keys() else []
            for k_name, v in self.properties_and_values.items():
                new_mp = {"name": k_name, "type": v[0].value}
                if len(v) > 1:
                    new_mp["values"] = sorted([p for p in v[1]])

                names_prop_in_file = {m_p['name']: m_p for m_i, m_p in enumerate(dataset['meta_properties'])} if 'meta_properties' in dataset.keys() else None

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
                elif names_prop_in_file is not None and k_name in names_prop_in_file.keys() and \
                        names_prop_in_file[k_name] != new_mp and names_prop_in_file[k_name]['type'] == new_mp['type']:
                    tmp_meta_prop = []
                    for p in meta_prop:
                        if p['name'] == new_mp['name']:
                            continue
                        tmp_meta_prop.append(p)
                    tmp_meta_prop.append(new_mp)
                    meta_prop = tmp_meta_prop

                else:
                    raise NameError("An annotation with the same name '{}' "
                                    "already exist in dataset, and it has different structure. Please, check the properties.".format(
                        k_name))
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
        self.save_button = self._create_button(labels_str.str_btn_download, False, self._on_save_clicked)
        if self.show_reset:
            self.reset_button = self._create_button(labels_str.str_btn_reset, False, self._on_reset_clicked)
            return [self.previous_button, self.next_button, self.reset_button, self.save_button]
        return [self.previous_button, self.next_button, self.save_button]

    def _on_save_clicked(self, b):
        """
        Function called when the 'download' button is clicked
        """
        self.save_state()
        file_name = os.path.basename(self.objects[self.current_pos])
        self.save_function(file_name, self.current_pos)

    def _on_reset_clicked(self, b):
        """
        Function called when the 'reset' button is clicked
        """
        current_index = self.mapping["observations"][self.objects[self.current_pos]]
        current_ann = self.dataset_annotated['observations'][current_index]

        if self.annotate_classes:
            if self.__is_multilabel() and "categories" in current_ann:
                current_ann["categories"] = []
            elif "category" in current_ann:
                del current_ann["category"]

        if self.annotate_meta_properties:
            for m_k, m_v in self.properties_and_values.items():
                if m_k in current_ann:
                    del current_ann[m_k]

        self.__change_check_radio_boxes_value(current_ann)
        self._execute_validation(current_ann)

    def _checkbox_changed(self, b):
        """
        Function called when the widget value change
        """
        if b['owner'].value is None or b['name'] != 'value':
            return

        class_name = b['owner'].description
        value = b['owner'].value
        annotation_name = b['owner']._dom_classes[0]
        current_index = self.mapping["observations"][self.objects[self.current_pos]]
        if annotation_name == "categories":
            if self.__is_multilabel():
                class_index = self.mapping["categories_name"][class_name]
                if not class_index in self.dataset_annotated["observations"][current_index]["categories"] and value:
                    self.dataset_annotated["observations"][current_index]["categories"].append(class_index)
                if class_index in self.dataset_annotated["observations"][current_index]["categories"] and not value:
                    self.dataset_annotated["observations"][current_index]["categories"].remove(class_index)
            else:
                class_index = self.mapping["categories_name"][value]
                self.dataset_annotated["observations"][current_index]["category"] = class_index
        else:
            if self.properties_and_values[annotation_name][0].value in [MetaPropertiesType.COMPOUND.value]:
                if annotation_name not in self.dataset_annotated["observations"][current_index].keys():
                    self.dataset_annotated["observations"][current_index][annotation_name] = {p: False for p in self.properties_and_values[annotation_name][1]}
                self.dataset_annotated["observations"][current_index][annotation_name][class_name] = value
            else:  # UNIQUE VALUE
                self.dataset_annotated["observations"][current_index][annotation_name] = value

        self._execute_validation(self.dataset_annotated["observations"][current_index])
        if self.current_pos == self.max_pos:
            self.save_state()

    def _create_check_radio_boxes(self):
        """
        Create the widgets for the annotation

        """
        labels = dict()

        if self.annotate_classes:
            labels["categories"] = self.classes_to_annotate
            if self.__is_multilabel():
                self.checkboxes["categories"] = [Checkbox(False, description='{}'.format(self.classes_to_annotate[i]),
                                                          indent=False, name="categories") for i in
                                                 range(len(self.classes_to_annotate))]
            else:
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
                    self.checkboxes[k_name] = [Checkbox(False, indent=False, name=k_name,
                                                        description=prop_name) for prop_name in v[1]]
                elif MetaPropertiesType.CONTINUE.value == v[0].value:
                    self.bounded_text[k_name] = BoundedFloatText(value=v[1][0], min=v[1][0], max=v[1][1])

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
            if self.__is_multilabel():
                for cb in self.checkboxes["categories"]:
                    cb.layout.width = '180px'
                    cb.observe(self._checkbox_changed)
                    cb.add_class("categories")
                html_title = HTML(value="<b>Categories</b>")
                self.check_radio_boxes_layout["categories"] = VBox(children=[cb for cb in self.checkboxes["categories"]])
                output_layout.append(VBox([html_title, self.check_radio_boxes_layout["categories"]]))
            else:
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

    def __change_check_radio_boxes_value(self, current_ann):
        """
        update the values of the widget based on the curent annotation
        Parameters
        ----------
        current_ann: dict
            current annotation
        """
        if self.annotate_classes:
            if self.__is_multilabel():
                for cb in self.checkboxes["categories"]:
                    cb.unobserve(self._checkbox_changed)
                    cb.value = self.mapping["categories_name"][cb.description] in current_ann["categories"] if "categories" in current_ann else False
                    cb.observe(self._checkbox_changed)
            else:
                self.radiobuttons["categories"].unobserve(self._checkbox_changed)
                self.radiobuttons["categories"].value = self.mapping["categories_id"][current_ann["category"]] if "category" in current_ann.keys() else None
                self.radiobuttons["categories"].observe(self._checkbox_changed)

        if self.annotate_meta_properties:
            for m_k, m_v in self.properties_and_values.items():
                if m_v[0].value == MetaPropertiesType.UNIQUE.value:  # radiobutton
                    self.radiobuttons[m_k].unobserve(self._checkbox_changed)
                    self.radiobuttons[m_k].value = current_ann[m_k] if m_k in current_ann.keys() else None
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
                        cb_v.observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.CONTINUE.value:  # textbound
                    self.bounded_text[m_k].unobserve(self._checkbox_changed)
                    self.bounded_text[m_k].value = float(current_ann[m_k]) if m_k in current_ann.keys() else \
                        self.bounded_text[m_k].min
                    self.bounded_text[m_k].observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.TEXT.value:  # text
                    self.box_text[m_k].unobserve(self._checkbox_changed)
                    self.box_text[m_k].value = current_ann[m_k] if m_k in current_ann.keys() else ""
                    self.box_text[m_k].observe(self._checkbox_changed)

    def __is_multilabel(self):
        """
        check if the task type is multi label
        Returns
        -------
            bool
        """
        return TaskType.CLASSIFICATION_MULTI_LABEL == self.task_type

    def _execute_validation(self, ann):
        """
        Validate the actual status of the annotation
        Parameters
        ----------
        ann: dict
            annotation to validate
        """
        if self.validate:
            if self.validate_function(ann):
                self.validation_show.value = labels_str.srt_validation_not_ok
            else:
                self.validation_show.value = labels_str.srt_validation_ok

    def start_classification(self):
        """
        Method to start the annotation
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
        counter = dict()
        for c in self.mapping["categories_id"]:
            counter[c] = 0

        for record in self.dataset_annotated["observations"]:

            if self.__is_multilabel():
                if "categories" in record:
                    for c in record["categories"]:
                        counter[c] += 1
            elif "category" in record:
                counter[record["category"]] += 1

        table = []
        for c in counter:
            table.append([self.mapping["categories_id"][c], counter[c]])
        table = sorted(table, key=lambda x: x[0])

        print(tabulate(table, headers=[labels_str.info_class_name, labels_str.info_ann_images]))

    def print_results(self):
        """
        Print the actual status of the annotations
        """
        if not self.annotate_classes:
            complete, incomplete = 0, 0
            incomplete_srt = ""
            if self.validate:
                for index, record in enumerate(self.dataset_annotated["observations"]):
                    if self.validate_function(record):
                        complete += 1
                    else:
                        incomplete += 1
                        incomplete_srt += f" {index + 1}"
            print(f"{labels_str.info_completed} {complete}")
            print(f"{labels_str.info_incomplete} {incomplete}")
            if incomplete > 0:
                print(f"{labels_str.info_positions} {incomplete_srt}")