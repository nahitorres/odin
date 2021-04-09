import abc
import os
from ipywidgets import Button, Output, HBox, VBox, Label, BoundedIntText, Checkbox, HTML, RadioButtons, \
    BoundedFloatText, Textarea
from IPython.display import Javascript, display
from enum import Enum
from odin.classes import strings as labels_str
import json


class MetaPropertiesType(Enum):
    UNIQUE = 0
    COMPOUND = 1
    CONTINUE = 2
    TEXT = 3


class AnnotatorInterface(metaclass=abc.ABCMeta):

    def __init__(self,
                 dataset,
                 properties_and_values,  # properties { name: (MetaPropType, [possible values], optional label)}
                 output_path=None,
                 show_name=False,
                 show_axis=False,
                 fig_size=(10, 10),
                 buttons_vertical=False,
                 ds_name=None,
                 custom_display_function=None,
                 classes_to_annotate=None,
                 validate_function=None,
                 show_reset=True
                 ):

        for k, v in properties_and_values.items():
            if v[0] not in [MetaPropertiesType.UNIQUE, MetaPropertiesType.TEXT]:
                raise NotImplementedError(f"Cannot use {v[0]}!")

        self.dataset_orig = dataset
        self.properties_and_values = properties_and_values
        self.show_axis = show_axis
        self.show_reset = show_reset
        self.show_name = show_name

        if classes_to_annotate is None:  # if classes_to_annotate is None, all the classes would be annotated
            self.classes_to_annotate = self.dataset_orig.get_categories_names()  # otherwise, the only the classes in the list

        if ds_name is None:
            self.name = (self.dataset_orig.dataset_root_param.split('/')[-1]).split('.')[0]  # get_original_file_name
        else:
            self.name = ds_name

        self.set_output(output_path)

        print("{} {}".format(labels_str.info_new_ds, self.file_path_for_json))

        self.current_pos = 0
        self.mapping, self.dataset_annotated = self.create_results_dict(self.file_path_for_json)
        self.set_objects()

        self.fig_size = fig_size
        self.buttons_vertical = buttons_vertical

        self.current_image = {}

        label_total = self.create_label_total()

        # create buttons
        buttons = self.create_buttons()

        self.updated = False

        self.validation_show = HTML(value="")
        self.out = Output()
        self.out.add_class("my_canvas_class")

        self.checkboxes = {}
        self.radiobuttons = {}
        self.bounded_text = {}
        self.box_text = {}

        labels = self.create_check_radio_boxes()

        self.validate = not validate_function is None
        if self.validate:
            self.validate_function = validate_function

        self.set_display_function(custom_display_function)

        output_layout = self.set_check_radio_boxes_layout(labels)

        self.all_widgets = VBox(children=
                                [HBox([self.text_index, label_total]),
                                 HBox(buttons),
                                 self.validation_show,
                                 HBox([self.out,
                                       VBox(output_layout)])])

        self.load_js()

    @abc.abstractmethod
    def set_objects(self):
        pass

    @abc.abstractmethod
    def set_display_function(self, custom_display_function):
        pass

    def set_output(self, output_path):
        if output_path is None:
            name = self.dataset_orig.dataset_root_param
            self.output_directory = name.replace(os.path.basename(name), "")
        else:
            self.output_directory = output_path

        self.file_path_for_json = os.path.join(self.output_directory, self.name + "_ANNOTATED.json")

    def create_label_total(self):
        label_total = Label(value='/ {}'.format(len(self.objects)))
        self.text_index = BoundedIntText(value=1, min=1, max=len(self.objects))
        self.text_index.layout.width = '80px'
        self.text_index.layout.height = '35px'
        self.text_index.observe(self.selected_index)
        return label_total

    def create_buttons(self):
        # create buttons
        self.previous_button = self.create_button(labels_str.str_btn_prev, (self.current_pos == 0),
                                                  self.on_previous_clicked)
        self.next_button = self.create_button(labels_str.str_btn_next, (self.current_pos == self.max_pos),
                                              self.on_next_clicked)
        self.save_button = self.create_button(labels_str.str_btn_download, False, self.on_save_clicked)
        self.save_function = self.save_function  # save_function

        if self.show_reset:
            self.reset_button = self.create_button(labels_str.str_btn_reset, False, self.on_reset_clicked)
            buttons = [self.previous_button, self.next_button, self.reset_button, self.save_button]
        else:
            buttons = [self.previous_button, self.next_button, self.save_button]
        return buttons

    def create_check_radio_boxes(self):
        labels = dict()
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

    def set_check_radio_boxes_layout(self, labels):
        output_layout = []
        self.check_radio_boxes_layout = {}
        for rb_k, rb_v in self.radiobuttons.items():
            rb_v.layout.width = '180px'
            rb_v.observe(self.checkbox_changed)
            rb_v.add_class(rb_k)
            html_title = HTML(value="<b>" + labels[rb_k] + "</b>")
            self.check_radio_boxes_layout[rb_k] = VBox([rb_v])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout[rb_k]]))

        for cb_k, cb_i in self.checkboxes.items():
            for cb in cb_i:
                cb.layout.width = '180px'
                cb.observe(self.checkbox_changed)
                cb.add_class(cb_k)
            html_title = HTML(value="<b>" + labels[cb_k] + "</b>")
            self.check_radio_boxes_layout[cb_k] = VBox(children=[cb for cb in cb_i])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout[cb_k]]))

        for bf_k, bf in self.bounded_text.items():
            bf.layout.width = '80px'
            bf.layout.height = '35px'
            bf.observe(self.checkbox_changed)
            bf.add_class(bf_k)
            html_title = HTML(value="<b>" + labels[bf_k] + "</b>")
            self.check_radio_boxes_layout[bf_k] = VBox([bf])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout[bf_k]]))

        for tb_k, tb_i in self.box_text.items():
            tb_i.layout.width = '500px'
            tb_i.observe(self.checkbox_changed)
            tb_i.add_class(tb_k)
            html_title = HTML(value="<b>" + labels[tb_k] + "</b>")
            self.check_radio_boxes_layout[tb_k] = VBox([tb_i])
            output_layout.append(VBox([html_title, self.check_radio_boxes_layout[tb_k]]))

        return output_layout

    def load_js(self):
        ## loading js library to perform html screenshots
        j_code = """
                        require.config({
                            paths: {
                                html2canvas: "https://html2canvas.hertzen.com/dist/html2canvas.min"
                            }
                        });
                    """
        display(Javascript(j_code))

    def change_check_radio_boxes_value(self, current_ann):
        for m_k, m_v in self.properties_and_values.items():
            if m_v[0].value == MetaPropertiesType.UNIQUE.value:  # radiobutton
                self.radiobuttons[m_k].unobserve(self.checkbox_changed)
                self.radiobuttons[m_k].value = current_ann[m_k] if m_k in current_ann.keys() else None
                self.radiobuttons[m_k].observe(self.checkbox_changed)
            elif m_v[0].value == MetaPropertiesType.COMPOUND.value:  # checkbox
                for cb_i, cb_v in enumerate(self.checkboxes[m_k]):
                    cb_v.unobserve(self.checkbox_changed)
                    if m_k in current_ann.keys():
                        if cb_v.description in current_ann[m_k].keys():
                            cb_v.value = current_ann[m_k][cb_v.description]
                        else:
                            cb_v.value = False
                    else:
                        cb_v.value = False
                    cb_v.observe(self.checkbox_changed)
            elif m_v[0].value == MetaPropertiesType.CONTINUE.value:  # textbound
                self.bounded_text[m_k].unobserve(self.checkbox_changed)
                self.bounded_text[m_k].value = float(current_ann[m_k]) if m_k in current_ann.keys() else \
                    self.bounded_text[m_k].min
                self.bounded_text[m_k].observe(self.checkbox_changed)
            elif m_v[0].value == MetaPropertiesType.TEXT.value:  # text
                self.box_text[m_k].unobserve(self.checkbox_changed)
                self.box_text[m_k].value = current_ann[m_k] if m_k in current_ann.keys() else ""
                self.box_text[m_k].observe(self.checkbox_changed)

    def create_results_dict(self, file_path):
        mapping = {}
        mapping["annotated_ids"] = set()  # key: object_id from dataset, values=[(annotations done)]
        mapping["categories_counter"] = dict.fromkeys([c for c in self.dataset_orig.get_categories_names()], 0)
        self.mapping = mapping

        if not os.path.exists(file_path):  # it does exist __ANNOTATED in the output directory
            with open(self.dataset_orig.dataset_root_param, 'r') as input_json_file:
                dataset_annotated = json.load(input_json_file)
                input_json_file.close()

            # take the same metaproperties already in file if it's not empty (like a new file)
            meta_prop = dataset_annotated['meta_properties'] if 'meta_properties' in dataset_annotated.keys() else []

            # adds the new annotations categories to dataset if it doesn't exist
            for k_name, v in self.properties_and_values.items():

                new_mp_to_append = {
                    "name": k_name,
                    "type": v[0].value,
                }

                if len(v) > 1:
                    new_mp_to_append["values"] = sorted([p for p in v[1]])

                names_prop_in_file = {m_p['name']: m_p for m_i, m_p in enumerate(dataset_annotated[
                                                                                     'meta_properties'])} if 'meta_properties' in dataset_annotated.keys() else None

                if 'meta_properties' not in dataset_annotated.keys():  # it is a new file
                    meta_prop.append(new_mp_to_append)
                    dataset_annotated['meta_properties'] = []

                elif names_prop_in_file is not None and k_name not in names_prop_in_file.keys():
                    # if there is a property with the same in meta_properties, it must be the same structure as the one proposed
                    meta_prop.append(new_mp_to_append)
                    self.update_annotation_counter_and_current_pos(dataset_annotated)

                elif names_prop_in_file is not None and k_name in names_prop_in_file.keys() and \
                        names_prop_in_file[k_name] == new_mp_to_append:
                    # we don't append because it's already there
                    self.update_annotation_counter_and_current_pos(dataset_annotated)

                else:
                    raise NameError("An annotation with the same name {} "
                                    "already exist in dataset {}, and it has different structure. Check properties.".format(
                        k_name, self.dataset_orig.dataset_root_param))

                # if k_name is in name_props_in_file and it's the same structure. No update is done.
            dataset_annotated['meta_properties'] = dataset_annotated['meta_properties'] + meta_prop
        else:
            with open(file_path, 'r') as classification_file:
                dataset_annotated = json.load(classification_file)
                classification_file.close()

            self.update_annotation_counter_and_current_pos(dataset_annotated)
        return mapping, dataset_annotated

    @abc.abstractmethod
    def add_annotation_to_mapping(self, ann):
        pass

    @abc.abstractmethod
    def update_mapping_from_whole_dataset(self):
        pass

    @abc.abstractmethod
    def update_annotation_counter_and_current_pos(self, dataset_annotated):
        pass

    def execute_validation(self, ann):
        if self.validate:
            if self.validate_function(ann):
                self.validation_show.value = labels_str.srt_validation_not_ok
            else:
                self.validation_show.value = labels_str.srt_validation_ok

    @abc.abstractmethod
    def show_name_func(self, image_record, path_img):
        pass

    @abc.abstractmethod
    def checkbox_changed(self, b):
        pass

    def create_button(self, description, disabled, function):
        button = Button(description=description)
        button.disabled = disabled
        button.on_click(function)
        return button

    @abc.abstractmethod
    def show_image(self, image_record, ann_key):
        pass

    @abc.abstractmethod
    def save_state(self):
        pass

    def save_function(self, image_path):
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

    @abc.abstractmethod
    def perform_action(self):
        pass

    @abc.abstractmethod
    def get_image_record(self):
        pass

    @abc.abstractmethod
    def on_reset_clicked(self, b):
        pass

    def on_previous_clicked(self, b):
        self.save_state()
        self.current_pos -= 1
        self.perform_action()

    def on_next_clicked(self, b):
        self.save_state()
        self.current_pos += 1
        self.perform_action()

    @abc.abstractmethod
    def on_save_clicked(self, b):
        pass

    def selected_index(self, t):
        if t['owner'].value is None or t['name'] != 'value':
            return
        self.current_pos = t['new'] - 1
        self.perform_action()

    def start_annotation(self):
        if self.max_pos < self.current_pos:
            print(labels_str.info_no_more_images)
            return
        display(self.all_widgets)
        self.perform_action()

    @abc.abstractmethod
    def print_statistics(self):
        pass

    @abc.abstractmethod
    def print_results(self):
        pass
