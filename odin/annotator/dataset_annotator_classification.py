import os
import json
import glob
from PIL import Image
from tabulate import tabulate
from matplotlib import pyplot as plt
from IPython.display import Javascript, display
from ipywidgets import Button, Output, HBox, VBox, Label, BoundedIntText, Checkbox, RadioButtons
from odin.classes import strings as labels_str
from odin.classes import TaskType


class ImagesLoader:
    def __init__(self, images_path, images_extension):
        self.images_path = images_path
        self.images_extension = images_extension

    def get_images_array(self):
        return glob.glob(os.path.join(self.images_path, "*" + self.images_extension))


# TODO: MODIFY TO USE FILE_NAME FOR MAPPING INSTEAD OF PATH
class DatasetAnnotatorClassification:
    supported_types = [TaskType.CLASSIFICATION_BINARY,
                       TaskType.CLASSIFICATION_SINGLE_LABEL,
                       TaskType.CLASSIFICATION_MULTI_LABEL]

    def __init__(self,
                 task_type,
                 observations,
                 output_path,
                 name,
                 classes,
                 show_name=True,
                 show_axis=False,
                 fig_size=(10, 10),
                 buttons_vertical=False,
                 custom_display_function=None,
                 is_image=True
                 ):

        if task_type not in self.supported_types:
            raise Exception(labels_str.warn_task_not_supported)

        if len(observations) == 0:
            raise Exception(labels_str.warn_no_images)

        num_classes = len(classes)
        if num_classes <= 1:
            raise Exception(labels_str.warn_little_classes)

        elif len(classes) > 2 and task_type.value == TaskType.CLASSIFICATION_BINARY.value:
            raise Exception(labels_str.warn_binary_only_two)

        self.is_image = is_image
        self.key = "path" if self.is_image else "observation"
        if not self.is_image and custom_display_function is None:
            raise Exception(labels_str.warn_display_function_needed)

        self.task_type = task_type
        self.show_axis = show_axis
        self.name = name
        self.show_name = show_name
        self.output_path = output_path
        self.file_path = os.path.join(self.output_path, self.name + ".json")
        print(labels_str.info_ds_output + self.file_path)
        self.mapping, self.dataset = self.__create_results_dict(self.file_path, classes)

        self.classes = list(self.mapping["categories_id"].values())

        if len(self.classes) > 2 and task_type.value == TaskType.CLASSIFICATION_BINARY.value:
            raise Exception(labels_str.warn_binary_only_two + " ".join(self.classes))

        self.observations = observations
        self.max_pos = len(self.observations) - 1
        self.pos = 0
        self.fig_size = fig_size
        self.buttons_vertical = buttons_vertical

        if custom_display_function is None:
            self.image_display_function = self.__show_image
        else:
            self.image_display_function = custom_display_function

        self.previous_button = self.__create_button(labels_str.str_btn_prev, (self.pos == 0),
                                                    self.__on_previous_clicked)
        self.next_button = self.__create_button(labels_str.str_btn_next, (self.pos == self.max_pos),
                                                self.__on_next_clicked)
        self.save_button = self.__create_button(labels_str.str_btn_download, False, self.__on_save_clicked)
        self.save_function = self.__save_function  # save_function

        buttons = [self.previous_button, self.next_button, self.save_button]

        label_total = Label(value='/ {}'.format(len(self.observations)))
        self.text_index = BoundedIntText(value=1, min=1, max=len(self.observations))
        self.text_index.layout.width = '80px'
        self.text_index.layout.height = '35px'
        self.text_index.observe(self.__selected_index)
        self.out = Output()
        self.out.add_class(name)

        if self.__is_multilabel():
            self.checkboxes = [Checkbox(False, description='{}'.format(self.classes[i]),
                                        indent=False) for i in range(len(self.classes))]
            for cb in self.checkboxes:
                cb.layout.width = '180px'
                cb.observe(self.__checkbox_changed)
            self.checkboxes_layout = VBox(children=[cb for cb in self.checkboxes])
        else:
            self.checkboxes = RadioButtons(options=self.classes,
                                           disabled=False,
                                           indent=False)
            self.checkboxes.layout.width = '180px'
            self.checkboxes.observe(self.__checkbox_changed)
            self.checkboxes_layout = VBox(children=[self.checkboxes])

        output_layout = HBox(children=[self.out, self.checkboxes_layout])
        if self.buttons_vertical:
            self.all_widgets = HBox(
                children=[VBox(children=[HBox([self.text_index, label_total])] + buttons), output_layout])
        else:
            self.all_widgets = VBox(
                children=[HBox([self.text_index, label_total]), HBox(children=buttons), output_layout])

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
        mapping = {"categories_id": {}, "categories_name": {}, "observations": {}}

        if not os.path.exists(file_path):
            dataset = {'categories': [], "observations": []}
            for index, c in enumerate(cc):
                category = {"supercategory": c, "name": c, "id": index + 1}
                dataset["categories"].append(category)
        else:
            with open(file_path, 'r') as classification_file:
                dataset = json.load(classification_file)
            for index, img in enumerate(dataset['observations']):
                mapping['observations'][img[self.key]] = index

        for c in dataset['categories']:
            mapping['categories_id'][c["id"]] = c["name"]
            mapping['categories_name'][c["name"]] = c["id"]
        index_categories = len(dataset['categories']) + 1

        for c in cc:
            if not c in mapping['categories_name'].keys():
                mapping['categories_id'][index_categories] = c
                mapping['categories_name'][c] = index_categories
                category = {"supercategory": c, "name": c, "id": index_categories}
                dataset["categories"].append(category)
                index_categories += 1

        return mapping, dataset

    def __checkbox_changed(self, b):

        if b['owner'].value is None or b['name'] != 'value':
            return

        class_name = b['owner'].description
        value = b['owner'].value
        current_index = self.mapping["observations"][self.observations[self.pos]]

        if self.__is_multilabel():
            class_index = self.mapping["categories_name"][class_name]
            if not class_index in self.dataset["observations"][current_index]["categories"] and value:
                self.dataset["observations"][current_index]["categories"].append(class_index)
            if class_index in self.dataset["observations"][current_index]["categories"] and not value:
                self.dataset["observations"][current_index]["categories"].remove(class_index)
        else:
            class_index = self.mapping["categories_name"][value]
            self.dataset["observations"][current_index]["category"] = class_index

        if self.pos == self.max_pos:
            self.save_state()

    def __is_multilabel(self):
        return TaskType.CLASSIFICATION_MULTI_LABEL.value == self.task_type.value

    def __create_button(self, description, disabled, function):
        button = Button(description=description)
        button.disabled = disabled
        button.on_click(function)
        return button

    def __show_image(self, image_record):
        if not 'path' in image_record:
            print("missing path")
        if not os.path.exists(image_record['path']):
            print("Image cannot be load" + image_record['path'])

        img = Image.open(image_record['path'])
        if self.show_name:
            print(os.path.basename(image_record['path']))
        plt.figure(figsize=self.fig_size)
        if not self.show_axis:
            plt.axis('off')
        plt.imshow(img)
        plt.show()

    def save_state(self):
        with open(self.file_path, 'w') as output_file:
            json.dump(self.dataset, output_file, indent=4)

    def __save_function(self, image_path, index):
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

        if not self.observations[self.pos] in self.mapping["observations"].keys():
            observation = {self.key: self.observations[self.pos]}
            if self.is_image:
                observation["file_name"] = os.path.basename(self.observations[self.pos])

            observation["id"] = len(self.mapping["observations"]) + 1
            if self.__is_multilabel():
                observation["categories"] = []
            self.dataset["observations"].append(observation)

            self.mapping["observations"][observation[self.key]] = len(self.dataset["observations"]) - 1

        current_index = self.mapping["observations"][self.observations[self.pos]]
        self.next_button.disabled = (self.pos == self.max_pos)
        self.previous_button.disabled = (self.pos == 0)

        if self.__is_multilabel():
            for cb in self.checkboxes:
                cb.unobserve(self.__checkbox_changed)
                if not "categories" in self.dataset["observations"][current_index]:
                    self.dataset["observations"][current_index]["categories"] = []
                categories = self.dataset["observations"][current_index]["categories"]
                cb.value = self.mapping["categories_name"][cb.description] in categories
                cb.observe(self.__checkbox_changed)
        else:
            self.checkboxes.unobserve(self.__checkbox_changed)
            obs = self.dataset["observations"][current_index]
            category = obs["category"] if "category" in obs else None

            if category:
                for k in self.mapping["categories_name"]:
                    if self.mapping["categories_name"][k] == category:
                        category = k
                        break
            self.checkboxes.value = category
            self.checkboxes.observe(self.__checkbox_changed)

        with self.out:
            self.out.clear_output()
            self.image_display_function(self.dataset["observations"][current_index])

        self.text_index.unobserve(self.__selected_index)
        self.text_index.value = self.pos + 1
        self.text_index.observe(self.__selected_index)

    def __on_previous_clicked(self, b):
        self.save_state()
        self.pos -= 1
        self.__perform_action()

    def __on_next_clicked(self, b):
        self.save_state()
        self.pos += 1
        self.__perform_action()

    def __on_save_clicked(self, b):
        self.save_state()
        self.save_function(self.observations[self.pos], self.pos)

    def __selected_index(self, t):
        if t['owner'].value is None or t['name'] != 'value':
            return
        self.pos = t['new'] - 1
        self.__perform_action()

    def start_classification(self):
        if self.max_pos < self.pos:
            print("No available observation")
            return
        display(self.all_widgets)
        self.__perform_action()

    def print_statistics(self):
        counter = dict()
        for c in self.mapping["categories_id"]:
            counter[c] = 0

        for record in self.dataset["observations"]:

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
