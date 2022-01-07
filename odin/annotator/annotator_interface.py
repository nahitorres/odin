import os
import abc
from enum import Enum

from IPython.core.display import display, Javascript
from ipywidgets import Output, Label, BoundedIntText, HTML, Button

from odin.classes import strings as labels_str


class ImagesLoader:
    def __init__(self, images_path, images_extension):
        self.images_path = images_path
        self.images_extension = images_extension

    def get_images_array(self):
        return [self.images_path + "/" + img for img in os.listdir(self.images_path) if img.endswith(self.images_extension)]


class MetaPropertiesType(Enum):
    UNIQUE = 0
    COMPOUND = 1
    CONTINUE = 2
    TEXT = 3


class AnnotatorInterface(metaclass=abc.ABCMeta):

    def __init__(self,
                 dataset=None,
                 task_type=None,
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

        if dataset is None:
            self.annotate_classes = True
            self.annotate_meta_properties = (properties_and_values is not None)
            self.task_type = task_type
            self.validate = False
        else:
            self.annotate_classes = False
            self.annotate_meta_properties = True
            self.task_type = dataset.task_type
            self.validate = (validate_function is not None)

        self.dataset_orig = dataset
        self.classes_to_annotate = self.dataset_orig.get_categories_names() if classes_to_annotate is None else classes_to_annotate
        if output_path is None:
            name = self.dataset_orig.dataset_root_param
            self.output_path = name.replace(os.path.basename(name), "")
        else:
            self.output_path = output_path

        if ds_name is None:
            self.name = (self.dataset_orig.dataset_root_param.split('/')[-1]).split('.')[
                            0] + "_ANNOTATED"  # get_original_file_name
        else:
            self.name = ds_name

        self.file_path = os.path.join(self.output_path, self.name + ".json")
        print(labels_str.info_ds_output + self.file_path)

        self.properties_and_values = properties_and_values
        self.show_name = show_name
        self.show_axis = show_axis
        self.fig_size = fig_size
        self.buttons_vertical = buttons_vertical

        self._set_display_function(custom_display_function)

        if self.validate:
            self.validate_function = validate_function
        self.show_reset = show_reset

        self.mapping, self.dataset_annotated = self._create_results_dict()
        self.max_pos = len(self.objects) - 1
        self.current_pos = 0
        self.validation_show = HTML(value="")

        self.buttons = self._create_buttons()
        self.save_function = self.__save_function  # save_function
        self.label_total = self.__create_label_total()
        self.checkboxes = {}
        self.radiobuttons = {}
        self.bounded_text = {}
        self.box_text = {}
        self.check_radio_boxes_layout = {}
        labels = self._create_check_radio_boxes()
        self.output_layout = self._set_check_radio_boxes_layout(labels)

    @abc.abstractmethod
    def print_statistics(self):
        pass

    @abc.abstractmethod
    def print_results(self):
        pass

    @abc.abstractmethod
    def save_state(self):
        pass

    @abc.abstractmethod
    def _set_display_function(self, custom_display_function):
        pass

    @abc.abstractmethod
    def _show_image(self):
        pass

    @abc.abstractmethod
    def _create_buttons(self):
        pass

    @abc.abstractmethod
    def _create_results_dict(self):
        pass

    @abc.abstractmethod
    def _create_check_radio_boxes(self):
        pass

    @abc.abstractmethod
    def _set_check_radio_boxes_layout(self, labels):
        pass

    @abc.abstractmethod
    def _on_save_clicked(self, b):
        pass

    @abc.abstractmethod
    def _on_reset_clicked(self, b):
        pass

    @abc.abstractmethod
    def _checkbox_changed(self, b):
        pass

    @abc.abstractmethod
    def _perform_action(self):
        pass

    @abc.abstractmethod
    def _execute_validation(self):
        pass

    def _create_button(self, description, disabled, function):
        """
        Create a new button
        Parameters
        ----------
        description: str
            Button description
        disabled: bool
            If the button is disabled
        function:
            function called by the 'on_click' listener

        Returns
        -------

        """
        button = Button(description=description)
        button.disabled = disabled
        button.on_click(function)
        return button

    def _on_previous_clicked(self, b):
        """
        When the 'previous' button is clicked update the current position and perform the action
        """
        self.save_state()
        self.current_pos -= 1
        self._perform_action()

    def _on_next_clicked(self, b):
        """
        When the 'next' button is clicked update the current position and perform the action
        """
        self.save_state()
        self.current_pos += 1
        self._perform_action()

    def _selected_index(self, t):
        """
        When the index is changed update the current position and perform the action
        """

        if t['owner'].value is None or t['name'] != 'value':
            return
        self.save_state()
        self.current_pos = t['new'] - 1
        self._perform_action()

    def _load_js(self):
        """
        loading js library to perform html screenshots
        """
        j_code = """
                        require.config({
                            paths: {
                                html2canvas: "https://html2canvas.hertzen.com/dist/html2canvas.min"
                            }
                        });
                    """
        display(Javascript(j_code))

    def __save_function(self, file_name, index):
        """
        download the current images
        Parameters
        ----------
        file_name: str
            file_name of the image
        """
        img_name = file_name.split('.')[0]
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

    def __create_label_total(self):
        """
        Create the text label with the current position
        Returns
        -------
            label

        """
        label_total = Label(value='/ {}'.format(len(self.objects)))
        self.text_index = BoundedIntText(value=1, min=1, max=len(self.objects))
        self.text_index.layout.width = '80px'
        self.text_index.layout.height = '35px'
        self.text_index.observe(self._selected_index)
        return label_total