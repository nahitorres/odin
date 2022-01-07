import os
import glob
import random
from PIL import Image
from matplotlib import pyplot as plt
from ipywidgets import Button, Output, HBox, VBox, Label, BoundedIntText
from IPython.display import Javascript, display


class ImagesLoader:
    def __init__(self, images_path, images_extension):
        self.images_path = images_path
        self.images_extension = images_extension

    def get_images_array(self):
        return glob.glob(os.path.join(self.images_path, "*" + self.images_extension))


class Iterator:
    def __init__(self,
                 images,
                 name="iterator",
                 show_name=True,
                 show_axis=False,
                 show_random=True,
                 fig_size=(10, 10),
                 buttons_vertical=False,
                 image_display_function=None
                 ):
        if len(images) == 0:
            raise Exception("No images provided")

        self.show_axis = show_axis
        self.name = name
        self.show_name = show_name
        self.show_random = show_random
        self.images = images
        self.max_pos = len(self.images) - 1
        self.pos = 0
        self.fig_size = fig_size
        self.buttons_vertical = buttons_vertical

        if image_display_function is None:
            self.image_display_function = self.__show_image
        else:
            self.image_display_function = image_display_function

        self.previous_button = self.__create_button("Previous", (self.pos == 0), self.__on_previous_clicked)
        self.next_button = self.__create_button("Next", (self.pos == self.max_pos), self.__on_next_clicked)
        self.save_button = self.__create_button("Save", False, self.__on_save_clicked)
        self.save_function = self.__save_function  # save_function

        buttons = [self.previous_button, self.next_button]

        if self.show_random:
            self.random_button = self.__create_button("Random", False, self.__on_random_clicked)
            buttons.append(self.random_button)

        buttons.append(self.save_button)

        label_total = Label(value='/ {}'.format(len(self.images)))
        self.text_index = BoundedIntText(value=1, min=1, max=len(self.images))
        self.text_index.layout.width = '80px'
        self.text_index.layout.height = '35px'
        self.text_index.observe(self.__selected_index)
        self.out = Output()
        self.out.add_class(name)

        if self.buttons_vertical:
            self.all_widgets = HBox(
                children=[VBox(children=[HBox([self.text_index, label_total])] + buttons), self.out])
        else:
            self.all_widgets = VBox(children=[HBox([self.text_index, label_total]), HBox(children=buttons), self.out])
        ## loading js library to perform html screenshots
        j_code = """
                require.config({
                    paths: {
                        html2canvas: "https://html2canvas.hertzen.com/dist/html2canvas.min"
                    }
                });
            """
        display(Javascript(j_code))

    def __create_button(self, description, disabled, function):
        button = Button(description=description)
        button.disabled = disabled
        button.on_click(function)
        return button

    def __show_image(self, image_path, index):
        img = Image.open(image_path)
        if self.show_name:
            print(os.path.basename(image_path))
        plt.figure(figsize=self.fig_size)
        if not self.show_axis:
            plt.axis('off')
        plt.imshow(img)
        plt.show()

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

    def __on_next_clicked(self, b):
        self.pos += 1
        self.__perform_action(self.pos, self.max_pos)

    def __on_save_clicked(self, b):
        if type(self.images[self.pos]) == str:
            self.save_function(self.images[self.pos], self.pos)
        else:
            self.save_function(self.images[self.pos]['file_name'], self.pos)

    def __perform_action(self, index, max_pos):
        self.next_button.disabled = (index == max_pos)
        self.previous_button.disabled = (index == 0)

        with self.out:
            self.out.clear_output()
        with self.out:
            self.image_display_function(self.images[index], index)

        self.text_index.unobserve(self.__selected_index)
        self.text_index.value = index + 1
        self.text_index.observe(self.__selected_index)

    def __on_previous_clicked(self, b):
        self.pos -= 1
        self.__perform_action(self.pos, self.max_pos)

    def __on_random_clicked(self, b):
        self.pos = random.randint(0, self.max_pos)
        self.__perform_action(self.pos, self.max_pos)

    def __selected_index(self, t):
        if t['owner'].value is None or t['name'] != 'value':
            return
        self.pos = t['new'] - 1
        self.__perform_action(self.pos, self.max_pos)

    def start_iteration(self):
        if self.max_pos < self.pos:
            print("No available images")
            return

        display(self.all_widgets)
        self.__perform_action(self.pos, self.max_pos)

    def refresh_output(self):
        with self.out:
            self.out.clear_output()
        with self.out:
            self.image_display_function(self.images[self.pos], self.pos)

