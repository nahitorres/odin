import os
import cv2
from matplotlib import pyplot as plt
from .dataset_classification import DatasetClassification
from odin.utils import Iterator
from .visulizer_interface import VisualizerInterface
from odin.classes import strings as labels_str


class VisualizerClassification(VisualizerInterface):

    def __init__(self, dataset: DatasetClassification, is_image=True, custom_display_function=None):
        super().__init__(dataset)
        self.dataset = dataset
        self.__colors = {}
        self.__is_image = is_image
        self.__display_function = self.__show_image
        
        if not custom_display_function is None:
            self.__display_function = custom_display_function
        else:
            if not is_image:
                raise Exception(labels_str.warn_display_function_needed)
    def __show_image(self, observation, index):

        if self.__is_image and ('path' in observation and not  os.path.exists(observation['path'])):
            print("Image path does not exist: " + observation)
        else:
            plt.figure(figsize=(10, 10))
            img = cv2.cvtColor(cv2.imread(observation['path']), cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    def visualize_annotations(self, categories=None):

        categories_ds = self.dataset.get_categories_names()
        if categories is None:
            observations = self.dataset.get_all_observations()
        else:
            for cat in categories:
                if not cat in categories_ds:
                    print(labels_str.warn_incorrect_class)
                    return
            observations = self.dataset.get_observations_from_categories(categories)
        self.__start_iterator(observations)

    def visualize_annotations_for_property(self, meta_annotation, meta_annotation_value):

        observations = self.dataset.get_observations_from_property(meta_annotation,
                                                                          meta_annotation_value)

        self.__start_iterator(observations)

    def visualize_annotations_for_class_for_property(self, category, meta_annotation, meta_annotation_value):

        if self.dataset.is_valid_category(category):
            category_id = self.dataset.get_category_id_from_name(category)
            observations = self.dataset.get_observations_from_property_category(category_id, meta_annotation,
                                                                                           meta_annotation_value)
            self.__start_iterator(observations)

        else:
            print(labels_str.warn_incorrect_class)

    def __start_iterator(self, observations):

        self.__current_observations = [i.to_dict() for b, i in observations.iterrows()]

        if len(self.__current_observations) == 0:
            print(labels_str.warn_no_images_criteria)
        else:
            iterator = Iterator(self.__current_observations, show_name=False, image_display_function=self.__display_function)
            iterator.start_iteration()