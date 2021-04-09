import abc

from odin.classes import DatasetInterface


class VisualizerInterface(metaclass=abc.ABCMeta):

    def __init__(self, dataset:DatasetInterface):
        self.dataset = dataset

    @abc.abstractmethod
    def visualize_annotations(self, categories=list):
        pass

    @abc.abstractmethod
    def visualize_annotations_for_property(self, meta_annotation=str, meta_annotation_value=str):
        pass

    @abc.abstractmethod
    def visualize_annotations_for_class_for_property(self, category=str, meta_annotation=str, meta_annotation_value=str):
        pass