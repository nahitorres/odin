import abc


class VisualizerInterface(metaclass=abc.ABCMeta):

    def __init__(self, dataset, analyzers):
        self.dataset = dataset
        self.analyzers = analyzers

    @abc.abstractmethod
    def visualize_annotations(self, categories, show_predictions=False):
        pass

    @abc.abstractmethod
    def visualize_annotations_for_property(self, meta_annotation, meta_annotation_value, show_predictions=False):
        pass

    @abc.abstractmethod
    def visualize_annotations_for_class_for_property(self, category, meta_annotation, meta_annotation_value,
                                                     show_predictions=False):
        pass

    @abc.abstractmethod
    def visualize_annotations_for_true_positive(self, categories=None, model=None):
        pass

    @abc.abstractmethod
    def visualize_annotations_for_false_positive(self, categories=None, model=None):
        pass

    @abc.abstractmethod
    def visualize_annotations_for_false_negative(self, categories=None, model=None):
        pass

    @abc.abstractmethod
    def visualize_annotations_for_error_type(self, error_type, categories=None, model=None):
        pass