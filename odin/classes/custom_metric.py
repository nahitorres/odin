import abc


class CustomMetric(metaclass=abc.ABCMeta):

    def __init__(self, name, is_single_threshold_metric):
        """
        This class provides an interface for a custom metric.

        Parameters
        ----------
        name: str
            Name of the custom metric
        is_single_threshold_metric: bool
            If true, it indicates that the metric uses a single threshold value (such as accuracy, F1, ...), otherwise
            it indicates that the metric does not use a single threshold value (such as average precision)
        """
        self.__name = name
        self.__single_threshold = is_single_threshold_metric

    def is_single_threshold(self):
        """
        Indicates whether or not the custom metric uses a single threshold value

        Returns
        -------
            bool
        """
        return self.__single_threshold

    def get_name(self):
        """
        Returns the custom metric name

        Returns
        -------
            str
        """
        return self.__name

    @abc.abstractmethod
    def evaluate_metric(self, gt, detections, matching, is_micro_required=False):
        """
        Custom metric evaluation

        Parameters
        ----------
        gt: array-like
            Ground Truth
        detections: array-like
            Detections
        matching:
            Ground Truth and Proposals matching. In classification is always None
        is_micro_required: bool, optional
            If True it is not a single class analysis

        Returns
        -------
        metric_value, standard_error
        """
        pass
