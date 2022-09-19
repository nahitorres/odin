import abc

# TODO: DISCUSS REFACTORING
class TSCustomMetric(metaclass=abc.ABCMeta):
	def __init__(self, name, is_single_threshold):
		"""Instantiate a custom metric.

		Parameters
		----------
		name : str
            Name of the custom metric
            
        is_single_threshold: bool
            If true, it indicates that the metric uses a single threshold value
            (such as accuracy, F1, ...), otherwise it indicates that the metric
            does not use a single threshold value (such as average precision)
		"""
		self.__name = name
		self.__single_threshold = is_single_threshold

	def get_name(self):
		"""Gets the name of the metric.

		Returns
		-------
		name : string
			The name of the metric
		"""
		return self.__name
	
	def is_single_threshold(self):
		"""Indicates whether or not the custom metric uses a single threshold value

        Returns
        -------
            bool
        """
		return self.__single_threshold

	@abc.abstractmethod
	def evaluate_metric(self, y_true,
						y_pred,
						threshold = None,
						inverse_threshold = False,
						evaluation_type = None,
						min_consecutive_samples = 1):
		"""Evaluates the metric.

		Parameters
		----------
		y_true : array-like
			The true value for the values.

		y_pred : array-like
			The predicted values for the points.

		threshold : float, default=None
			If needed, it is the threshold used to compute anomalies.

		inverse_threshold : bool, default=False
			If needed, it says whether the threshold has to be treated
			inversely.

		evaluation_type : TPEvaluation, default=None
			If needed, the evaluation type for the metric.

		min_consecutive_samples : int, default=1
			If needed, the minimum number of points in a window to be considered
			an anomaly.

		Returns
		-------
		metric_value : float
			The value for the metric.
		"""
		pass
