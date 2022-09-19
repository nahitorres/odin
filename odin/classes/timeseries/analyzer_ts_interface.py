import abc

from aenum import extend_enum

from odin.classes.timeseries.ts_custom_metric import TSCustomMetric
from odin.classes import Metrics, Curves
from odin.classes.strings import err_type, err_value, err_ts_metric, \
    err_analyzer_invalid_curve
from odin.classes.timeseries.metrics import coefficient_of_determination, coefficient_of_variation, f_beta_score, matthews_correlation_coefficient, mean_absolute_ranged_relative_error, overall_percentage_error, precision_score, recall_score, f1_score, accuracy, nab_score, \
    mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, false_alarm_rate, \
    miss_alarm_rate, root_mean_squared_log_error, symmetric_mean_absolute_percentage_error

from odin.classes.timeseries.anomaly_matching_strategies import *
from odin.utils import get_root_logger
from odin.utils.draw_utils import make_multi_category_plot, plot_multiple_curves


class AnalyzerTimeSeriesInterface(metaclass=abc.ABCMeta):
    _METRICS_WITH_THRESHOLD = [Metrics.ACCURACY, Metrics.PRECISION_SCORE,
                               Metrics.RECALL_SCORE, Metrics.F1_SCORE,
                               Metrics.F_BETA_SCORE, Metrics.NAB_SCORE,
                               Metrics.FALSE_ALARM_RATE, Metrics.MISS_ALARM_RATE,
                               Metrics.MATTHEWS_COEF]

    def __init__(self,
                 model_name,
                 dataset,
                 available_metrics,
                 metric,
                 result_saving_path,
                 save_graphs_as_png,
                 metrics_with_threshold = None,
                 valid_curves = None
                 ):

        if not isinstance(model_name, str):
            raise TypeError(err_type.format('model_name'))
        elif model_name not in dataset.proposals.keys():
            raise ValueError(err_value.format('model_name', list(dataset.proposals.keys())))

        if not isinstance(result_saving_path, str):
            raise TypeError(err_type.format('result_saving_path'))

        if not isinstance(save_graphs_as_png, bool):
            raise TypeError(err_type.format('save_graphs_as_png'))

        self.model_name = model_name
        self.dataset = dataset
        self._available_metrics = available_metrics
        self._metrics_with_threshold = metrics_with_threshold if metrics_with_threshold is not None else self._METRICS_WITH_THRESHOLD.copy()
        self._valid_curves = valid_curves
        self.default_metric = metric
        self.result_saving_path = result_saving_path
        self.save_graphs_as_png = save_graphs_as_png

        self.beta_score = 0.1

        self._custom_metrics = {}
        self._saved_analysis = {}

    def add_custom_metric(self, custom_metric):
        """Adds a custom metric.

        Parameters
        ----------
        custom_metric : TSCustomMetric
            A custom metric to be added.

        Returns
        -------
        None
        """
        if not isinstance(custom_metric, TSCustomMetric):
            get_root_logger().error(err_type.format("custom_metric"))
            return -1

        if custom_metric.get_name() not in [item.value for item in Metrics]:
            extend_enum(Metrics, custom_metric.get_name().upper().replace(" ", "_"), custom_metric.get_name())

        if custom_metric.is_single_threshold() and Metrics(custom_metric.get_name()) not in self._metrics_with_threshold:
            self._metrics_with_threshold.append(Metrics(custom_metric.get_name()))

        self._custom_metrics[Metrics(custom_metric.get_name())] = custom_metric

        if custom_metric.get_name() not in self._available_metrics:
            self._available_metrics.append(Metrics(custom_metric.get_name()))

    def _get_metrics_with_threshold(self):
        """
        Returns the list of the metrics that use thresholding
        Returns
        -------
        list
        """
        return self._metrics_with_threshold

    def get_custom_metrics(self):
        """Returns all the custom metrics added by the user.

        Returns
        -------
        custom_metrics : dict
            A dictionary with keys the metrics names and as values the metric
            objects.
        """
        return self._custom_metrics
    
    def _is_valid_curve(self, curve):
        """Checks if the curve is valid

        Parameters
        ----------
        curve: Curves
            Curve to check the validity

        Returns
        -------
        bool
            True if the curve is valid
        """
        if curve not in self._valid_curves:
            return False
        return True

    def _compute_metric(self, observations, proposals, y_true, y_score, metric, threshold=0, nab_config={}, evaluation_type=AnomalyMatchingStrategyPointToPoint(), inverse_threshold=False, min_consecutive_samples=1):
        if metric == Metrics.ACCURACY:
            return accuracy(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)
        if metric == Metrics.PRECISION_SCORE:
            return precision_score(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)
        if metric == Metrics.RECALL_SCORE:
            return recall_score(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)
        if metric == Metrics.F1_SCORE:
            return f1_score(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)
        if metric == Metrics.F_BETA_SCORE:
            return f_beta_score(y_true, y_score, threshold, self.beta_score, evaluation_type, inverse_threshold, min_consecutive_samples)
        if metric == Metrics.NAB_SCORE:
            return nab_score(nab_config['windows_index'], proposals, y_score, threshold,
                             nab_config['A_tp'], nab_config['A_fp'], nab_config['A_fn'], inverse_threshold, min_consecutive_samples)
        if metric == Metrics.FALSE_ALARM_RATE:
            return false_alarm_rate(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)
        if metric == Metrics.MISS_ALARM_RATE:
            return miss_alarm_rate(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)
        if metric == Metrics.MATTHEWS_COEF:
            return matthews_correlation_coefficient(y_true, y_score, threshold, evaluation_type, inverse_threshold, min_consecutive_samples)

        # REGRESSION
        if metric == Metrics.MAE:
            return mean_absolute_error(y_true, y_score)
        if metric == Metrics.MSE:
            return mean_squared_error(y_true, y_score)
        if metric == Metrics.RMSE:
            return root_mean_squared_error(y_true, y_score)
        if metric == Metrics.MAPE:
            return mean_absolute_percentage_error(y_true, y_score)
        if metric == Metrics.MARRE:
            return mean_absolute_ranged_relative_error(y_true, y_score)
        if metric == Metrics.OPE:
            return overall_percentage_error(y_true, y_score)
        if metric == Metrics.RMSLE:
            return root_mean_squared_log_error(y_true, y_score)
        if metric == Metrics.SMAPE:
            return symmetric_mean_absolute_percentage_error(y_true, y_score)
        if metric == Metrics.COEFFICIENT_VARIATION:
            return coefficient_of_variation(y_true, y_score)
        if metric == Metrics.COEFFICIENT_DETERMINATION:
            return coefficient_of_determination(y_true, y_score)

        custom_metrics = self.get_custom_metrics()
        if metric not in custom_metrics:
            raise NotImplementedError(err_ts_metric.format(metric))
        else:
            return custom_metrics[metric].evaluate_metric(y_true, y_score, threshold, inverse_threshold, evaluation_type, min_consecutive_samples)

    def analyze_performance(self, metrics=None, show=True):
        if metrics is None:
            metrics = self._available_metrics
        elif not isinstance(metrics, list):
            get_root_logger().error(err_type.format('metrics'))
            return -1
        elif not all(m in self._available_metrics for m in metrics):
            get_root_logger().error(err_value.format('metrics', self._available_metrics))
            return -1

        observations = self.dataset.get_observations(scaled=self._scaler_values[0])

        proposals = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])

        return self._analyze_performance(observations, proposals, metrics, show)

    def analyze_properties(self, properties=None, metric=None):
        available_properties = self.dataset.get_available_properties()
        if len(available_properties) == 0:
            get_root_logger().error('No properties available')
            return -1

        if properties is None:
            properties = available_properties
        elif not all(p in available_properties for p in properties):
            get_root_logger().error(err_value.format('properties', available_properties))
            return -1

        if metric is None:
            metric = self.default_metric
        elif not isinstance(metric, Metrics):
            get_root_logger().error(err_type.format('metric'))
            return -1
        elif metric not in self._available_metrics:
            get_root_logger().error(err_value.format('metric', self._available_metrics))
            return -1
        elif metric == Metrics.NAB_SCORE:
            get_root_logger().error('NAB_SCORE not supported')
            return -1

        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        for p_name in properties:
            results = {'overall': {}}
            results['overall']['all'] = {'value': self._analyze_performance(observations, proposals, [metric], show=False)['micro'][0],
                                         'std': None}
            available_values = self.dataset.get_values_for_property(p_name)
            results['overall'][p_name] = {}
            for p_value in available_values:
                obs = self.dataset.get_observations_for_property_value(p_name, p_value)
                props = proposals.loc[proposals.index.get_level_values(self.dataset._index_gt).isin(obs.index.get_level_values(self.dataset._index_gt))].copy()
                results['overall'][p_name][p_value] = {'value': self._analyze_performance(obs, props, [metric], show=False)['micro'][0],
                                              'std': None}

            make_multi_category_plot(results,
                                     p_name,
                                     available_values,
                                     {'overall': {'display_name': ''}},
                                     "Analysis of {} property".format(p_name),
                                     metric,
                                     self.save_graphs_as_png,
                                     self.result_saving_path,
                                     split_by='meta-annotations',
                                     sort=False)

    @abc.abstractmethod
    def _analyze_performance(self, observations, proposals, metric, show=True):
        pass
