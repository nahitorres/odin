import abc

from odin.classes import Metrics
from odin.classes.strings import err_type, err_value
from odin.classes.timeseries.metrics import precision_score, recall_score, f1_score, accuracy, nab_score, \
    mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, false_alarm_rate, \
    miss_alarm_rate
from odin.utils import get_root_logger
from odin.utils.draw_utils import make_multi_category_plot


class AnalyzerTimeSeriesInterface(metaclass=abc.ABCMeta):

    def __init__(self,
                 model_name,
                 dataset,
                 available_metrics,
                 metric,
                 result_saving_path,
                 save_graphs_as_png
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
        self.default_metric = metric
        self.result_saving_path = result_saving_path
        self.save_graphs_as_png = save_graphs_as_png

    def _compute_metric(self, observations, proposals, y_true, y_score, metric, threshold=0, nab_config={}):
        if metric == Metrics.ACCURACY:
            return accuracy(y_true, y_score, threshold)
        if metric == Metrics.PRECISION_SCORE:
            return precision_score(y_true, y_score, threshold)
        if metric == Metrics.RECALL_SCORE:
            return recall_score(y_true, y_score, threshold)
        if metric == Metrics.F1_SCORE:
            return f1_score(y_true, y_score, threshold)
        if metric == Metrics.NAB_SCORE:
            return nab_score(nab_config['windows_index'], proposals, y_score, threshold,
                             nab_config['A_tp'], nab_config['A_fp'], nab_config['A_fn'])
        if metric == Metrics.FALSE_ALARM_RATE:
            return false_alarm_rate(y_true, y_score, threshold)
        if metric == Metrics.MISS_ALARM_RATE:
            return miss_alarm_rate(y_true, y_score, threshold)
        if metric == Metrics.MAE:
            return mean_absolute_error(y_true, y_score)
        if metric == Metrics.MSE:
            return mean_squared_error(y_true, y_score)
        if metric == Metrics.RMSE:
            return root_mean_squared_error(y_true, y_score)
        if metric == Metrics.MAPE:
            return mean_absolute_percentage_error(y_true, y_score)

        raise NotImplementedError(metric)

    def analyze_performance(self, metrics=None):
        if metrics is None:
            metrics = self._available_metrics
        elif not isinstance(metrics, list):
            get_root_logger().error(err_type.format('metrics'))
            return -1
        elif not all(m in self._available_metrics for m in metrics):
            get_root_logger().error(err_value.format('metrics', self._available_metrics))
            return -1

        observations = self.dataset.get_observations()

        proposals = self.dataset.get_proposals(self.model_name)

        self._analyze_performance(observations, proposals, metrics)

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
