import abc
from odin.utils import get_root_logger
from odin.classes.strings import err_type, err_value
from odin.utils.draw_utils import plot_performance_comparison


class ComparatorTSInterface(metaclass=abc.ABCMeta):

    def __init__(self, 
                 ts_type,
                 metric,
                 result_saving_path='./result',
                 save_graphs_as_png=False):

        self.ts_type = ts_type
        self.default_metric = metric
        self.result_saving_path = result_saving_path
        self.save_graphs_as_png = save_graphs_as_png

        self.models = {}


    def analyze_performance(self, metrics=None, models=None, show=True):
        if metrics is None:
            metrics = self._available_metrics
        elif not isinstance(metrics, list):
            get_root_logger().error(err_type.format('metrics'))
            return -1
        elif not all(m in self._available_metrics for m in metrics):
            get_root_logger().error(err_value.format('metrics', self._available_metrics))
            return -1

        if models is None:
            models = list(self.models.keys())
        elif not all([name in self.models for name in models]):
            return -1

        all_performance = {}
        for model in models:
            all_performance[model] = self.models[model]['analyzer'].analyze_performance(metrics, show=False)['micro']
        
        if not show:
            return all_performance
        
        plot_performance_comparison(all_performance, [m.value for m in metrics], self.save_graphs_as_png, self.result_saving_path)
        
