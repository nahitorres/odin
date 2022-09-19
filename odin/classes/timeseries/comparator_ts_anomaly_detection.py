from odin.classes.timeseries import DatasetTSAnomalyDetection, ComparatorTSInterface, AnalyzerTSAnomalyDetection
from odin.classes import Metrics
from odin.classes.timeseries.anomaly_matching_strategies import *
from odin.utils import get_root_logger
from odin.classes.strings import *
from odin.utils.draw_utils import plot_threshold_analysis, plot_models_comparison_on_tp_fp_fn_tn, plot_gain_chart, plot_lift_chart
import numpy as np

class ComparatorTSAnomalyDetection(ComparatorTSInterface):

    _available_metrics = [Metrics.ACCURACY, Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE,
                           Metrics.NAB_SCORE, Metrics.MISS_ALARM_RATE, Metrics.FALSE_ALARM_RATE]

    def __init__(self, 
                 dataset_path,
                 ts_type,
                 anomalies_path,
                 multiple_proposals_path=None, # [("name", "path", proposals_type, threshold, anomaly_evaluation, scaler_bool)]
                 nab_data_percentage=0.1,
                 properties_path=None,
                 csv_separator=',',
                 index_gt='Time',
                 index_proposals='Time',
                 metric=Metrics.F1_SCORE,
                 matching_strategy=AnomalyMatchingStrategyPointToPoint(),
                 min_consecutive_points=1,
                 scaler=None,
                 scaler_values=None,
                 result_saving_path='./result',
                 save_graphs_as_png=False):
        
        proposals_paths = [(m[0], m[1], m[2]) for m in multiple_proposals_path]

        self._default_dataset = DatasetTSAnomalyDetection(dataset_path, 
                                                          ts_type,
                                                          anomalies_path,
                                                          proposals_paths,
                                                          nab_data_percentage,
                                                          properties_path,
                                                          csv_separator, 
                                                          index_gt, 
                                                          index_proposals, 
                                                          result_saving_path, 
                                                          save_graphs_as_png,
                                                          scaler)
        
        super().__init__(ts_type,
                 metric,
                 result_saving_path,
                 save_graphs_as_png)

        # load models analyzers
        for m in multiple_proposals_path:
            model_name, _, _, threshold, anomaly_evaluation = m

            analyzer = AnalyzerTSAnomalyDetection(model_name=model_name,
                                                  dataset=self._default_dataset,
                                                  metric=metric,
                                                  threshold=threshold,
                                                  matching_strategy=matching_strategy,
                                                  min_consecutive_points=min_consecutive_points,
                                                  anomaly_evaluation=anomaly_evaluation,
                                                  scaler_values=scaler_values)
            self.models[model_name] = {"dataset": self._default_dataset,
                                       "analyzer": analyzer}

        
    def analyze_performance_for_threshold(self, metric=None, models=None, show=True):

        """
        It compares the performance of multiple models for different threshold values. 
        The thresholds are scaled between 0 and 1 for all the models.

        Parameters
        ----------
        metric: Metrics, optional
            Indicates the evaluation metric to include in the analysis. If None, use the default one. (default is None)
        models: list of str, optional
            Indicates the models to include in the analysis. If None, include all. (default None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        
        Returns
        -------
        dict
            Returns the thresholds and the correspondig metrics score, for each model.
        """
        if metric is None:
            metric = self.default_metric
        elif not isinstance(metric, Metrics):
            get_root_logger().error(err_type.format('metric'))
            return -1
        elif metric not in self._available_metrics:
            get_root_logger().error(err_value.format('metric', self._available_metrics))
            return -1

        if models is None:
            models = list(self.models.keys())
        elif not all([name in self.models for name in models]):
            get_root_logger().error(err_value.format('models', self.models))
            return -1

        thresholds = np.arange(0, 1.001, 0.05).round(5)
        all_performance = {metric:{}}
        for model in models:
            all_performance[metric][model] = {'results': self.models[model]['analyzer'].analyze_performance_for_threshold([metric], show=False)['results'][metric],
                                               'thresholds': thresholds}
        
        if not show:
            return all_performance
        
        plot_threshold_analysis(all_performance,
                                "Threshold analysis comparison",
                                self.save_graphs_as_png,
                                self.result_saving_path,
                                comparison=True)
    
    def analyze_confusion_matrix(self, models=None, show=True):
        """
        It compares the confusion matrix of multiple models.

        Parameters
        ----------
        models: list of str, optional
            Indicates the models to include in the analysis. If None, include all. (default None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        
        Returns
        -------
        dict
            Returns the confusion matrix for each model.
        
        """
        if models is None:
            models = list(self.models.keys())
        elif not all([name in self.models for name in models]):
            get_root_logger().error(err_value.format('models', self.models))
            return -1
        
        all_performance = {}
        for model in models:
            #tn, fp, fn, tp
            tn, fp, fn, tp = self.models[model]['analyzer'].show_confusion_matrix(show=False).ravel()
            all_performance[model] = {'tp': tp,
                                      'tn': tn,
                                      'fp': fp,
                                      'fn': fn}
        
        if not show:
            return all_performance
        
        plot_models_comparison_on_tp_fp_fn_tn(all_performance, ["TP", "TN", "FP", "FN"], 
                                              "Confusion matrix comparison", "Confusion Matrix",
                                              self.save_graphs_as_png, self.result_saving_path)
        
    def analyze_gain_lift(self, models=None, show=True):
        """
        It compares the gain and lift analysis of multiple models.

        Parameters
        ----------
        models: list of str, optional
            Indicates the models to include in the analysis. If None, include all. (default None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        
        Returns
        -------
        dict
            Returns the gain and lift for each model.
        """


        if models is None:
            models = list(self.models.keys())
        elif not all([name in self.models for name in models]):
            get_root_logger().error(err_value.format('models', self.models))
            return -1
        
        all_performance = {'gain': {},
                           'lift': {}}
        for model in models:
            results = self.models[model]['analyzer'].analyze_gain_lift(show=False)
            all_performance['gain'][model] = results['gain']
            all_performance['lift'][model] = results['lift']
        
        if not show:
            return all_performance
        
        plot_gain_chart(all_performance['gain'],
                        self.save_graphs_as_png, 
                        self.result_saving_path,
                        comparison=True)
        
        plot_lift_chart(all_performance['lift'],
                        self.save_graphs_as_png, 
                        self.result_saving_path,
                        comparison=True)