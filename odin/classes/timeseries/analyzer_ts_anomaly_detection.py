import math
from collections import defaultdict

import numpy as np
from numbers import Number
import pandas as pd
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

import odin.classes.timeseries.anomaly_definition_strategies as ads
from odin.classes.timeseries.anomaly_definition_strategies.strategy_ts_interface import AnomalyDefinitionStrategyTSInterface
from odin.classes import Metrics, Curves, Errors, CustomError, ErrCombination
from odin.classes.strings import err_type, err_value, \
    err_ts_only_interval_interval, err_analyzer_invalid_curve
from odin.classes.timeseries import DatasetTSAnomalyDetection, TSProposalsType
from odin.classes.timeseries.analyzer_ts_interface import AnalyzerTimeSeriesInterface
from odin.classes.timeseries.anomaly_matching_strategies import *
from odin.classes.timeseries.metrics import precision_recall_curve_values
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_threshold_analysis, \
    display_confusion_matrix, pie_plot, plot_distribution, \
    plot_reliability_diagram, plot_gain_chart, plot_lift_chart, \
    plot_multiple_curves, plot_errors_impact, hist_plot, bar_plot, \
    iou_histogram_diagram


class AnalyzerTSAnomalyDetection(AnalyzerTimeSeriesInterface):
    """Object capable of analysing anomaly detection proposals.
    
    Parameters
    ----------
    model_name : str
        Name of the model
        
    dataset
    
    metric : Metrics, default=Metrics.F1_SCORE
    
   
    threshold : float, default=0.5
        The threshold to be used to compute metrics
        
    matching_strategy: AnomalyMatchingStrategyInterface
    
    min_consecutive_points: int, default = 1
        
    anomaly_evaluation : EvaluatorTSInterface, default=None
        Object used to evaluate errors on the time series. If it is None, the
        absolute error method is used as default.
        
    scaler_values : tuple or None, default=None
        If ground truth values or proposals values must be scaled, this parameter
        must be a tuple of size 2 composed of two bool, in which the first bool
        states if the ground truth must be scaled and the second bool states if
        the proposals must be scaled.
        
    result_saving_path : str, default="./result"
    
    save_graphs_as_png : bool, default=False
    """

    __available_metrics = [Metrics.ACCURACY, Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE, Metrics.F_BETA_SCORE,
                           Metrics.NAB_SCORE, Metrics.MISS_ALARM_RATE, Metrics.FALSE_ALARM_RATE, Metrics.MATTHEWS_COEF]
    __valid_curves = [Curves.ROC_CURVE, Curves.PRECISION_RECALL_CURVE]
    __available_iou_metrics = [Metrics.ACCURACY, Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE, Metrics.F_BETA_SCORE,
                               Metrics.NAB_SCORE, Metrics.MISS_ALARM_RATE, Metrics.FALSE_ALARM_RATE, Metrics.MATTHEWS_COEF]

    def __init__(self,
                 model_name,
                 dataset,
                 metric=Metrics.F1_SCORE,
                 threshold=0.5,
                 matching_strategy=AnomalyMatchingStrategyPointToPoint(),
                 min_consecutive_points=1,
                 anomaly_evaluation=ads.AnomalyDefinitionStrategyTSAE(),
                 scaler_values=None,
                 result_saving_path='./result',
                 save_graphs_as_png=False):

        if not isinstance(dataset, DatasetTSAnomalyDetection):
            raise TypeError(err_type.format('dataset'))

        if not isinstance(metric, Metrics):
            raise TypeError(err_type.format('metric'))
        elif metric not in self.__available_metrics:
            raise ValueError(err_value.format('metric', self.__available_metrics))
            
        if not isinstance(threshold, Number):
            raise TypeError(err_type.format('threshold'))
            
        # Check that the scaler_values is either None (no scaling) or a tuple
        # composed of two booleans. The first states if the gt must
        # be scaled. The second states if the predictions must be scaled.
        if scaler_values is not None and not isinstance(scaler_values, tuple):
            raise TypeError(err_type.format('scaler_values'))
        elif scaler_values is not None and len(scaler_values) != 2:
            raise ValueError(err_value.format('scaler_values', 'None or tuple(bool, bool)'))
        elif scaler_values is not None and (not isinstance(scaler_values[0], bool) or not isinstance(scaler_values[1], bool)):
            raise ValueError(err_value.format('scaler_values', 'None or tuple(bool, bool)'))

        if not isinstance(matching_strategy, AnomalyMatchingStrategyInterface):
            raise TypeError(err_type.format('matching_strategy'))

        if not isinstance(min_consecutive_points, int):
            raise TypeError(err_type.format('min_consecutive_points'))
        elif min_consecutive_points < 1:
            raise ValueError(err_value.format('min_consecutive_points', '>0'))
        
        if not isinstance(anomaly_evaluation, AnomalyDefinitionStrategyTSInterface):
            raise TypeError(err_type.format('anomaly_evaluation'))
        
        self._threshold = threshold
        self._anomaly_evaluation = anomaly_evaluation
        self._scaler_values = scaler_values if scaler_values is not None else (False, False)

        self._matching_strategy = matching_strategy
        self._min_consecutive_points = min_consecutive_points

        super().__init__(model_name, dataset, self.__available_metrics, metric, result_saving_path, save_graphs_as_png, valid_curves=self.__valid_curves)

        if not self.dataset._analysis_available:
            raise Exception('No anomaly labels available')
        
        if not self._check_selected_strategy(anomaly_evaluation):
            raise ValueError("Invalid proposals file format.")
        
        self._gt_truncation = ads.OverlappingTruncation.NEITHER

        if isinstance(anomaly_evaluation, ads.AnomalyDefinitionStrategyTSOverlappingWindows):
            self._gt_truncation = anomaly_evaluation.truncation_flag
        elif isinstance(anomaly_evaluation, ads.AnomalyDefinitionStrategyTSGaussianDistribution):
            self._gt_truncation = ads.OverlappingTruncation.HEAD

        self.proposals_type = self.dataset.get_proposals_type(self.model_name)

        self.nab_params = {'windows_index': self.dataset.nab_config['windows_index'],
                           'A_tp': 1,
                           'A_fp': 0.1,
                           'A_fn': 1}

        self._all_y_scores = None

    def change_strategy(self, anomaly_evaluation):
        """Change the evaluation strategy.
        
        Parameters
        ----------
        anomaly_evaluation : AnomalyDefinitionStrategyTSInterface
            The new evaluation strategy to be used.

        Returns
        -------
        None
        """
        if not isinstance(anomaly_evaluation, AnomalyDefinitionStrategyTSInterface):
            get_root_logger().error(err_type.format('anomaly_evaluation'))
            return -1

        if not self._check_selected_strategy(anomaly_evaluation):
            get_root_logger().error("Invalid proposals file for evaluation.")
            return -1
        self._anomaly_evaluation = anomaly_evaluation

        if isinstance(anomaly_evaluation, ads.AnomalyDefinitionStrategyTSOverlappingWindows):
            self._gt_truncation = anomaly_evaluation.truncation_flag

        
    def _check_selected_strategy(self, anomaly_evaluation, is_init=False):
        """Checks if the selected strategy is possible with the proposals file format.
        
        Parameters
        ----------
        anomaly_evaluation : AnomalyDefinitionStrategyTSInterface
            The new evaluation strategy to be used.
            
        is_init : bool, default=False
            States if the control has been called form init or from another
            method.

        Returns
        -------
        is_strategy_valid : bool
            States if the strategy can be used.
        """
        proposals = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])
        return anomaly_evaluation.is_proposals_file_valid(proposals)

    def analyze_performance_for_threshold(self, metrics=None, show=True):
        """
        It computes multiple evaluation metrics for different threshold values.

        Parameters
        ----------
        metrics: list of Metrics, optional
            Indicates the evaluation metrics to include in the analysis. If None, include all (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        
        Returns
        -------
        dict
            Returns the thresholds and the correspondig metrics score
        """
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

        y_true = observations['anomaly'].values
        
        if self._all_y_scores is None:
            self._all_y_scores = self._get_y_score(observations, proposals)
        y_score = self._all_y_scores.copy()
            
        y_true, proposals = self._cut_y_true(y_true, y_score, proposals, True)

        max_value, step_value = (1.001, 0.05) if self.proposals_type == TSProposalsType.LABEL else (np.max(y_score.reshape(-1)), np.max(y_score.reshape(-1))/21)

        thresholds = np.arange(0, max_value, step_value).round(5)

        results = defaultdict(list)
        for threshold in thresholds:
            for metric in metrics:
                results[metric].append(self._compute_metric(observations, proposals, y_true, y_score, metric,
                                                            threshold, self.nab_params, inverse_threshold=self._anomaly_evaluation.is_inverse_threshold(), evaluation_type=self._matching_strategy, min_consecutive_samples=self._min_consecutive_points))
        if not show:
            return {"thresholds": thresholds,
                    "results": results}
                    
        plot_threshold_analysis({'x': thresholds,
                                 'y': results},
                                "Threshold analysis",
                                self.save_graphs_as_png,
                                self.result_saving_path)

    def show_confusion_matrix(self, show=True):
        """
        It calculates the confusion matrix.

        Parameters
        ----------
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        
        Returns
        -------
        numpy.ndarray
            Returns the confusion matrix
        
        """
        observations = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])

        y_true = observations['anomaly'].values
        y_score = self._get_y_score(observations, proposals)
        
        y_true = self._cut_y_true(y_true, y_score)
        
        cm = self._matching_strategy.get_confusion_matrix(y_true, y_score, self._threshold, inverse_threshold=self._anomaly_evaluation.is_inverse_threshold())
        
        if not show:
            return cm

        display_confusion_matrix(np.array([cm]),
                                 ['Anomaly'],
                                 None,  # properties filter
                                 self.save_graphs_as_png,
                                 self.result_saving_path)

    def analyze_false_positive_errors(self, threshold=None, bins=8, show=True, metric = None, error_combination = None, parameters_dicts = None):
        """
        This function allows analyzing False Positive errors by dividing them into mutually-exclusive categories. 
        
        Parameters
        ----------
        threshold:  float
            Threshold to apply to the scores vector to obtain binary predictions
        bins:  int
            Number of bins to use in the visualization (default: 8)
        show: bool
            If true, the plots are shown, otherwise the computed values are returned
        metric: Metric
            Metric to use for evaluating the FP impact
        error_combination: ErrCombination
            Errors combination
        parameters_dicts: list
            List of dictionaries containing the parameters to be used when the error is computed
        
        Returns
        -------
        categories_dict: dict
            Dictionary that, for each category, contains the number of FP errors
        distances: dict
            Dictionary that, for each category, contains the distance wrt the closest GT anomaly
        errors_index: dict
            Dictionary that, for each category, lists the indexes of the errors in the DataFrame
        matching: DataFrame
            DataFrame containing an 'eval' column where 1 indicates the presence of a GT anomaly, -1 the presence of a FP, and 0 every other point
        """
        
        
        if metric is None:
            metric = self.default_metric
        elif not isinstance(metric, Metrics):
            get_root_logger().error(err_type.format('metric'))
            return -1
      
        if threshold is None or not isinstance(threshold, float):
            threshold = self._threshold
            
        if not isinstance(error_combination, ErrCombination):
            raise TypeError(err_type.format('error_combination'))
            
        
            
        if parameters_dicts is not None:
            if not isinstance(parameters_dicts, list):
                get_root_logger().error(err_type.format('parameters_dicts'))
                return -1
        
        observations = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])
        y_true = observations['anomaly'].values
        y_score = self._get_y_score(observations, proposals)
        
        
        categories_dict, distances, errors_index, matching = None, None, None, None
        
        
        categories_dict, distances, errors_index, matching = error_combination.compute_errors_combination(y_true, y_score, threshold, observations, parameters_dicts)
        

        if not show:
            return categories_dict, distances, errors_index, matching

        categories = list(categories_dict.keys())
        
        pie_plot([categories_dict[k] for k in categories],
                 categories,
                 'False Positive errors',
                 self.result_saving_path,
                 self.save_graphs_as_png)

        plot_distribution(distances,
                          bins,
                          'Steps from nearest anomaly',
                          'Error distance distribution',
                          self.save_graphs_as_png,
                          self.result_saving_path,
                          categories)
        
        is_inverse_threshold = self._anomaly_evaluation.is_inverse_threshold()
        
        plot_errors_impact(self._analyze_errors_impact(matching, errors_index, metric, is_inverse_threshold),
                            metric,
                          self.save_graphs_as_png,
                          self.result_saving_path,
                          categories)
    
    def _analyze_errors_impact(self, matching, errors_index, metric, is_inverse_threshold):
        tot_score = self._compute_metric(None, None, matching["y_true"], matching["y_pred"], metric,
                                                            self._threshold, self.nab_params, inverse_threshold = is_inverse_threshold)
        errors_impact = {}

        for e in errors_index.keys():
            tmp = matching.copy()
            tmp.loc[tmp.index.isin(errors_index[e]), "y_pred"] = 0
            score = self._compute_metric(None, None, matching["y_true"], tmp["y_pred"], metric,
                                                            self._threshold, self.nab_params)
            errors_impact[e] = score - tot_score
        
        return errors_impact

    def analyze_reliability(self, num_bins=10, min_threshold=0, show=True):
        """
        It provides the reliability analysis by showing the distribution of the proposals among different confidence values and the corresponding confidence calibration.

        Parameters
        ----------
        num_bins: int, optional
            The number of bins the confidence values are split into. (default is 10)
        min_threshold: int, optional
            The minimum confidence score considered in the analysis (default is 0)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if not isinstance(num_bins, int):
            get_root_logger().error(err_type.format("num_bins"))
            return -1
        elif not 1 < num_bins <= 50:
            get_root_logger().error(err_value.format("num_bins", "1 < num_bins <= 50"))
            return -1

        if not isinstance(min_threshold, Number):
            get_root_logger().error(err_type.format("min_threshold"))
            return -1
        elif not (0 <= min_threshold <= 1):
            get_root_logger().error(err_value.format("min_threshold", "0 <= min_threshold <= 1"))
            return -1

        observations = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])

        y_true = observations['anomaly'].values

        if self._all_y_scores is None:
            self._all_y_scores = self._get_y_score(observations, proposals)
        y_score = self._all_y_scores.copy()

        y_true = self._cut_y_true(y_true, y_score)
        y_pred = np.ones(len(y_true))

        if self.proposals_type == TSProposalsType.REGRESSION:
            scaler = MinMaxScaler()
            if y_score.ndim > 1:
                y_score = np.mean(y_score, axis=1)
            y_score = scaler.fit_transform(y_score.reshape(-1, 1))
            y_score = y_score.reshape(-1)
            if self._anomaly_evaluation.is_inverse_threshold():
                y_score = 1-y_score

        if min_threshold > 0:
            ix = np.where(y_score >= min_threshold)[0]
            y_true = y_true[ix]
            y_pred = y_pred[ix]
            y_score = y_score[ix]

        result = self._calculate_reliability(y_true, y_pred, y_score, num_bins)

        if not show:
            return result

        plot_reliability_diagram(result,
                                 self.save_graphs_as_png,
                                 self.result_saving_path,
                                 is_classification=True)

    def analyze_gain_lift(self, show=True):
        """
        It provides the gain and lift analysis.

        Parameters
        ----------
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        observations = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])

        if self._all_y_scores is None:
            self._all_y_scores = self._get_y_score(observations, proposals)
        y_score = self._all_y_scores.copy()
        if y_score.ndim > 1:
            y_score = np.mean(y_score, axis=1)
        y_true = observations['anomaly'].values
        y_true = self._cut_y_true(y_true, y_score)

        match = pd.DataFrame({'anomaly': y_true,
                              'confidence': y_score})

        match = match.sort_values(by='confidence', ascending=self._anomaly_evaluation.is_inverse_threshold())
        labels = match['anomaly'].values

        step = round(len(match.index)/10)
        n_total = len(match.loc[match['anomaly'] == 1].index)
        n_lift = n_total / 10

        cum_sum = 0
        gain = []
        lift = []
        for i in range(0, 10):
            start = i*step
            end = start+step if (start+step) <= len(match.index) else len(match.index)
            cum_sum += sum(labels[start:end])
            gain.append(cum_sum / n_total)
            lift.append(cum_sum/(n_lift*(i+1)))

        if not show:
            return {'gain': gain,
                    'lift': lift}

        plot_gain_chart({'model': self.model_name, 'values': gain},
                        self.save_graphs_as_png,
                        self.result_saving_path)

        plot_lift_chart({'model': self.model_name, 'values': lift},
                        self.save_graphs_as_png,
                        self.result_saving_path)

    def _calculate_reliability(self, y_true, y_pred, y_score, num_bins):
        """
        Calculates the reliability

        Parameters
        ----------
        y_true:  array-like
            Ground Truth
        y_pred:  array-like
            Predictions label
        y_score: array-like
            Predictions scores
        num_bins: int
            Number of bins used to split the confidence values

        Returns
        -------
        dict : {'values': bin_accuracies, 'gaps': gaps, 'counts': bin_counts, 'bins': bins,
                  'avg_value': avg_acc, 'avg_conf': avg_conf, 'ece': ece, 'mce': mce}
        """
        bins = np.linspace(0.0, 1.0, num_bins + 1)

        indices = np.digitize(y_score, bins, right=True)

        bin_accuracies = np.zeros(num_bins, dtype=float)
        bin_confidences = np.zeros(num_bins, dtype=float)
        bin_counts = np.zeros(num_bins, dtype=int)

        np_y_true = np.array(y_true)
        np_y_pred = np.array(y_pred)
        np_y_score = np.array(y_score)

        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(np_y_true[selected] == np_y_pred[selected])
                bin_confidences[b] = np.mean(np_y_score[selected])
                bin_counts[b] = len(selected)

        avg_acc = self._compute_metric(None, None, y_true, y_score, Metrics.ACCURACY, self._threshold, inverse_threshold=False, evaluation_type=self._matching_strategy, min_consecutive_samples=self._min_consecutive_points)

        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)
        if np.isnan(avg_conf):
            avg_conf = 0

        gaps = bin_confidences - bin_accuracies
        ece = np.sum(np.abs(gaps) * bin_counts) / np.sum(bin_counts)
        if np.isnan(ece):
            ece = 0
        mce = np.max(np.abs(gaps))
        if np.isnan(mce):
            mce = 0

        result = {'values': bin_accuracies, 'gaps': gaps, 'counts': bin_counts, 'bins': bins,
                  'avg_value': avg_acc, 'avg_conf': avg_conf, 'ece': ece, 'mce': mce}
        return result

    def _analyze_performance(self, observations, proposals, metrics, show=True):
        observations_all = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals_all = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])

        indexes = np.where(observations.index.isin(observations_all.index))[0]
        y_true = observations_all['anomaly'].values
        if self.proposals_type == TSProposalsType.LABEL:
            y_score = proposals_all['confidence'].values
        else:
            self._all_y_scores = self._get_y_score(observations_all, proposals_all)
            
            y_score = self._all_y_scores.copy()
            if y_true.shape[0] != y_score.shape[0]:
                to_remove = int(y_true.shape[0] - y_score.shape[0]) if self._gt_truncation != ads.OverlappingTruncation.BOTH else int((y_true.shape[0] - y_score.shape[0])/2)
                if self._gt_truncation == ads.OverlappingTruncation.HEAD:
                    indexes = indexes[to_remove:]
                elif self._gt_truncation == ads.OverlappingTruncation.TAIL:
                    indexes = indexes[:to_remove]
                else: # BOTH
                    indexes = indexes[to_remove:-to_remove]

                indexes = indexes - to_remove

            y_true, proposals = self._cut_y_true(y_true, y_score, proposals, True)
        
            
        
        results = [self._compute_metric(observations, proposals, y_true[indexes], y_score[indexes], metric, self._threshold, self.nab_params, inverse_threshold=self._anomaly_evaluation.is_inverse_threshold(), evaluation_type=self._matching_strategy, min_consecutive_samples=self._min_consecutive_points) for metric in metrics]
        if not show:
            return {'micro': results}

        data = {}
        for r, m in zip(results, metrics) :
            data[m.value] = [r]
        data = pd.DataFrame(data, index=["micro"])
        data.index.name = "label"
        return data

    def _get_y_score(self, observations, proposals):
        """Get the scores of the proposals.
        
        Parameters
        ----------
        observations : DataFrame
            The observations.
            
        proposals : DataFrame
            The proposals.

        Returns
        -------
        scores : ndarray
            The scores computed by the model.
        """
        return self._anomaly_evaluation.get_anomaly_scores(observations, proposals)

    def _cut_y_true(self, y_true, y_score, proposals=None, also_proposals=False):
        """Cut y_score and proposals if evaluator is overlapping.
        
        Parameters
        ----------
        proposals : DataFrame
            The proposals to cut.
        
        also_proposals : bool
            Whether to cut also proposals.

        Returns
        -------
        y_score, proposals: ndarray, DataFrame
            Cut array and DataFrame.
        """
        if y_true.shape[0] != y_score.shape[0]:
            to_remove = (y_true.shape[0] - y_score.shape[0])
            if self._gt_truncation == ads.OverlappingTruncation.BOTH:
                per_side = int(to_remove / 2)
                y_true = y_true[per_side:-per_side]
                if also_proposals:
                    proposals = proposals.iloc[per_side:-per_side]
            elif self._gt_truncation == ads.OverlappingTruncation.TAIL:
                y_true = y_true[:-to_remove]
                if also_proposals:
                    proposals = proposals.iloc[:-to_remove]
            elif self._gt_truncation == ads.OverlappingTruncation.HEAD:
                y_true = y_true[to_remove:]
                if also_proposals:
                    proposals = proposals.iloc[to_remove:]
                
        if also_proposals:
            return y_true, proposals
        else:
            return y_true

    def analyze_curve(self, curve=Curves.PRECISION_RECALL_CURVE, show=True):
        if not isinstance(curve, Curves):
            get_root_logger().error(err_type.format("curve"))
            return -1
        elif not self._is_valid_curve(curve):
            get_root_logger().error(err_analyzer_invalid_curve.format(curve))
            return -1
        
        if not isinstance(show, bool):
            get_root_logger().error(err_type.format("show"))
            return -1
        
        if isinstance(self._anomaly_evaluation, ads.AnomalyDefinitionStrategyTSOverlappingWindows):
            get_root_logger().error("Overlapping windows give a vector of scores to a point. Therefore, curves cannot be drawn.")
            return -1
            
        x, y, auc_ = self._compute_curve(curve)
    
        curve_results = {
            curve: {
                "auc": auc_,
                "x": x,
                "y": y
            }
        }
        display_names = {
            curve: {
                "display_name": "overall"
            }
        }

        self._saved_analysis[curve.value] = { "auc": auc_, "x": x, "y": y }
            
        if show:
            plot_multiple_curves(curve_results, curve, display_names, self.save_graphs_as_png, self.result_saving_path)
        else:
            return curve_results
        
    def _compute_curve(self, curve):
        observations = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals = self.dataset.get_proposals(self.model_name,
                                               scaled=self._scaler_values[1])
    
        y_true = observations["anomaly"].values
        y_score = self._get_y_score(observations, proposals)
        y_true = self._cut_y_true(y_true, y_score, also_proposals=False)
        
        # invert scores such that the evaluation can be performed in standard way
        if self._anomaly_evaluation.is_inverse_threshold():
            y_score -= np.min(y_score)
            y_score = np.max(y_score) - np.array(y_score)
    
        if curve == Curves.ROC_CURVE:
            fpr, tpr, threshold = roc_curve(y_true, y_score)
            auc_ = auc(fpr, tpr)
            return fpr, tpr, auc_
        elif curve == Curves.PRECISION_RECALL_CURVE:
            prec, rec, threshold = precision_recall_curve(y_true, y_score)
            auc_ = auc(rec, prec)
            return rec, prec, auc_

    def analyze_performance_for_iou_threshold(self, metrics = None, granularity = 10, show = True):
        """Analyze the performance of the metrics for different iou thresholds.
        
        Parameters
        ----------
        metrics : list[Metrics], default=None
            The list of the metrics to assess for iou threshold.
        
        granularity : int, default=10
            The number of iou thresholds to analyze in the interval (0,1]. It
            must be at least 3 and no more than 50.
        
        show : bool, default=True
            Whether to show the results or to return them.

        Returns
        -------
        results : dict
            The results of the analysis, if them must be returned instead of
            being plotted.
        """
        if metrics is not None and not isinstance(metrics, list):
            get_root_logger().error(err_type.format('metrics'))
            return -1
        elif metrics is not None and not all(m in self.__available_iou_metrics for m in metrics):
            get_root_logger().error(err_value.format('metrics', self.__available_iou_metrics))
            return -1
        
        if not isinstance(self._matching_strategy, AnomalyMatchingStrategyIntervalToInterval):
            get_root_logger().error(err_ts_only_interval_interval)
            return -1
        
        if not isinstance(granularity, Number):
            get_root_logger().error(err_type.format('granularity'))
            return -1
        elif granularity < 3 or granularity > 50:
            get_root_logger().error(err_value.format('granularity', "Any integer in [3, 50]"))
            return -1
        elif math.ceil(granularity) != int(granularity) or math.floor(granularity) != int(granularity):
            get_root_logger().error(err_value.format("granularity", "granularity must be an integer number, also values like 5.0 are accepted"))
            return -1
        
        granularity = int(granularity)
        
        if not isinstance(show, bool):
            get_root_logger().error(err_type.format('show'))
            return -1
            
        if metrics is None:
            metrics = self.__available_iou_metrics
            
        observations = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])
    
        y_true = observations['anomaly'].values
            
        if self._all_y_scores is None:
            self._all_y_scores = self._get_y_score(observations, proposals)
        y_score = self._all_y_scores.copy()
            
        y_true, proposals = self._cut_y_true(y_true, y_score, proposals, True)
    
        iou_thresholds = np.linspace(0, 1, granularity)
        iou_thresholds = iou_thresholds[1:]
    
        results = defaultdict(list)
        for iou_threshold in iou_thresholds:
            for metric in metrics:
                matching_copy = AnomalyMatchingStrategyIntervalToInterval(iou_threshold)
                results[metric].append(self._compute_metric(observations,
                                                            proposals,
                                                            y_true,
                                                            y_score,
                                                            metric,
                                                            self._threshold,
                                                            self.nab_params,
                                                            evaluation_type=matching_copy,
                                                            inverse_threshold=self._anomaly_evaluation.is_inverse_threshold(),
                                                            min_consecutive_samples=self._min_consecutive_points))
    
        if not show:
            return {"thresholds": iou_thresholds,
                    "results": results}
                        
        plot_threshold_analysis({'x': iou_thresholds,
                                 'y': results},
                                "IOU Threshold analysis",
                                self.save_graphs_as_png,
                                self.result_saving_path)
            
    # TODO: DISCUSS REFACTORING static? could be a util function?
    def get_windows_tuples(self, observations, anomalies):
        """Get a list of lists of dimension 3 containing the information of windows.
        
        Parameters
        ----------
        observations : DataFrame
            The observations as a dataframe.
        
        anomalies : ndarray
            The array of the anomalies of the dataset in which 1 identifies an
            anomaly and 0 identifies a normal data point.

        Returns
        -------
        anomaly_windows : list
            A list of lists of dimension 3 in ordered as: index at which anomaly
            starts, index at which anomaly ends and length of the anomaly.
        """
        # get start and end indices of all anomaly windows
        anomaly_windows = []
        is_inside = False
        start = None
        length = 0
        end = None
        for idx, elem in enumerate(anomalies.reshape(-1).tolist()):
            if elem == 1:
                # TODO: DISCUSS REFACTORING when an anomaly is last point I assume that granularity is constant and use second-last with last
                if not is_inside:
                    try:
                        length = pd.to_timedelta(pd.to_datetime(observations.index[idx + 1]) - pd.to_datetime(observations.index[idx])).seconds / 60
                    except IndexError:
                        length = pd.to_timedelta(pd.to_datetime(observations.index[idx]) - pd.to_datetime(observations.index[idx - 1])).seconds / 60
                    
                    start = observations.index[idx]
                    is_inside = True
                else:
                    try:
                        length += pd.to_timedelta(pd.to_datetime(observations.index[idx + 1]) - pd.to_datetime(observations.index[idx])).seconds / 60
                    except IndexError:
                        length += pd.to_timedelta(pd.to_datetime(observations.index[idx]) - pd.to_datetime(observations.index[idx - 1])).seconds / 60
            else:
                if is_inside:
                    end = observations.index[idx - 1]
                    anomaly_windows.append([start, end, length])
                    is_inside = False

        # handle anomalies on the end
        if is_inside:
            end = observations.index[idx - 1]
            anomaly_windows.append([start, end, length])

        return anomaly_windows

    # TODO: DISCUSS REFACTORING
    def analyze_iou_distribution(self, nbins, threshold = None, show = True):
        """Plots the histogram of TP anomaly windows differences with GT.
        
        Parameters
        ----------
        nbins : int
            The number of bins in the histogram.
        
        threshold : float, default=None
            The threshold to identify a point as anomalous. If None, that of the
            analyzer is used.
        
        show : bool, default=True
            Whether to show the results or to return them.

        Returns
        -------
        results: dict
            If show is True, the dictionary with results of the histogram plot.
        """
        if threshold is None:
            threshold = self._threshold
        elif not isinstance(threshold, Number):
            get_root_logger().error(err_type.format("threshold"))
            return -1
        
        if not isinstance(nbins, Number):
            get_root_logger().error(err_type.format("nbins"))
            return -1
        elif math.ceil(nbins) != int(nbins) or math.floor(nbins) != int(nbins):
            get_root_logger().error(err_value.format("nbins", "nbins must be an integer number, also values like 2.0 are accepted"))
            return -1
        elif nbins <= 1 or nbins > 50:
            get_root_logger().error(err_value.format("nbins", "[2, 50]"))
            return -1
        
        if not isinstance(show, bool):
            get_root_logger().error(err_type.format("show"))
            return -1
        
        nbins = int(nbins)

        observations = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])

        y_pred = self._get_model_predictions(observations, proposals, threshold)

        if np.sum(y_pred) == 0:
            get_root_logger().error("The model does not predict anomalies")
        else:
            gt_windows = self.get_windows_tuples(observations, observations["anomaly"].values)
            pr_windows = self.get_windows_tuples(observations, y_pred)

            # TODO: what if the predicted anomaly is so long that it contains multiple gt windows?
            # compute the histogram
            results = defaultdict(list)
            iou_values = []
            for window in pr_windows:
                is_window_found = False
                for gt_window in gt_windows:
                    # this window is the one to which the prediction refers to
                    # two windows overlap if they have at least one point in common
                    # GT: 0011100 0011100 0011100 00111100
                    # PR: 0110000 0001110 0011100 00011000
                    if window[0] <= gt_window[0] <= window[1] or window[0] <= gt_window[1] <= window[1] or gt_window[0] <= window[0] <= gt_window[1] or gt_window[0] <= window[1] <= gt_window[1]:
                        low = window[0] if window[0] >= gt_window[0] else gt_window[0]
                        high = window[1] if window[1] <= gt_window[1] else gt_window[1]
                        intersection = np.sum(observations.loc[low:high, "anomaly"].values)
                        gt = observations.loc[gt_window[0]:gt_window[1], "anomaly"].values.shape[0]
                        pred = observations.loc[window[0]:window[1], "anomaly"].values.shape[0]
                        union = gt + pred - intersection
                        iou = intersection / union
                        iou_values.append(iou)
                        is_window_found = True
                        break
                
                if not is_window_found:
                    iou_values.append(0.0)
                        
            results["iou_values"] = iou_values
            if not show:
                return results

            iou_histogram_diagram(iou_values, nbins)

    # TODO: DISCUSS REFACTORING
    def analyze_true_predicted_difference_distribution(self, nbins, threshold = None, iou_threshold = None, show = True):
        """Plots the histogram of TP anomaly windows differences with GT.
        
        Parameters
        ----------
        nbins : int
            The number of bins in the histogram.
        
        threshold : float, default=None
            The threshold to identify a point as anomalous. If None, that of the
            analyzer is used.
        
        iou_threshold : float, default=None
            The IOU needed to consider a predicted window a TP. If None, that
            of the specified matching strategy is used.
        
        show : bool, default=True
            Whether to show the results or to return them.

        Returns
        -------
        results: dict
            If show is True, the dictionary with results of the histogram plot.
        """
        if threshold is None:
            threshold = self._threshold
        elif not isinstance(threshold, Number):
            get_root_logger().error(err_type.format("threshold"))
            return -1
        
        if not isinstance(self._matching_strategy, AnomalyMatchingStrategyIntervalToInterval):
            get_root_logger().error(err_ts_only_interval_interval)
            return -1
        
        if iou_threshold is None:
            iou_threshold = self._matching_strategy.get_iou_threshold()
        elif not isinstance(iou_threshold, Number):
            get_root_logger().error(err_type.format("iou_threshold"))
            return -1
        elif not 0 < iou_threshold <= 1:
            get_root_logger().error(err_value.format("iou_threshold", "(0,1]"))
            return -1
        
        if not isinstance(nbins, Number):
            get_root_logger().error(err_type.format("nbins"))
            return -1
        elif nbins <= 1 or nbins > 50:
            get_root_logger().error(err_value.format("nbins", "[2, 50]"))
            return -1
        elif math.ceil(nbins) != int(nbins) or math.floor(nbins) != int(nbins):
            get_root_logger().error(err_value.format("nbins", "nbins must be an integer number, also values like 5.0 are accepted"))
            return -1
        
        nbins = int(nbins)
        
        if not isinstance(show, bool):
            get_root_logger().error(err_type.format("show"))
            return -1
        
        bar_colors = ["lightgreen", "khaki", "indianred"]
        
        observations = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals = self.dataset.get_proposals(self.model_name, scaled=self._scaler_values[1])

        if nbins % 2 == 0:
            nbins += 1
    
        y_pred = self._get_model_predictions(observations, proposals, threshold)
    
        if np.sum(y_pred) == 0:
            get_root_logger().error("The model does not predict anomalies")
            return -1
        else:
            gt_windows = self.get_windows_tuples(observations, observations["anomaly"].values)
            pr_windows = self.get_windows_tuples(observations, y_pred)

            # TODO: what if the predicted anomaly is so long that it contains multiple gt windows?
            # compute the histogram
            results = defaultdict(list)
            differences = []
            for window in pr_windows:
                for gt_window in gt_windows:
                    # this window is the one to which the prediction refers to
                    # two windows overlap if they have at least one point in common
                    # GT: 0011100 0011100 0011100 00111100
                    # PR: 0110000 0001110 0011100 00011000
                    if window[0] <= gt_window[0] <= window[1] or window[0] <= gt_window[1] <= window[1] or gt_window[0] <= window[0] <= gt_window[1] or gt_window[0] <= window[1] <= gt_window[1]:
                        low = window[0] if window[0] >= gt_window[0] else gt_window[0]
                        high = window[1] if window[1] <= gt_window[1] else gt_window[1]
                        intersection = np.sum(observations.loc[low:high, "anomaly"].values)
                        gt = observations.loc[gt_window[0]:gt_window[1], "anomaly"].values.shape[0]
                        pred = observations.loc[window[0]:window[1], "anomaly"].values.shape[0]
                        union = gt + pred - intersection
                        iou = intersection / union

                        if iou > iou_threshold:
                            difference = gt - pred
                            differences.append(difference)
                            break

            if len(differences) == 0:
                get_root_logger().warn("The model does not predict windows with that IOU threshold")
                return -1

            results["tp_differences"] = differences
            if not show:
                return results
            
            min_diff = min(differences)
            max_diff = max(differences)
            
            if abs(min_diff) > abs(max_diff):
                range_ = (-abs(min_diff), abs(min_diff))
            else:
                range_ = (-abs(max_diff), abs(max_diff))
            
            colors = []
            for i in range(nbins):
                if i == (nbins - 1)/2:
                    colors.append(bar_colors[0])
                elif i < (nbins -1)/4 or i > (nbins -1)/2 + (nbins -1)/4:
                    colors.append(bar_colors[2])
                else:
                    colors.append(bar_colors[1])

            hist_plot(differences,
                      nbins,
                      range_,
                      colors,
                      title="Distribution",
                      y_label="Number of True Positive anomalies",
                      x_label="Anomaly duration difference of True Positives",
                      tight_layout=True)
    
    # TODO: DISCUSS REFACTORING
    def analyze_true_predicted_distributions(self, groups = 4, threshold = None, show = True):
        """Plots a bar plot with the duration of GT windows and predicted windows.
        
        Parameters
        ----------
        groups : int, default=4
            The number of groups to be used (number of columns)
        
        threshold : float, default=None
            The threshold to identify a point as anomalous. If None, that of the
            analyzer is used.
        
        show : bool, default=True
            Whether to show the results or to return them.

        Returns
        -------
        results: dict
            If show is True, the dictionary with results of the bar plot.
        """
        bars_width = 0.8
        
        if threshold is None:
            threshold = self._threshold
        elif not isinstance(threshold, Number):
            get_root_logger().error(err_type.format("threshold"))
            return -1
            
        if not isinstance(groups, Number):
            get_root_logger().error(err_type.format("groups"))
            return -1
        elif groups <= 0 or groups > 50:
            get_root_logger().error(err_value.format("groups", "[1, 50]"))
            return -1
        elif math.ceil(groups) != int(groups) or math.floor(groups) != int(groups):
            get_root_logger().error(err_value.format("groups", "groups must be an integer number, also values like 2.0 are accepted"))
            return -1
        
        groups = int(groups)
            
        if not isinstance(bars_width, float):
            get_root_logger().error(err_type.format("bars_width"))
            return -1
        elif bars_width <= 0:
            get_root_logger().error(err_value.format("bars_width", "(0, infinity)"))
            return -1
        
        if not isinstance(show, bool):
            get_root_logger().error(err_type.format("show"))
            return -1
            
        observations = self.dataset.get_observations(scaled=self._scaler_values[0])
        proposals = self.dataset.get_proposals(self.model_name,
                                               scaled=self._scaler_values[1])
            
        y_pred = self._get_model_predictions(observations, proposals, threshold)

        if np.sum(y_pred) == 0:
            get_root_logger().error("The model does not predict anomalies")
        else:
            gt_windows = self.get_windows_tuples(observations, observations["anomaly"].values)
            pr_windows = self.get_windows_tuples(observations, y_pred)

            results = defaultdict(list)
            
            # compute the bar plot
            np_windows = np.array(gt_windows)
            np_predictions = np.array(pr_windows)
            max_val = max(np.max(np_windows[:, 2]), np.max(np_predictions[:, 2]))
            divisors = np.linspace(0, max_val, groups + 1)
            text_groups = []
            pred_groups = []
            gt_groups = []
            pred_lengths = np_predictions[:, 2]
            gt_lengths = np_windows[:, 2]
            for i in range(divisors.shape[0] - 1):
                lower = int(divisors[i])
                higher = int(divisors[i + 1])
    
                # compute predicted in this group
                if i < divisors.shape[0] - 2:
                    pred_h = pred_lengths < higher
                else:
                    pred_h = pred_lengths <= higher
                pred_l = pred_lengths >= lower
                group_pred = np.sum(pred_h & pred_l)
    
                # compute gt in this group
                if i < divisors.shape[0] - 2:
                    gt_h = gt_lengths < higher
                else:
                    gt_h = gt_lengths <= higher
                gt_l = gt_lengths >= lower
                group_gt = np.sum(gt_h & gt_l)
    
                if i < divisors.shape[0] - 2:
                    text_groups.append("[{}-{})".format(lower, higher))
                else:
                    text_groups.append("[{}-{}]".format(lower, higher))
        
                pred_groups.append(np.sum(group_pred))
                gt_groups.append(np.sum(group_gt))

            results["groups"] = text_groups
            results["predictions"] = pred_groups
            results["gt"] = gt_groups
            if not show:
                return results

            X_pos = np.arange(len(text_groups)) * bars_width * 3

            arguments = {
                "groups_pos": X_pos,
                "bars_width": bars_width,
                "groups": [gt_groups, pred_groups],
                "groups_labels": ["GT Anomaly windows", "Anomaly windows predicted"],
                "groups_ticks": text_groups,
                "plot_title": "Anomaly duration distribution",
                "y_label": "Number of anomalies",
                "x_label": "Duration (min)",
                "ticks_rotation": 60,
                "tight_layout": True
            }

            bar_plot(**arguments)

    def _get_model_predictions(self, observations_scaled, proposals_scaled, threshold = None):
        """Gets the predictions of the model for the given threshold.
        
        Parameters
        ----------
        observations_scaled : DataFrame
            The observations, eventually scaled if necessary.
        
        proposals_scaled : DataFrame
            The proposals of the model, eventually scaled if necessary.
        
        threshold : float, default=None
            The threshold that identifies an anomaly.

        Returns
        -------
        predictions : ndarray
            The predictions of the model.
        """
        y_score = self._get_y_score(observations_scaled, proposals_scaled)
    
        if self._anomaly_evaluation.is_inverse_threshold():
            return np.where(y_score <= threshold, 1, 0)
        else:
            return np.where(y_score >= threshold, 1, 0)
