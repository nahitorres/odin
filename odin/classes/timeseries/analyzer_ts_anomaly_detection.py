from collections import defaultdict

import numpy as np
from numbers import Number
import pandas as pd
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

from odin.classes import Metrics, Curves
from odin.classes.strings import err_type, err_value
from odin.classes.timeseries import DatasetTSAnomalyDetection, TSProposalsType
from odin.classes.timeseries.analyzer_ts_interface import AnalyzerTimeSeriesInterface
from odin.classes.timeseries.metrics import get_confusion_matrix, precision_recall_curve_values
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_threshold_analysis, display_confusion_matrix, pie_plot, plot_distribution, \
    plot_reliability_diagram, plot_gain_chart, plot_lift_chart, plot_multiple_curves


class AnalyzerTSAnomalyDetection(AnalyzerTimeSeriesInterface):

    __available_metrics = [Metrics.ACCURACY, Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE,
                           Metrics.NAB_SCORE, Metrics.MISS_ALARM_RATE, Metrics.FALSE_ALARM_RATE]

    __available_evaluations = [Metrics.MAE, Metrics.MSE]

    def __init__(self,
                 model_name,
                 dataset,
                 metric=Metrics.F1_SCORE,
                 threshold=0.5,
                 anomaly_evaluation=Metrics.MAE,
                 scaler_values=None,
                 result_saving_path='./result',
                 save_graphs_as_png=False
                 ):

        if not isinstance(dataset, DatasetTSAnomalyDetection):
            raise TypeError(err_type.format('dataset'))

        if not isinstance(metric, Metrics):
            raise TypeError(err_type.format('metric'))
        elif metric not in self.__available_metrics:
            raise ValueError(err_value.format('metric', self.__available_metrics))

        if not isinstance(threshold, Number):
            raise TypeError(err_type.format('threshold'))

        if not isinstance(anomaly_evaluation, Metrics):
            raise TypeError(err_type.format('anomaly_evaluation'))
        elif anomaly_evaluation not in self.__available_evaluations:
            raise ValueError(err_value.format('anomaly_evaluation', self.__available_evaluations))

        if scaler_values is not None and (not isinstance(scaler_values, dict) or 'mean' not in scaler_values.keys() or 'std' not in scaler_values.keys()):
            raise TypeError(err_type.format('scaler_values'))

        self._threshold = threshold
        self._anomaly_evaluation = anomaly_evaluation
        self._scaler_values = scaler_values

        super().__init__(model_name, dataset, self.__available_metrics, metric, result_saving_path, save_graphs_as_png)

        if not self.dataset._analysis_available:
            raise Exception('No anomaly labels available')

        self.proposals_type = self.dataset.get_proposals_type(self.model_name)

        self.nab_params = {'windows_index': self.dataset.nab_config['windows_index'],
                           'A_tp': 1,
                           'A_fp': 0.1,
                           'A_fn': 1}

        if self._scaler_values is None:
            observations = self.dataset.get_observations()
            gt_values = observations[observations.columns.difference(['anomaly', 'anomaly_window'])].values

            self._scaler_values = {'mean': np.mean(gt_values),
                                   'std': np.std(gt_values)}

    def analyze_performance_for_threshold(self, metrics=None):
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

        y_true = observations['anomaly'].values
        y_score = self._get_y_score(observations, proposals)

        max_value, step_value = (1.001, 0.05) if self.proposals_type == TSProposalsType.LABEL else (max(y_score), max(y_score)/21)
        thresholds = np.arange(0, max_value, step_value).round(5)

        results = defaultdict(list)
        for threshold in thresholds:
            for metric in metrics:
                results[metric].append(self._compute_metric(observations, proposals, y_true, y_score, metric,
                                                            threshold, self.nab_params))

        plot_threshold_analysis({'x': thresholds,
                                 'y': results},
                                "Threshold analysis",
                                self.save_graphs_as_png,
                                self.result_saving_path)

    def show_confusion_matrix(self):
        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)

        y_true = observations['anomaly'].values
        y_score = self._get_y_score(observations, proposals)
        cm = get_confusion_matrix(y_true, y_score, self._threshold)
        display_confusion_matrix(np.array([cm]),
                                 ['Anomaly'],
                                 None,  # properties filter
                                 self.save_graphs_as_png,
                                 self.result_saving_path)

    def analyze_false_positive_errors(self, distance=0, threshold=None, bins=8, show=True):
        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        y_true = observations['anomaly'].values
        y_score = self._get_y_score(observations, proposals)
        if threshold is None:
            threshold = self._threshold
        y_pred = np.where(y_score >= threshold, 1, 0)

        matching = pd.DataFrame(data={'y_true': y_true,
                                      'y_pred': y_pred},
                                index=observations.index)
        matching['eval'] = 0
        matching.loc[matching['y_true'] == 1, 'eval'] = 1
        matching.loc[(matching['y_true'] == 0) & (matching['y_pred'] == 1), 'eval'] = -1

        generic, affected, continuous = 0, 0, 0

        anomalies_pos = np.where(matching['eval'] == 1)[0]

        previous_anomaly_pos = -1
        anomaly_pos_index = 0
        next_anomaly_pos = anomalies_pos[0]
        is_previous_anomaly = False

        distances = {'affected': [],
                     'continuous': [],
                     'generic': []}

        index_values = matching.index
        errors_index = {'affected': [],
                        'continuous': [],
                        'generic': []}

        for i, v in enumerate(matching['eval'].values):
            if (i > next_anomaly_pos) and (next_anomaly_pos != -1):
                previous_anomaly_pos = next_anomaly_pos
                anomaly_pos_index += 1
                next_anomaly_pos = anomalies_pos[anomaly_pos_index] if anomaly_pos_index < len(anomalies_pos) else -1
            if v == -1:
                previous_d = i - previous_anomaly_pos if previous_anomaly_pos != -1 else float('inf')
                next_d = i - next_anomaly_pos if next_anomaly_pos != -1 else float('inf')

                d = previous_d if previous_d < np.abs(next_d) else next_d
                # AFFECTED errors
                if np.abs(d) <= distance:
                    affected += 1
                    distances['affected'].append(d)
                    errors_index['affected'].append(index_values[i])
                # CONTINUOUS errors
                elif is_previous_anomaly:
                    continuous += 1
                    distances['continuous'].append(d)
                    errors_index['continuous'].append(index_values[i])
                # GENERIC errors
                else:
                    generic += 1
                    distances['generic'].append(d)
                    errors_index['generic'].append(index_values[i])

                is_previous_anomaly = True

            else:
                is_previous_anomaly = False

        if not show:
            return {'generic': generic, 'affected': affected, 'continuous': continuous}, distances, errors_index

        pie_plot([generic, affected, continuous],
                 ['generic', 'affected', 'continuous'],
                 'False Positive errors',
                 self.result_saving_path,
                 self.save_graphs_as_png,
                 colors=['royalblue', 'darkorange', 'forestgreen'])

        plot_distribution(distances,
                          bins,
                          'Steps from nearest anomaly',
                          'Error distance distribution',
                          self.save_graphs_as_png,
                          self.result_saving_path)

    def analyze_reliability(self, num_bins=10, min_threshold=0):
        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)

        y_true = observations['anomaly'].values
        y_pred = np.ones(len(y_true))
        y_score = self._get_y_score(observations, proposals)

        if self.proposals_type == TSProposalsType.REGRESSION:
            scaler = MinMaxScaler()
            y_score = scaler.fit_transform(y_score.reshape(-1, 1))
            y_score = y_score.reshape(-1)

        if min_threshold > 0:
            ix = np.where(y_score >= min_threshold)[0]
            y_true = y_true[ix]
            y_pred = y_pred[ix]
            y_score = y_score[ix]

        result = self._calculate_reliability(y_true, y_pred, y_score, num_bins)

        plot_reliability_diagram(result,
                                 self.save_graphs_as_png,
                                 self.result_saving_path,
                                 is_classification=True)

    def analyze_gain_lift(self, show=True):
        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)

        match = pd.DataFrame({'anomaly': observations['anomaly'].values,
                              'confidence': self._get_y_score(observations, proposals)},
                             index=observations.index)
        match = match.sort_values(by='confidence', ascending=False)
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

    def precision_recall_curve(self):
        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        y_true = observations['anomaly'].values
        if self.proposals_type == TSProposalsType.LABEL:
            y_score = proposals['confidence'].values
        else:
            y_score = self._get_predictions_errors(observations, proposals)
        precision, recall = precision_recall_curve_values(y_true, y_score)
        plot_multiple_curves({'overall': {'auc': auc(recall, precision),
                                          'x': recall,
                                          'y': precision}},
                             Curves.PRECISION_RECALL_CURVE,
                             {'overall': {'display_name': self.model_name}},
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

        avg_acc = self._compute_metric(None, None, y_true, y_score, Metrics.ACCURACY, self._threshold)

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
        y_true = observations['anomaly'].values
        if self.proposals_type == TSProposalsType.LABEL:
            y_score = proposals['confidence'].values
        else:
            y_score = self._get_predictions_errors(observations, proposals)

        results = [self._compute_metric(observations, proposals, y_true, y_score, metric, self._threshold, self.nab_params) for metric in metrics]
        if not show:
            return {'micro': results}

        data = [results]

        print(tabulate(data, headers=[metric.value for metric in metrics]))

    def _get_y_score(self, observations, proposals):
        if self.proposals_type == TSProposalsType.LABEL:
            return proposals['confidence'].values
        return self._get_predictions_errors(observations, proposals)

    def _get_predictions_errors(self, observations, proposals):
        gt_values = observations[observations.columns.difference(['anomaly', 'anomaly_window'])].values
        props_values = proposals[proposals.columns.difference(['confidence'])].values

        m = self._scaler_values['mean']
        s = self._scaler_values['std']

        gt_values = self._standardize_data(gt_values, m, s)
        props_values = self._standardize_data(props_values, m, s)

        if self._anomaly_evaluation == Metrics.MAE:
            errors = np.sum(np.abs(gt_values - props_values), axis=1)
        else:
            errors = np.sum(np.abs(gt_values - props_values)**2, axis=1)

        return errors

    def _standardize_data(self, data, m, s):
        np_data = np.array(data)
        np_data = (np_data - m)/s
        return np_data
