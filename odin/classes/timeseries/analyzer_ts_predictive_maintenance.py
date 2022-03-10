from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from numbers import Number
from statistics import mean

from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from odin.classes import Metrics, Curves
from odin.classes.strings import err_type, err_value
from odin.classes.timeseries import AnalyzerTimeSeriesInterface, DatasetTSPredictiveMaintenance, TSProposalsType
from odin.classes.timeseries.metrics import precision_recall_curve_values
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_gain_chart, plot_lift_chart, plot_predicted_vs_actual, plot_regression_residuals, \
    plot_reliability_diagram, plot_threshold_analysis, plot_multiple_curves, plot_RUL_trend


class AnalyzerTSPredictiveMaintenance(AnalyzerTimeSeriesInterface):

    __available_metrics_label = [Metrics.ACCURACY, Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE,
                                 Metrics.MISS_ALARM_RATE, Metrics.FALSE_ALARM_RATE]

    __available_metrics_regression = [Metrics.MAE, Metrics.MSE, Metrics.RMSE, Metrics.MAPE]

    __available_metrics = []

    def __init__(self,
                 model_name,
                 dataset,
                 metric=Metrics.F1_SCORE,
                 threshold=0.5,
                 result_saving_path='./result',
                 save_graphs_as_png=False):

        if not isinstance(model_name, str):
            raise TypeError(err_type.format('model_name'))
        elif model_name not in dataset.proposals.keys():
            raise ValueError(err_value.format('model_name', list(dataset.proposals.keys())))

        if not isinstance(dataset, DatasetTSPredictiveMaintenance):
            raise TypeError(err_type.format('dataset'))

        if not isinstance(threshold, Number):
            raise TypeError(err_type.format('threshold'))

        self._threshold = threshold

        self.proposals_type = dataset.get_proposals_type(model_name)

        self.__available_metrics = self.__available_metrics_label if self.proposals_type == TSProposalsType.LABEL else self.__available_metrics_regression

        if not isinstance(metric, Metrics):
            raise TypeError(err_type.format('metric'))
        elif metric not in self.__available_metrics:
            raise ValueError(err_value.format('metric', self.__available_metrics))

        super().__init__(model_name, dataset, self.__available_metrics, metric, result_saving_path, save_graphs_as_png)

        if not self.dataset._analysis_available:
            raise Exception('No predictive maintenance labels available')

    def analyze_performance_for_unit(self, metrics=None, threshold=None, show=True):
        if metrics is None:
            metrics = self._available_metrics
        elif not isinstance(metrics, list):
            get_root_logger().error(err_type.format('metrics'))
            return -1
        elif not all(m in self._available_metrics for m in metrics):
            get_root_logger().error(err_value.format('metrics', self._available_metrics))
            return -1

        if threshold is None:
            threshold = self._threshold
        elif not isinstance(threshold, Number):
            get_root_logger().error(err_type.format('threshold'))
            return -1

        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        observations = observations.loc[observations.index.get_level_values(self.dataset._index_gt).isin(proposals.index.get_level_values(self.dataset._index_proposals))]

        results = {}
        all_data = []
        for unit_id in observations.index.get_level_values('unit_id').unique():
            unit_obs = observations.loc[observations.index.get_level_values('unit_id') == unit_id]
            unit_props = proposals.loc[proposals.index.get_level_values('unit_id') == unit_id]
            if self.proposals_type == TSProposalsType.LABEL:
                y_true = unit_obs['label'].values
                y_score = unit_props['confidence'].values
            else:
                y_true = unit_obs['RUL'].values
                y_score = unit_props['RUL'].values
            values = [self._compute_metric(unit_obs, unit_props, y_true, y_score, metric, threshold=threshold) for metric in metrics]
            all_data.append([unit_id]+values)
            results[unit_id] = values

        if not show:
            return results

        print(tabulate(all_data, headers=['ID']+[metric.value for metric in metrics]))

    def analyze_performance_for_threshold(self, metrics=None, average='micro', show=True):
        if metrics is None:
            metrics = self._available_metrics
        elif not isinstance(metrics, list):
            get_root_logger().error(err_type.format('metrics'))
            return -1
        elif not all(m in self._available_metrics for m in metrics):
            get_root_logger().error(err_value.format('metrics', self._available_metrics))
            return -1

        thresholds = np.arange(0, 1.001, 0.05).round(5)
        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        observations = observations.loc[observations.index.get_level_values(self.dataset._index_gt).isin(
            proposals.index.get_level_values(self.dataset._index_proposals))]

        if average == 'micro':
            results = self._analyze_performance_for_threshold_micro(observations, proposals, metrics, thresholds)
        else:
            results = self._analyze_performance_for_threshold_macro(observations, proposals, metrics, thresholds)

        if not show:
            return results

        plot_threshold_analysis({'x': thresholds,
                                 'y': results},
                                "Threshold analysis",
                                self.save_graphs_as_png,
                                self.result_saving_path)

    def _analyze_performance_for_threshold_micro(self, observations, proposals, metrics, thresholds):
        match = pd.merge(proposals, observations, left_on=self.dataset._index_proposals, right_on=self.dataset._index_gt)
        y_true = match['label'].values
        y_score = match['confidence'].values

        results = defaultdict(list)
        for t in thresholds:
            for m in metrics:
                results[m].append(self._compute_metric(observations, proposals, y_true, y_score, m, threshold=t))
        return results

    def _analyze_performance_for_threshold_macro(self, observations, proposals, metrics, thresholds):
        results = defaultdict(list)
        for t in thresholds:
            res = np.array(list(self.analyze_performance_for_unit(metrics, threshold=t, show=False).values()))
            for i, m in enumerate(metrics):
                results[m].append(mean(res[:, i]))
        return results

    def precision_recall_curve(self):
        if self.proposals_type != TSProposalsType.LABEL:
            get_root_logger().error('Not supported for task type: {}'.format(self.proposals_type))
            return -1

        observations = self.dataset.get_observations()[['label']]
        proposals = self.dataset.get_proposals(self.model_name)[['confidence']]
        match = pd.merge(proposals, observations, left_on=self.dataset._index_proposals,
                         right_on=self.dataset._index_gt)

        y_true = match['label'].values
        y_score = match['confidence'].values
        precision, recall = precision_recall_curve_values(y_true, y_score)

        plot_multiple_curves({'overall': {'auc': auc(recall, precision),
                                          'x': recall,
                                          'y': precision}},
                             Curves.PRECISION_RECALL_CURVE,
                             {'overall': {'display_name': self.model_name}},
                             self.save_graphs_as_png,
                             self.result_saving_path)

    def analyze_gain_lift(self, show=True):
        if self.proposals_type != TSProposalsType.LABEL:
            get_root_logger().error('Not supported for task type: {}'.format(self.proposals_type))
            return -1

        observations = self.dataset.get_observations()[['label']]
        proposals = self.dataset.get_proposals(self.model_name)[['confidence']]
        match = pd.merge(proposals, observations, left_on=self.dataset._index_proposals, right_on=self.dataset._index_gt)
        match = match.sort_values(by='confidence', ascending=False)
        labels = match['label'].values

        step = round(len(match.index)/10)
        n_total = len(match.loc[match['label'] == 1].index)
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

    def show_predicted_vs_actual(self):
        if self.proposals_type != TSProposalsType.REGRESSION:
            get_root_logger().error('Not supported for task type: {}'.format(self.proposals_type))
            return -1

        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        observations = observations.loc[observations.index.get_level_values(self.dataset._index_gt).isin(
                proposals.index.get_level_values(self.dataset._index_proposals))]

        results = {}
        match = pd.merge(proposals, observations, left_on=self.dataset._index_proposals,
                         right_on=self.dataset._index_gt)
        residuals = match['RUL_x'].values - match['RUL_y'].values
        scaler = StandardScaler()
        scaler.fit(residuals.reshape(-1, 1))

        for unit_id in observations.index.get_level_values('unit_id').unique():
            unit_obs = observations.loc[observations.index.get_level_values('unit_id') == unit_id][['RUL']]
            unit_props = proposals.loc[proposals.index.get_level_values('unit_id') == unit_id][['RUL']]
            match = pd.merge(unit_props, unit_obs, left_on=self.dataset._index_proposals,
                             right_on=self.dataset._index_gt)
            residuals = match['RUL_x'].values - match['RUL_y'].values
            residuals = scaler.transform(residuals.reshape(-1, 1)).reshape(-1)
            results[unit_id] = {'predicted': match['RUL_x'],
                                'actual': match['RUL_y'],
                                'residuals': residuals}

        plot_predicted_vs_actual(results,
                                 self.save_graphs_as_png,
                                 self.result_saving_path,
                                 self.model_name)

        plot_regression_residuals(results,
                                 self.save_graphs_as_png,
                                 self.result_saving_path,
                                  self.model_name)

    def show_predicted_vs_actual_for_unit(self):
        if self.proposals_type != TSProposalsType.REGRESSION:
            get_root_logger().error('Not supported for proposals type: {}'.format(self.proposals_type))
            return -1

        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        observations = observations.loc[observations.index.get_level_values(self.dataset._index_gt).isin(
                proposals.index.get_level_values(self.dataset._index_proposals))]

        for unit_id in observations.index.get_level_values('unit_id').unique():
            unit_obs = observations.loc[observations.index.get_level_values('unit_id') == unit_id]
            unit_props = proposals.loc[proposals.index.get_level_values('unit_id') == unit_id]
            unit_obs = unit_obs[['RUL']]
            unit_props = unit_props[['RUL']]
            match = pd.merge(unit_props, unit_obs, left_on=self.dataset._index_proposals, right_on=self.dataset._index_gt)
            residuals = match['RUL_x'].values - match['RUL_y'].values
            scaler = StandardScaler()
            residuals = scaler.fit_transform(residuals.reshape(-1, 1)).reshape(-1)

            result = {unit_id: {'predicted': match['RUL_x'],
                                'actual': match['RUL_y'],
                                'residuals': residuals}}

            plot_predicted_vs_actual(result,
                                     self.save_graphs_as_png,
                                     self.result_saving_path,
                                     id_name=unit_id)
            plot_regression_residuals(result,
                                      self.save_graphs_as_png,
                                      self.result_saving_path,
                                      id_name=unit_id)

    def analyze_reliability(self, num_bins=10, min_threshold=0):
        if self.proposals_type != TSProposalsType.LABEL:
            get_root_logger().error('Not supported for proposals type: '.format(self.proposals_type))
            return -1

        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        observations = observations.loc[observations.index.get_level_values(self.dataset._index_gt).isin(proposals.index.get_level_values(self.dataset._index_proposals))]

        y_true = observations['label'].values
        y_pred = np.ones(len(y_true))
        y_score = proposals['confidence'].values

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

    def analyze_reliability_for_unit(self, num_bins=10, min_threshold=0):
        if self.proposals_type != TSProposalsType.LABEL:
            get_root_logger().error('Not supported for proposals type: '.format(self.proposals_type))
            return -1

        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        observations = observations.loc[observations.index.get_level_values(self.dataset._index_gt).isin(
                proposals.index.get_level_values(self.dataset._index_proposals))]

        for unit_id in observations.index.get_level_values('unit_id').unique():
            unit_obs = observations.loc[observations.index.get_level_values('unit_id') == unit_id]
            unit_props = proposals.loc[proposals.index.get_level_values('unit_id') == unit_id]
            y_true = unit_obs['label'].values
            y_pred = np.ones(len(y_true))
            y_score = unit_props['confidence'].values

            if min_threshold > 0:
                ix = np.where(y_score >= min_threshold)[0]
                y_true = y_true[ix]
                y_pred = y_pred[ix]
                y_score = y_score[ix]

            result = self._calculate_reliability(y_true, y_pred, y_score, num_bins)

            plot_reliability_diagram(result,
                                     self.save_graphs_as_png,
                                     self.result_saving_path,
                                     is_classification=True,
                                     category=str(unit_id))

    def analyze_predicted_RUL_trend(self):
        observations = self.dataset.get_observations()
        proposals = self.dataset.get_proposals(self.model_name)
        observations = observations.loc[observations.index.get_level_values(self.dataset._index_gt).isin(proposals.index.get_level_values(self.dataset._index_proposals))]

        difference = []
        optimal = []
        for unit_id in observations.index.get_level_values('unit_id').unique():
            unit_obs = observations.loc[observations.index.get_level_values('unit_id') == unit_id].copy()
            unit_props = proposals.loc[proposals.index.get_level_values('unit_id') == unit_id].copy()
            unit_obs['diff'] = unit_obs['RUL'].diff()

            new_cycles_indexes = np.where(unit_obs['diff'].values > 0)[0]
            if not new_cycles_indexes:
                difference.extend(unit_props['RUL'].diff().dropna().values)
                optimal.extend(unit_obs['diff'].dropna().values)
            else:
                for start, end in zip([0]+new_cycles_indexes, new_cycles_indexes + [len(unit_obs.index)]):
                    difference.extend(unit_props[start:end]['RUL'].diff().fillna(0).values)

        plot_RUL_trend(difference,
                       self.save_graphs_as_png,
                       self.result_saving_path)

    def _analyze_performance(self, observations, proposals, metrics, show=True):
        results = np.array(list(self.analyze_performance_for_unit(metrics, show=False).values()))

        r_macro = [mean(results[:, i]) for i in range(0, len(metrics))]
        data_macro = ['macro'] + r_macro

        # observations = self.dataset.get_observations()
        # proposals = self.dataset.get_proposals(self.model_name)
        observations = observations.loc[observations.index.get_level_values(self.dataset._index_gt).isin(
                proposals.index.get_level_values(self.dataset._index_proposals))].copy()

        match = pd.merge(proposals, observations, left_on=self.dataset._index_proposals, right_on=self.dataset._index_gt)
        if self.proposals_type == TSProposalsType.LABEL:
            y_true = match['label'].values
            y_score = match['confidence'].values
        else:
            y_true = match['RUL_y'].values
            y_score = match['RUL_x'].values

        r_micro = [self._compute_metric(observations, proposals, y_true, y_score, metric, threshold=self._threshold) for metric in metrics]
        data_micro = ['micro'] + r_micro

        if not show:
            return {'macro': r_macro,
                    'micro': r_micro}

        print(tabulate([data_macro, data_micro], headers=[''] + [metric.value for metric in metrics]))

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

