import json
from numbers import Number

import numpy as np
import pandas as pd
import os

from tabulate import tabulate

from odin.classes import TaskType
from odin.classes.strings import err_type, err_value
from odin.classes.timeseries.dataset_ts_interface import DatasetTimeSeriesInterface
from odin.classes.timeseries.timeseries_type import TSProposalsType
from odin.utils import get_root_logger


class DatasetTSAnomalyDetection(DatasetTimeSeriesInterface):

    def __init__(self,
                 dataset_path,
                 ts_type,
                 anomalies_path,
                 proposals_paths=None,
                 nab_data_percentage=0.1,
                 properties_path=None,
                 csv_separator=',',
                 index_gt='Time',
                 index_proposals='Time',
                 result_saving_path='./result',
                 save_graphs_as_png=False):

        if not isinstance(anomalies_path, str):
            raise TypeError(err_type.format('proposals_type'))

        if not isinstance(nab_data_percentage, Number):
            raise TypeError(err_type.format('nab_data_percentage'))
        elif nab_data_percentage <=0 or nab_data_percentage >=1:
            raise ValueError(err_value('nab_data_percentage', '0 < nab_data_percentage < 1'))

        if not isinstance(index_gt, str):
            raise TypeError(err_type.format('index_gt'))

        if not isinstance(index_proposals, str):
            raise TypeError(err_type.format('index_proposals'))

        self._anomalies_path = anomalies_path
        self._index_gt = index_gt
        self._index_proposals = index_proposals

        self._analysis_available = False

        self._start_end_anomalies = []

        self.nab_config = {'data_percentage': nab_data_percentage}

        super().__init__(dataset_path, TaskType.TS_ANOMALY_DETECTION, ts_type, proposals_paths, properties_path,
                         csv_separator, result_saving_path, save_graphs_as_png)

    def _add_anomalies(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        self.observations['anomaly'] = 0
        self._start_end_anomalies = []

        f = open(path, 'r')
        f_j = json.load(f)
        f.close()
        if 'anomalies' not in f_j.keys():
            get_root_logger().error("'anomalies' key not present in anomaly file")
            return -1

        single_points = []

        for anomaly in f_j['anomalies']:
            if isinstance(anomaly, list):
                self.observations.loc[(self.observations.index >= pd.to_datetime(anomaly[0])) &
                                      (self.observations.index <= pd.to_datetime(anomaly[1])), 'anomaly'] = 1

                self._start_end_anomalies.append([anomaly[0], anomaly[1]])
            else:
                single_points.append(pd.to_datetime(anomaly))
                self._start_end_anomalies.append([anomaly, anomaly])

        self.observations.loc[self.observations.index.isin(single_points), 'anomaly'] = 1

        self.observations['anomaly'] = self.observations['anomaly']
        self._analysis_available = True

        self._set_anomaly_windows()

    def _set_anomaly_windows(self):
        self.nab_config['data_length'] = len(self.get_observations().index)
        self.nab_config['n_anomalies'] = self.get_aggregate_anomalies()

        w_length = self.nab_config['data_length'] * self.nab_config['data_percentage'] / self.nab_config['n_anomalies']
        # For each anomaly, get start-time and end-time
        anomalies_start_end = self.get_start_end_anomalies()

        self.nab_config['windows_index'] = []
        self.observations['anomaly_window'] = 0

        index_values = self.get_observations().index

        for start, end in anomalies_start_end:
            s_i = np.where(index_values == pd.to_datetime(start))[0]
            if not s_i:
                continue
            start_w = s_i[0] - round(w_length / 2)

            e_i = np.where(index_values == pd.to_datetime(end))[0]
            if not e_i:
                continue
            end_w = e_i[0] + round(w_length / 2)

            if start_w < 0:
                start_w = 0
            if end_w >= self.nab_config['data_length']:
                end_w = self.nab_config['data_length'] - 1

            self.nab_config['windows_index'].append([index_values[start_w], index_values[end_w]])
            self.observations.loc[(self.observations.index >= index_values[start_w]) &
                                  (self.observations.index <= index_values[end_w]), 'anomaly_window'] = 1

    def get_start_end_anomalies(self):
        return self._start_end_anomalies

    def get_proposals_type(self, model_name):
        if model_name not in self.proposals.keys():
            get_root_logger().error(err_value.format('model_name', list(self.proposals.keys())))
            return -1
        return self.proposals[model_name][1]

    def get_aggregate_anomalies(self):
        if 'anomaly' not in self.observations.columns:
            get_root_logger().error('No anomaly data available')
            return -1

        anomalies = self.observations['anomaly'].values - self.observations['anomaly'].shift(1).values
        return len(np.where(anomalies == 1)[0])

    def get_anomaly_percentage(self, aggregate=False):
        if 'anomaly' not in self.observations.columns:
            get_root_logger().error('No anomaly data available')
            return -1
        total_data = len(self.observations.index)
        total_anomalies = len(self.observations.loc[self.observations['anomaly'] == 1].index) if not aggregate else self.get_aggregate_anomalies()
        anomalies_percentage = (total_anomalies / total_data) * 100
        data = [[total_data,
                total_anomalies,
                f"{anomalies_percentage:.2f}%"]]
        print(tabulate(data, headers=['Total data', 'Anomaly data', 'Anomalies%'], numalign='right', stralign='right'))

    # def extract_anomalies_type(self):
    #     if not self._analysis_available:
    #         get_root_logger().error('No anomaly labels available')
    #         return -1

    #     if self.ts_type == TimeSeriesType.MULTIVARIATE:
    #         get_root_logger().error('Not available for multivariate time series')
    #         return -1

    #     observations = self.get_observations()

    #     gt_values = observations[observations.columns.difference(['anomaly'])].values
    #     mean = np.mean(gt_values)
    #     std = np.std(gt_values)
    #     # TODO

    def _load_gt(self, force_loading=False):
        if not force_loading and self.observations is not None:
            return

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError('Invalid data set path:', self.dataset_path)

        print('Loading data set...')

        data = pd.read_csv(self.dataset_path, sep=self.csv_separator)

        if self._index_gt not in data.columns:
            raise ValueError("'{}' column not present. Unable to index data.".format(self._index_gt))

        data[self._index_gt] = pd.to_datetime(data[self._index_gt])
        data = data.set_index(self._index_gt)
        data = data.sort_index()

        if 'anomaly' in data.columns:
            if not all(a_l in [0, 1] for a_l in data['anomaly'].unique().tolist()):
                raise ValueError("Invalid 'anomaly' values. Possible values: [0, 1]")
            self._analysis_available = True

        self.observations = data

        self._add_anomalies(self._anomalies_path)

        print('Done!')

    def _load_proposals(self, model_name, path, proposals_type):
        print("Loading proposals of {} model...".format(model_name))

        if not os.path.exists(path):
            raise FileNotFoundError('Invalid proposals path for {} model: {}'.format(model_name, path))

        data = pd.read_csv(path, sep=self.csv_separator)

        if self._index_proposals not in data.columns:
            raise ValueError("'{}' column not present. Unable to index data for {} model".format(self._index_proposals, model_name))

        data[self._index_proposals] = pd.to_datetime(data[self._index_proposals])
        data = data.set_index(self._index_proposals)
        data = data.sort_index()

        if proposals_type == TSProposalsType.LABEL:
            if 'confidence' not in data.columns:
                raise ValueError("'confidence' column not present for {} model".format(model_name))
        else:
            missing_features = list((set(self.observations.columns) - set(data.columns)) - {'anomaly', 'anomaly_window'})
            if missing_features:
                raise ValueError("Features {} missing for {} model".format(missing_features, model_name))

        if self.observations.index.difference(data.index).tolist():
            raise ValueError("Difference between 'Time' index of {} model and 'Time' index of data set".format(model_name))

        print('Done!')

        return data, proposals_type


