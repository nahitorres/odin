import abc
import os.path
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

from odin.classes import TaskType
from odin.classes.strings import err_type, err_value
from odin.classes.timeseries.timeseries_type import TimeSeriesType
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_seasonality_trend
from odin.utils.lazy_dictionary import LazyDict


class DatasetTimeSeriesInterface(metaclass=abc.ABCMeta):

    def __init__(self,
                 dataset_path,
                 task_type,
                 ts_type,
                 proposals_paths,
                 properties_path,
                 csv_separator,
                 result_saving_path,
                 save_graphs_as_png):

        if not isinstance(dataset_path, str):
            raise TypeError(err_type.format('dataset_path'))

        if not isinstance(ts_type, TimeSeriesType):
            raise TypeError(err_type.format('ts_type'))

        if not isinstance(csv_separator, str):
            raise TypeError(err_type.format('csv_separator'))

        if not isinstance(result_saving_path, str):
            raise TypeError(err_type.format('result_saving_path'))

        if not isinstance(save_graphs_as_png, bool):
            raise TypeError(err_type.format('save_graphs_as_png'))

        self.dataset_path = dataset_path
        self.task_type = task_type
        self.ts_type = ts_type
        self.proposals_paths = proposals_paths
        self.properties_path = properties_path
        self.csv_separator = csv_separator
        self.result_saving_path = result_saving_path
        self.save_graphs_as_png = save_graphs_as_png

        self.observations = None
        self.proposals = None
        self.properties = None

        self.load()
        if properties_path is not None:
            self.load_properties()

    def load(self, force_loading=False):
        self._load_gt(force_loading)

        if self.proposals_paths is not None and (force_loading or self.proposals is None):
            tmp_dict = {}
            if self.task_type in [TaskType.TS_ANOMALY_DETECTION, TaskType.TS_PREDICTIVE_MAINTENANCE]:
                for model_name, path, proposals_type in self.proposals_paths:
                    tmp_dict[model_name] = (self._load_proposals, model_name, path, proposals_type)
            else:
                for model_name, path in self.proposals_paths:
                    tmp_dict[model_name] = (self._load_proposals, model_name, path)
            self.proposals = LazyDict(tmp_dict)

    def load_properties(self):
        if self.properties_path is None:
            get_root_logger().error('No properties file available')
            return -1
        elif not os.path.exists(self.properties_path):
            get_root_logger().error('Invalid properties file path: {}'.format(self.properties_path))
            return -1

        data = pd.read_csv(self.properties_path, sep=self.csv_separator)
        if self._index_gt not in data.columns:
            get_root_logger().error("'{}' column not present. Unable to index data.".format(self._index_gt))
            return -1

        if self.task_type != TaskType.TS_PREDICTIVE_MAINTENANCE:
            data[self._index_gt] = pd.to_datetime(data[self._index_gt])
        data = data.set_index(self._index_gt)
        data = data.sort_index()

        self.properties = data

    def get_available_properties(self):
        if self.properties is None:
            return []
        return self.properties.columns.tolist()

    def get_values_for_property(self, p_name):
        if p_name not in self.get_available_properties():
            get_root_logger().error(err_value.format("p_name", self.get_available_properties()))
            return -1
        return self.properties[p_name].unique().tolist()

    def get_observations_for_property_value(self, p_name, p_value):
        if self.properties is None:
            get_root_logger().error('No properties available')
            return -1
        if p_name not in self.get_available_properties():
            get_root_logger().error(err_value.format("p_name", self.get_available_properties()))
            return -1

        return self.observations.loc[self.observations.index.get_level_values(self._index_gt).isin(self.properties.loc[self.properties[p_name] == p_value].index)].copy()

    def analyze_seasonality_trend(self, column=None, model_type='additive', period=None):
        available_columns = self.observations.columns.difference(['anomaly', 'anomaly_window'])

        if column is None:
            c_name = available_columns[0]
        else:
            if column not in available_columns:
                get_root_logger().error(err_value.format('column', available_columns))
                return -1
            c_name = column

        decomposition = seasonal_decompose(self.observations[c_name], model=model_type, period=period)

        plot_seasonality_trend([decomposition.observed,
                                decomposition.trend,
                                decomposition.seasonal,
                                decomposition.resid],
                               "Seasonal-Trend analysis", self.save_graphs_as_png, self.result_saving_path)

    def analyze_stationarity(self, column=None):
        available_columns = self.observations.columns.difference(['anomaly', 'anomaly_window'])

        if column is None:
            c_name = available_columns[0]
        else:
            if column not in available_columns:
                get_root_logger().error(err_value.format('column', available_columns))
                return -1
            c_name = column

        adf, pvalue, _, _, critical_values, _ = adfuller(self.observations[c_name])

        is_stationarity = "STATIONARY" if pvalue <= 0.05 else "NOT STATIONARY"

        data = [[adf,
                pvalue,
                critical_values,
                is_stationarity]]

        print(tabulate(data, headers=['ADF', 'p-value', 'Critical values', 'Result']))

    def get_observations(self):
        return self.observations.copy()

    def get_proposals(self, model_name):
        if model_name not in self.proposals.keys():
            get_root_logger().error(err_value.format('model_name', list(self.proposals.keys())))
            return -1
        if self.task_type in [TaskType.TS_ANOMALY_DETECTION, TaskType.TS_PREDICTIVE_MAINTENANCE]:
            return self.proposals[model_name][0].copy()
        else:
            return self.proposals[model_name].copy()

    @abc.abstractmethod
    def _load_gt(self, force_loading=False):
        pass

    @abc.abstractmethod
    def _load_proposals(self, model_name, path, proposals_type=None):
        pass
