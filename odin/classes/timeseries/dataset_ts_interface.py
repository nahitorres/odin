import abc
import json
import math
import os.path
from copy import copy
from numbers import Number

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

from odin.classes.timeseries import ScalerInterface
from odin.classes import TaskType
from odin.classes.strings import err_type, err_value, \
    err_ts_properties_not_loaded, err_property_not_loaded, \
    err_ts_properties_json, info_loading_properties, info_done, \
    err_ts_property_file_format_invalid
from odin.classes.timeseries.timeseries_type import TimeSeriesType, \
    TSProposalsType
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_seasonality_trend
from odin.utils.lazy_dictionary import LazyDict
from odin.utils import periodicity_utils, properties_utils


class DatasetTimeSeriesInterface(metaclass=abc.ABCMeta):

    def __init__(self,
                 dataset_path,
                 task_type,
                 ts_type,
                 proposals_paths,
                 properties_path,
                 csv_separator,
                 result_saving_path,
                 save_graphs_as_png,
                 scaler=None,
                 properties_json = "properties.json",
                 common_properties = None):

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

        # extend paths that do not specify csv separator to be a tuple of 4 elements
        for i in range(len(proposals_paths)):
            # if csv separator is not specified, we give it the standard value
            if len(proposals_paths[i]) == 3:
                proposals_paths[i] = (proposals_paths[i][0],
                                      proposals_paths[i][1],
                                      proposals_paths[i][2],
                                      ',')
        
        # Check that it is actually a scaler
        if scaler is not None and not isinstance(scaler, ScalerInterface):
            raise TypeError(err_type.format('scaler_values'))
        

        self.dataset_path = dataset_path
        self.task_type = task_type
        self.ts_type = ts_type
        self.proposals_paths = proposals_paths
        self.properties_path = properties_path
        self.properties_json = properties_json
        self.common_properties = {} if common_properties is None else common_properties
        self.csv_separator = csv_separator
        self.result_saving_path = result_saving_path
        self.save_graphs_as_png = save_graphs_as_png
        self.scaler = scaler

        self.observations = None
        self.scaled_observations = None
        self.proposals = None
        self.scaled_proposals = dict()
        self.properties = None

        self._analyses_with_properties_available = False
        self._analyses_without_properties_available = False

        self._properties_json = {}

        self.OBSERVATIONS_COLUMN = None

        self.load()
        if properties_path is not None:
            self.load_properties()

    def load(self, force_loading=False):
        self._load_gt(force_loading)

        if self.proposals_paths is not None and (force_loading or self.proposals is None):
            tmp_dict = {}
            if self.task_type in [TaskType.TS_ANOMALY_DETECTION, TaskType.TS_PREDICTIVE_MAINTENANCE]:
                for model_name, path, proposals_type, csv_sep in self.proposals_paths:
                    tmp_dict[model_name] = (self._load_proposals, model_name, path, proposals_type, csv_sep)
            else:
                for model_name, path in self.proposals_paths:
                    tmp_dict[model_name] = (self._load_proposals, model_name, path)
            self.proposals = LazyDict(tmp_dict)
            self._analyses_without_properties_available = True

    def load_properties(self):
        if self.properties_path is None:
            raise ValueError('No properties file available')
        elif not os.path.exists(self.properties_path):
            raise ValueError('Invalid properties file path: {}'.format(self.properties_path))

        data = pd.read_csv(self.properties_path, sep=self.csv_separator)
        if self._index_gt not in data.columns:
            raise ValueError("'{}' column not present. Unable to index data.".format(self._index_gt))

        if self.task_type != TaskType.TS_PREDICTIVE_MAINTENANCE:
            data[self._index_gt] = pd.to_datetime(data[self._index_gt])
        data = data.set_index(self._index_gt)
        data = data.sort_index()

        self.properties = data

    def _is_properties_file_valid(self, properties_dataset, common_properties, check_properties=True):
        """Check if the values in the existing 'properties.json' file (default filename) are valid

        Parameters
        ----------
        properties_dataset: pandas.DatFrame
            DataFrame containing the properties as columns
            
        common_properties: set
            Common properties to exclude

        Returns
        -------
        bool
            True if the file is valid, otherwise False
        """
        try:
            file = open(self.properties_json, "r")
            file_data = json.load(file)
            file.close()

            if not check_properties:
                return True

            file_properties = file_data["properties"]
            props = set(properties_dataset.columns)
            all_properties = props.difference(common_properties)

            if len(file_properties) == 0 and len(all_properties) > 0:
                return False

            # check first only properties validity
            for p in file_properties.keys():
                if p not in all_properties:
                    return False

            # load all possible properties values
            properties_values = {}

            for p in all_properties:
                properties_values[p] = list(set(properties_dataset[p].values.tolist()))

            for property in file_properties.keys():
                for value in file_properties[property]['values']:
                    if value['value'] not in properties_values[property]:
                        if isinstance(value["value"], float) and math.isnan(value["value"]):
                            if any(math.isnan(x) if isinstance(x, float) else False for x in properties_values[property]):
                                continue
                            
                        if value['value'] == "no value" and any(math.isnan(x) if isinstance(x, float) else False for x in properties_values[property]):
                            continue
                        
                        return False

            return True
        except Exception as e:
            return False
        
    def are_analyses_without_properties_available(self):
        return self._analyses_without_properties_available

    def are_analyses_with_properties_available(self):
        return self._analyses_with_properties_available
    
    def are_valid_properties(self, properties):
        """ Check if the properties are valid.
        
        Parameters
        ----------
        properties: list
            Properties names to be checked

        Returns
        -------
        bool
            True if the properties names are valid, otherwise False
        """
        if not isinstance(properties, list):
            get_root_logger().error(err_type.format("properties"))
            return False
        elif len(properties) == 0:
            get_root_logger().error(f"Empty properties list.")
            return False

        for p in properties:
            if p not in self.get_property_keys():
                if self.is_possible_property(p):
                    get_root_logger().error(err_property_not_loaded.format(p))
                    return False
                get_root_logger().error(err_value.format("property", list(self.get_property_keys())))
                return False

        return True

    def is_possible_property(self, property_name):
        """Indicates whether the property could be a valid property for the evaluation.

        Parameters
        ----------
        property_name: str
            Name of the property to be verified.

        Returns
        -------
            bool
        """
        if not isinstance(property_name, str):
            get_root_logger().error(err_type.format("property_name"))
            return False
        elif self.properties is None:
            get_root_logger().error(err_ts_properties_not_loaded)
        
        return property_name in set(self.properties.columns).difference([self._index_gt])
    
    def is_valid_property(self, property_name, possible_values):
        """Checks if the properties and the corresponding values are valid.
        
        Parameters
        ----------
        property_name: str
            Name of the property
        possible_values: list
            List of the possible values
        Returns
        -------
        bool
            True if the property and the corresponding values are valid, otherwise False
        """
        if not isinstance(property_name, str):
            get_root_logger().error(err_type.format("property_name"))
            return False

        if not isinstance(possible_values, list):
            get_root_logger().error(err_type.format("possible_values"))
            return False

        if property_name in self.get_property_keys():
            if len(possible_values) == 0:
                get_root_logger().error(f"Empty possible values list")
                return False
            for value in possible_values:
                if value not in self.get_values_for_property(property_name):
                    get_root_logger().error(f"Property value '{value}' not valid for property '{property_name}'")
                    return False
        else:
            get_root_logger().error("Property '{}' not valid. Possible values: {}".format(property_name, list(self.get_property_keys())))
            return False

        return True

    def get_display_name_of_property(self, pkey):
        """Returns the name of the property to display

        Parameters
        ----------
        pkey: str
            Name of the property

        Returns
        -------
        str or None
            Name of the property to display or None if not found
        """
        if not isinstance(pkey, str):
            get_root_logger().error(err_type.format("pkey"))
            return -1

        if pkey not in self._properties_json.keys():
            return None

        return self._properties_json[pkey]["display_name"]
    
    def get_display_name_of_property_value(self, property, property_value):
        """Returns the name to display of a specific property value

        Parameters
        ----------
        property: str
            Name of the property containing the value
        property_value: str or Number
            Name of the property value

        Returns
        -------
        str or None
            Name to display of the property value or None if not found
        """
        if not isinstance(property, str):
            get_root_logger().error(err_type.format("property"))
            return -1
        if not isinstance(property_value, str) and not isinstance(property_value, Number):
            get_root_logger().error(err_type.format("property_value"))
            return -1

        if property in self._properties_json.keys():
            values = self._properties_json[property]["values"]
            for value in values:
                if value["value"] == property_value:
                    return value["display_name"]
        return None

    def get_property_keys(self):
        """Returns the names of the properties

        Returns
        -------
        dict_keys
        """
        return self.properties.keys()

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

        if isinstance(p_value, float) and math.isnan(p_value):
            property_indexer = self.properties.isna().values
        else:
            property_indexer = self.properties[p_name] == p_value

        return self.observations.loc[self.observations.index.get_level_values(self._index_gt).isin(self.properties.loc[property_indexer].index)].copy()

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

    def get_observations(self, scaled=False):
        """Get the observations.
        
        Parameters
        ----------
        scaled : bool
            States if the observations must be scaled using the scaler.

        Returns
        -------
        observations : DataFrame
            The DataFrame of the observations, eventually scaled.
        """

        if scaled and self.scaler is not None:
            if self.scaled_observations is None:
                self.scaled_observations = self.observations.copy()
                scaled_obs = self.scaled_observations[self.OBSERVATIONS_COLUMN].values
                scaled_obs = self.scaler.transform(scaled_obs)
                self.scaled_observations[self.OBSERVATIONS_COLUMN] = scaled_obs
            return self.scaled_observations.copy()
        return self.observations.copy()

    def get_proposals(self, model_name, scaled=False):
        """Get the proposals.
        
        Parameters
        ----------
        model_name : str
            The name of the model.
        
        scaled : bool
            Specified if the proposals must be obtained scaled.

        Returns
        -------
        proposals : DataFrame
            The DataFrame of the proposals, eventually scaled.
        """
        # if the model does not exist an error is triggered
        if model_name not in self.proposals.keys():
            get_root_logger().error(err_value.format('model_name', list(self.proposals.keys())))
            return -1
        
        if scaled and self.scaler is not None:
            if model_name in self.scaled_proposals.keys():
                return self.scaled_proposals[model_name]

            if self.task_type in [TaskType.TS_ANOMALY_DETECTION, TaskType.TS_PREDICTIVE_MAINTENANCE]:
                self.scaled_proposals[model_name] = self.proposals[model_name][0].copy()
            else:
                self.scaled_proposals[model_name] = self.proposals[model_name].copy()
                
            scaled_props = np.array(self.scaled_proposals[model_name][self.OBSERVATIONS_COLUMN].tolist())
            scaled_props = self.scaler.transform(scaled_props)
            
            self.scaled_proposals[model_name][self.OBSERVATIONS_COLUMN] = scaled_props.tolist()
            return self.scaled_proposals[model_name].copy()

        if self.task_type in [TaskType.TS_ANOMALY_DETECTION, TaskType.TS_PREDICTIVE_MAINTENANCE]:
            return self.proposals[model_name][0].copy()
        else:
            return self.proposals[model_name].copy()

        # if isinstance(proposals_lists[0, 0], str):
            # props_values = np.array(list(map(lambda x: np.array(ast.literal_eval(x[0])), proposals_lists)), dtype=object)
            # window_size = max(list(map(lambda x: x.shape[0], props_values)))
    # 
            # if props_values.ndim == 1:
            #     for i in range(window_size - 1):
            #         inf_to_add = [-np.inf] * (window_size - 1 - i)
            # 
            #         # extend the head
            #         extended_head = copy(inf_to_add)
            #         extended_head.extend(props_values[i].tolist())
            #         props_values[i] = np.array(extended_head)
            # 
            #         # extend the tail
            #         extended_tail = props_values[
            #             props_values.shape[0] - 1 - i].tolist()
            #         extended_tail.extend(copy(inf_to_add))
            #         props_values[props_values.shape[0] - 1 - i] = np.array(
            #             extended_tail)
        # 
            #     props_values = np.vstack(props_values)
            #     
            # return props_values
            
        # return proposals_lists

    # TODO to remove
    def _get_proposal_column(self, model_name):
        """Gets which is the column for the proposal for the model.
        
        Parameters
        ----------
        model_name : str
            The name of the model.

        Returns
        -------
        proposal_column : str
            The name of the column in which the proposals are stored
        """
        if self.task_type in [TaskType.TS_ANOMALY_DETECTION, TaskType.TS_PREDICTIVE_MAINTENANCE]:
            if self.proposals[model_name][1] == TSProposalsType.REGRESSION:
                return "value"
            else:
                return "confidence"
        else:
            return "value"

    @abc.abstractmethod
    def _load_gt(self, force_loading=False):
        pass

    @abc.abstractmethod
    def _load_proposals(self, model_name, path, proposals_type=None, csv_sep=','):
        pass
    
    
    def analyze_periodicity(self, column=None):
        '''
        Determine if the time series is periodic or not using the FFT and compute the number of standard deviations of the main period power peak with respect to the power spectrum mean.
        
        Args:
            column: name of the column containing the time series
        '''
        
        if column is not None:
            if column is not isinstance(column, str):
                raise TypeError(err_type.format('column'))
            elif not (column in self.observations.columns):
                raise Exception("The selected column is not available")
                        
        available_columns = self.observations.columns.difference(['anomaly', 'anomaly_window'])

        if column is None:
            c_name = available_columns[0]
        else:
            if column not in available_columns:
                get_root_logger().error(err_value.format('column', available_columns))
                return -1
            c_name = column

        period, number_sigma = periodicity_utils.get_fft_periodicity(self.observations[c_name])

        is_periodic = "PERIODIC" if number_sigma >= 3 else "NOT PERIODIC"

        data = [[period,
                number_sigma,
                is_periodic]]

        print(tabulate(data, headers=['Period', 'Number of standard deviations', 'Result']))
        
        return period, number_sigma, is_periodic

        
    def plot_fft(self, column = None):
        '''
        Interface to plot the FFT
        
        Args:
            column: name of the column containing the time series
        '''
        available_columns = self.observations.columns.difference(['anomaly', 'anomaly_window'])

        if column is None:
            c_name = available_columns[0]
        else:
            if column not in available_columns:
                get_root_logger().error(err_value.format('column', available_columns))
                return -1
            c_name = column

        periodicity_utils.plot_fft(self.observations[c_name])
        
        
    def extract_metaproperties(self):
        '''
        Interface to extract basic meta-properties (hour of the day, day of the week, day of the month, month)
        The dataset must be indexed by timestamp.
        It presents the distribution of anomalies for different temporal properties in a tabular form.
        '''
        hours_distribution, day_week_distribution, day_month_distribution, month_distribution = properties_utils.get_temporal_metaproperties_distribution(self.observations)    
        
        print(tabulate(hours_distribution.items(), headers = ["Hour of the day", "Number of anomalies"]))
        print("\n")
        print(tabulate(day_week_distribution.items(), headers = ["Day of the week", "Number of anomalies"]))
        print("\n")
        print(tabulate(day_month_distribution.items(), headers = ["Day of the month", "Number of anomalies"]))
        print("\n")
        print(tabulate(month_distribution.items(), headers = ["Month", "Number of anomalies"]))
        
        
    def analyze_metaproperties(self):
        '''
        Interface to show temporal meta-properties (hour of the day, day of the week, day of the month, month) using histograms.
        The dataset must be indexed by timestamp.
        '''
        hours_distribution, day_week_distribution, day_month_distribution, month_distribution = properties_utils.get_temporal_metaproperties_distribution(self.observations)    
        
        properties_utils.plot_distribution(hours_distribution, "Hour of the day distribution")
        properties_utils.plot_distribution(day_week_distribution, "Day of the week distribution")
        properties_utils.plot_distribution(day_month_distribution, "Day of the month distribution")
        properties_utils.plot_distribution(month_distribution, "Month distribution")

        

        