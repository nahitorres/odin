import json
import math
from numbers import Number

import numpy as np
import pandas as pd
import os
import ast

from IPython.display import display
from ipywidgets import Checkbox, VBox, Button, HBox

from odin.utils.draw_utils import pie_plot, bar_plot
from tabulate import tabulate

from odin.classes import TaskType
from odin.classes.strings import err_type, err_value, err_ts_property_invalid, \
    err_ts_property_values_invalid, info_loading_properties, info_done, \
    err_ts_property_file_format_invalid, err_ts_properties_json
from odin.classes.timeseries.dataset_ts_interface import DatasetTimeSeriesInterface
from odin.classes.timeseries.timeseries_type import TSProposalsType
from odin.utils import get_root_logger
from odin.utils.env import is_notebook


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
                 save_graphs_as_png=False,
                 scaler=None,
                 properties_json = "properties.json"):

        if not isinstance(anomalies_path, str):
            raise TypeError(err_type.format('anomalies_path'))
            
        if not os.path.exists(anomalies_path):
            raise TypeError(err_type.format('anomalies_path'))

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
                         csv_separator, result_saving_path, save_graphs_as_png, scaler=scaler, properties_json=properties_json)
        

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

    def load_properties(self):
        super().load_properties()
        
        if is_notebook():
            self._load_or_create_properties_notebook(self.properties, self.common_properties)
        else:
            self._load_or_create_properties(self.properties, self.common_properties)
            self._set_analyses_with_properties_available()

    def _load_or_create_properties_notebook(self, properties_dataset, common_properties):
        """Method to create and/or load the 'properties.json' (default name) file if ODIN is executed from CLI

        Parameters
        ----------
        properties_dataset: pandas.DataFrame
            DataFrame containing the properties. For DatasetLocalization it
            represents the annotations, for DatasetClassification it represents
            the observations.
        
        common_properties: set
            Represents the common properties to not consider.
        """

        def on_load_selected_clicked(b):
            properties = set([p.description for p in checkboxes if p.value])
           
            if not properties:
                return
            self._create_properties(properties, properties_dataset)
            self.load_properties_display_names()
            for b in buttons:
                b.disabled = True
            for b in checkboxes:
                b.disabled = True
            self._set_analyses_with_properties_available()

        def on_load_all_clicked(b):
            on_select_all(None)
            on_load_selected_clicked(None)

        def on_select_all(b):
            for ckb in checkboxes:
                ckb.value = True

        def on_deselect_all(b):
            for ckb in checkboxes:
                ckb.value = False

        props = set(properties_dataset.columns)
        self._possible_properties = list(props.difference(common_properties))

        if (not os.path.exists(self.properties_json)) or (not self._is_properties_file_valid(properties_dataset, common_properties)):
            all_properties = props.difference(common_properties)
            if len(all_properties) == 0:
                self._create_properties({}, properties_dataset)
                self.load_properties_display_names()
            else:
                checkboxes = [Checkbox(False, description=p) for p in all_properties]
                ui_boxes = VBox(checkboxes)

                select_all_button = Button(description="select all")
                deselect_all_button = Button(description="deselect all")
                select_all_button.on_click(on_select_all)
                deselect_all_button.on_click(on_deselect_all)
                ui_selection_buttons = HBox([select_all_button, deselect_all_button])

                load_selected_button = Button(description='load')
                load_all_button = Button(description='load all')
                load_selected_button.on_click(on_load_selected_clicked)
                load_all_button.on_click(on_load_all_clicked)
                ui_buttons = HBox([load_all_button, load_selected_button])

                buttons = [select_all_button, deselect_all_button, load_selected_button, load_all_button]

                print("Select at least one property to load:")
                display(ui_selection_buttons, ui_boxes, ui_buttons)
        else:
            self.load_properties_display_names(check_validity=False)

    def _load_or_create_properties(self, properties_dataset, common_properties):
        """Method to create and/or load the 'properties.json' (default name) file if ODIN is executed from CLI

        Parameters
        ----------
        properties_dataset: pandas.DataFrame
            DataFrame containing the properties. For DatasetLocalization it
            represents the annotations, for DatasetClassification it represents
            the observations.
        
        common_properties: set
            Represents the common properties to not consider.
        """
        props = set(properties_dataset.columns)
        self._possible_properties = list(props.difference(common_properties))

        if (not os.path.exists(self.properties_json)) or (not self._is_properties_file_valid(properties_dataset, common_properties)):
            all_properties = props.difference(common_properties)
            properties = self.__select_properties_terminal(all_properties)
            self._create_properties(properties, properties_dataset)
            self.load_properties_display_names()
        else:
            self.load_properties_display_names(check_validity=False)

    def _create_properties(self, properties, properties_dataset):
        """Method to create the 'properties.json' (default name) file

        Parameters
        ----------
        properties: set
            Properties to include
        
        properties_dataset: pandas.Dataframe
            All possible properties
        """
        print("Creating properties file...")
        
        if properties_dataset is not None:
            properties_dataset = properties_dataset.reset_index()
            
        p_values = {}
        values = {}
        for p in properties:
            p_values[p] = list(set(properties_dataset[p].values.tolist()))
            
            found_nan = False
            for el in p_values[p]:
                if isinstance(el, float) and math.isnan(el):
                    found_nan = True
                    break
                    
            if found_nan:
                p_values[p] = [el for el in p_values[p].copy() if not math.isnan(el)]
                p_values[p].append(math.nan)
            
            values_array = []
            
            for value in p_values[p]:
                display_name = self.get_display_name_of_property_value(p, value)
                if display_name is None:
                    display_name = value
                values_array.append({"value": value, "display_name": display_name})
            
            display_name = self.get_display_name_of_property(p)
            if display_name is None:
                display_name = p
            values[p] = {"values": values_array, "display_name": display_name}

        file = open(self.properties_json, "w")
        json.dump({"properties": values}, file, indent=4)
        file.close()

        print("Done!")

    def __select_properties_terminal(self, all_properties):
        """Method that allows the user to select the properties to load

        Parameters
        ----------
        all_properties: set
            All available properties names

        Returns
        -------
        set
            Properties selected
        """
        if len(all_properties) == 0:
            return {}
        
        all_properties_np = np.array(list(all_properties))
        text_selection = ""
        
        for i, p in enumerate(all_properties_np):
            text_selection = text_selection + f"[{i + 1}] - {p}\n"
            
        indexes_selectable = np.array(range(1, all_properties_np.size + 1))
        
        while True:
            user_input = input(f"Found {all_properties_np.size} properties:\n{text_selection}"
                               f"Select properties to load.\nDefault {indexes_selectable}: ")
            
            try:
                properties_selected = np.unique(list(map(int, user_input.split())))
            except:
                print("\nPlease select the properties from the list below.")
                continue
            
            indexes_selected = []
            if properties_selected.size == 0:
                indexes_selected = np.array(indexes_selectable) - 1
            else:
                try:
                    for p_index in properties_selected:
                        if p_index in indexes_selectable:
                            indexes_selected.append(p_index - 1)
                        else:
                            raise IndexError()
                except IndexError:
                    print("\nPlease select the properties from the list below.")
                    continue

            user_input = input(f"Are you sure you want to load these properties: \n"
                               f"{all_properties_np[indexes_selected]}? [Y/n]: ")
            if not user_input or user_input.lower() == 'y':
                properties = set(all_properties_np[indexes_selected])
                return properties

    def load_properties_display_names(self, check_validity=True):
        """Loads into memory the properties and their values.

        The properties are retrieved from the properties file (default is 'properties.json')

        Parameters
        ----------
        check_validity : bool, default=True
            Checks if the json file is valid.
        """
        try:
            print(info_loading_properties)
        
            file = open(self.properties_json, "r")
            properties = json.load(file)["properties"]
            file.close()
        
            is_valid = True
            if check_validity:
                is_valid = self._are_properties_from_file_valid(properties)
        
            if is_valid:
                self._properties_json = properties
            
                print(info_done)
            else:
                get_root_logger().error(err_ts_property_file_format_invalid)
        except FileNotFoundError as e:
            get_root_logger().error(err_ts_properties_json.format(self.properties_json))

    def _are_properties_from_file_valid(self, properties_json):
        for p in properties_json:
            if p not in self.properties.columns:
                return False
            if "display_name" not in properties_json[p] or "values" not in properties_json[p]:
                return False
            for value in properties_json[p]["values"]:
                if "value" not in value or "display_name" not in value:
                    return False
        return True

    def _set_analyses_with_properties_available(self):
        self._analyses_with_properties_available = True
        self._analyses_without_properties_available = True

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

    def get_resampled_dataset(self, rule, fill_method=None):
        """
        Resample the ground truth dataset

        Parameters
        ----------
        rule: str
            Resample rule to be applied (See pandas documentation for all the rules: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects")
        fill_method: str, optional
            Method used to fill NaN values after the resampling (default is None)
        
        Returns
        -------
        pd.DataFrame
            Resampled dataset
        
        """
        if not isinstance(rule, str):
            get_root_logger().error(err_type.format('rule'))
            return -1
        if fill_method is not None and not isinstance(fill_method, str):
            get_root_logger().error(err_type.format('fill_method'))
            return -1

        gt = self.get_observations()
        gt = gt.drop(["anomaly", "anomaly_window"], axis=1).copy()
        try:
            data = gt.resample(rule)
        except:
            get_root_logger().error(err_value.format('rule', "See pandas documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects"))
            return -1 

        if fill_method is None:
            return data.asfreq().copy()
        if fill_method == "sum":
            return data.sum().copy()
        if fill_method == "mean":
            return data.mean().copy()
        if fill_method == "pad":
            return data.pad().copy()
        if fill_method == "bfill":
            return data.bfill().copy()
        
        get_root_logger().error(err_value.format('fill_method', "[None, sum, mean, pad, bfill]"))
        return -1
        

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

        # TODO multivariate
        self.OBSERVATIONS_COLUMN = list(set(self.observations.columns) - {'anomaly', 'anomaly_window'})[0]

        self._add_anomalies(self._anomalies_path)

        print('Done!')

    def _load_proposals(self, model_name, path, proposals_type=None, csv_sep=','):
        print("Loading proposals of {} model...".format(model_name))

        if not os.path.exists(path):
            raise FileNotFoundError('Invalid proposals path for {} model: {}'.format(model_name, path))

        data = pd.read_csv(path, sep=csv_sep)

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

        if proposals_type == TSProposalsType.REGRESSION:
            # parse lists stored as strings if there are any before finishing
            props = data.copy()[self.OBSERVATIONS_COLUMN].values
            props = self._parse_proposal_lists(props)
            data[self.OBSERVATIONS_COLUMN] = props.copy()
        
        print('Done!')

        return data, proposals_type
    
    
    def _parse_proposal_lists(self, proposals_lists: np.ndarray):
        """Parse the proposals lists and translate them into lists.
        
        Parameters
        ----------
        proposals_lists : ndarray
            The extracted ndarray of the proposals lists.

        Returns
        -------
        parsed_proposals : ndarray
            The parsed proposals lists.
        """
        # Detect if proposals are lists, in case convert them to numpy array

        props_values = []
        for v in proposals_lists:
            if type(v) == str:
                props_values.append(ast.literal_eval(v))
            else:
                props_values.append(v)

        return props_values

    def show_distribution_of_property(self, property_name, property_values = None, plot_type = "pie", bars_width = 0.8, show=True):
        """Shows the distribution of the property in the dataset.
        
        Parameters
        ----------
        property_name
            The name of the property to be analyzed for distribution.
            
        property_values : list, default=None
            The list of the property's values to be included in the analysis. If
            None, all values are included.
        
        plot_type : str, default="pie"
            The type of plot to be realized.
            
        bars_width : float, default=0.8
            The width of the bars in case of bar plot.
            
        show : bool, default=True
            Whether to show or not the plot. If False, results are returned as
            a dict.

        Returns
        -------
        None
        """
        if not self._analyses_with_properties_available:
            if not self._analyses_without_properties_available:
                get_root_logger().error("Please complete the properties selection first")
                return -1
            else:
                get_root_logger().error("No properties available. Please make sure to load the properties")
                return -1
        
        if not isinstance(property_name, str):
            get_root_logger().error(err_type.format("property_name"))
            return -1
        elif not self.are_valid_properties([property_name]):
            return -1
        
        if property_values is not None and not isinstance(property_values, list):
            get_root_logger().error(err_type.format("property_values"))
            return -1
        elif property_values is not None and not self.is_valid_property(property_name, property_values):
            return -1
        
        if plot_type not in ["pie", "bar"]:
            get_root_logger().error(err_value.format("plot_type", ["pie", "bar"]))
            return -1
        
        if property_values is None:
            property_values = self.get_values_for_property(property_name)

        property_types = np.unique(property_values)

        property_name_to_show = self.get_display_name_of_property(property_name)
        property_value_names_to_show = [self.get_display_name_of_property_value(property_name, v) for v in property_types]

        results = {"property": {}}
        sizes = np.zeros(property_types.shape[0])
        for i, type_ in enumerate(property_types):
            sizes[i] = self.get_observations_for_property_value(property_name, type_).shape[0]
            results["property"][type_] = sizes[i]
        sizes /= np.sum(sizes)
        sizes *= 100

        if not show:
            return results

        if plot_type == "pie":
            pie_plot(sizes, property_types,
                     "{} distribution".format(property_name_to_show),
                     "",
                     False)
        elif plot_type == "bar":
            arguments = {
                "groups_pos": np.arange(property_types.shape[0]),
                "bars_width": bars_width,
                "groups": [sizes.tolist()],
                "groups_ticks": property_value_names_to_show,
                "plot_title": "{} distribution".format(property_name_to_show),
                "y_label": "Percentage",
                "x_label": property_name_to_show,
                "ticks_rotation": 60,
                "tight_layout": True
            }
    
            bar_plot(**arguments)
