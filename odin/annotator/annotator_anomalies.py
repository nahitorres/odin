import os
import pandas as pd
import plotly.graph_objs as go
from IPython.core.display import display
from ipywidgets import Button, IntSlider, VBox, HBox, Layout, Label, RadioButtons, Text, HTML
from plotly.colors import DEFAULT_PLOTLY_COLORS

from odin.annotator import MetaPropertiesType
from odin.classes import TaskType
from odin.classes import strings as labels_str
from odin.classes.strings import err_type
from odin.annotator.annotator_interface import AnnotatorInterface
from odin.classes.timeseries import DatasetTSAnomalyDetection
from odin.utils.leaflet_zoom_utils import get_image_container_zoom, show_new_image

import json
import numpy as np

class AnomalyAnnotator(AnnotatorInterface):

    supported_types = [TaskType.TS_ANOMALY_DETECTION]#, TaskType.TS_PREDICTIVE_MAINTENANCE]
    
    __mandatory_params_no_dataset = {'task_type', 'anomaly_path', 'anomaly_types', 'ds_name', 'dataset'}
                              
    def __init__(self,
                 dataset=None,
                 task_type=None,
                 resample = False,
                 resampling_frequency = "1min",
                 nanToZero = True,
                 nan_replacement = -20,
                 anomaly_types = None,
                 properties_and_values = None,
                 ds_name=None,
                 anomaly_path=None,
                 metaproperties_path = None):
               
        if not os.path.exists(anomaly_path):
            raise TypeError("Please insert a valid path for the input anomalies file.")
                
        if dataset is None: # create new dataset
            #check mandatory parameters validity
            if task_type is None or anomaly_path is None or anomaly_types is None or ds_name is None:
                raise Exception(f"Invalid parameters. Please be sure to specify the following parameters: {self.__mandatory_params_no_dataset}")
                
            if not isinstance(dataset, DatasetTSAnomalyDetection):
                raise TypeError(err_type.format("dataset"))
                
                
            if not isinstance(task_type, TaskType):
                raise TypeError(err_type.format("task_type"))
                
            elif task_type not in self.supported_types:
                raise Exception(labels_str.warn_task_not_supported)

            if not isinstance(anomaly_types, list):
                raise TypeError(err_type.format("anomaly_types"))
                
            if properties_and_values is None:
                raise Exception("Please be sure to specify the properties and the values")
            elif not isinstance(properties_and_values, dict):
                raise TypeError(err_type.format("properties_and_values"))
            
        else:  # dataset already exists
            if type(dataset) is not DatasetTSAnomalyDetection:
                raise TypeError(f"Invalid dataset type: {type(dataset)}. Use DatasetTSAnomalyDetection.")
                

            dataset_obs = dataset.get_observations()
            self.objects = dataset_obs
            
            if resample:
                dataset_obs = dataset_obs.resample(resampling_frequency).mean()
            if nanToZero:
                dataset_obs = dataset_obs.fillna(nan_replacement)

        self._n_max_data = len(dataset_obs) - 1
        self._data_to_show = dataset_obs.copy()

        self._pos = 0
        
        self.anomaly_path = anomaly_path
        self.anomaly_types = anomaly_types
                

        self._n_data = 1000 if self._n_max_data > 1000 else self._n_max_data # how many data should be visualized? 1000 if there are enough
        self._n_data_update = round(self._n_data/2)

        if os.path.exists(anomaly_path):
            
            with open(anomaly_path, "r") as f:
                data = json.load(f)
                
            anomalies_dict = dict()
            anomalies_dict["start_date"] = []
            anomalies_dict["end_date"] = []
            
            anomalies_list = data["anomalies"]
            
            for el in anomalies_list:
                if isinstance(el, str):
                    anomalies_dict["start_date"].append(pd.to_datetime(el))
                    anomalies_dict["end_date"].append(pd.to_datetime(el))
                elif isinstance(el, list) and len(el) == 2:
                    anomalies_dict["start_date"].append(pd.to_datetime(el[0]))
                    anomalies_dict["end_date"].append(pd.to_datetime(el[1]))
                    
                    
            self.anomalies = pd.DataFrame.from_dict(anomalies_dict) 
            
        else:
            self.anomalies = pd.DataFrame(columns=["start_date", "end_date"])
            
            
        metaproperties_categories = list(properties_and_values.keys())
        
        # Read or write metaproperties csv file and save its content in self.metaproperties
        if os.path.exists(metaproperties_path):
            with open(metaproperties_path, "r") as f:
                data = pd.read_csv(f)
                
            self.metaproperties = data
            self.metaproperties = self.metaproperties.where(pd.notnull(self.metaproperties), None)
        else:
            self.metaproperties = pd.DataFrame(columns=["timestamp"] + metaproperties_categories)
            self.metaproperties.set_index("timestamp", inplace = True)
            
            with open(metaproperties_path, "w") as f:
                self.metaproperties.to_csv(f) 
                
        self.metaproperties_status = dict()
        self.metaproperties_path = metaproperties_path
        self.folder_anomaly_path = os.path.dirname(os.path.abspath(anomaly_path)) # folder provided as the one containing the anomalies file
        self.anomaly_path = os.path.join(self.folder_anomaly_path, ds_name.split(".")[0] + '.json')
        
        super().__init__(dataset, task_type, [], self.folder_anomaly_path, ds_name, properties_and_values, False,
                       False, None, None, None, None, False)
    
    
            
    
    def _create_widgets(self):
        self.__next_btn = Button(description='next', disabled=True)
        self.__next_btn.on_click(self._on_next_clicked)

        self.__previous_btn = Button(description='previous', disabled=True)
        self.__previous_btn.on_click(self._on_previous_clicked)

        self.__position_label = Label(value="Datapoint {}/{}".format(self._pos+self._n_data, self._n_max_data))

        self.__n_data_visualization = IntSlider(value=self._n_data,
                                                min=1,
                                                max=self._n_max_data,
                                                description='Data points',
                                                continuous_update=False)
        self.__n_data_visualization.observe(self._set_n_data_visualization, names='value')

        self.__n_data_update = IntSlider(value=self._n_data_update,
                                         min=2,
                                         max=self._n_data,
                                         description='Update steps',
                                         continuous_update=False)
        self.__n_data_update.observe(self._set_update_steps, names='value')

        self.__ann_label = Label("")
        self.__info_label = HTML("")
        self.__add_ann_btn = Button(description='add anomaly', disabled=True)
        self.__add_ann_btn.on_click(self._on_add_clicked)
        self.__remove_ann_btn = Button(description='remove anomaly', disabled=True)
        self.__remove_ann_btn.on_click(self._on_remove_clicked)
        self.__update_ann_btn = Button(description='update anomaly', disabled=True)
        self.__update_ann_btn.on_click(self._on_update_clicked)
        self.__export_to_csv_btn = Button(description='export', disabled=False)
        self.__export_to_csv_btn.on_click(self._on_export_clicked)
        
        layout = Layout(display='flex', justify_content='center')
  
        self.output_layout = VBox([HBox([self.__n_data_visualization, self.__n_data_update], layout=layout),
                        HBox([self.__previous_btn, self.__next_btn, self.__position_label], layout=layout),
                        HBox([self.__info_label], layout=layout),
                        HBox(self.output_layout, layout = layout),
                        HBox([self._fig], layout=layout),
                        HBox([self.__add_ann_btn, self.__remove_ann_btn, self.__update_ann_btn, self.__export_to_csv_btn], layout=layout),
                        HBox([self.__ann_label], layout=layout)])
                
        return self.output_layout
    
    
    def _create_check_radio_boxes(self):
        """
        Create the widgets for the annotation

        """
        
        labels = dict()
        
        if self.annotate_meta_properties:
            for k_name, v in self.properties_and_values.items():
                if len(v) == 3:
                    label = v[2]
                elif MetaPropertiesType.TEXT.value == v[0].value and len(v) == 2:
                    label = v[1]
                else:
                    label = k_name

                labels[k_name] = label

                if MetaPropertiesType.UNIQUE.value == v[0].value:  # radiobutton
                    self.radiobuttons[k_name] = RadioButtons(name=k_name, options=v[1],
                                                             disabled=False,
                                                             indent=False)
                elif MetaPropertiesType.COMPOUND.value == v[0].value:  # checkbox
                    self.checkboxes[k_name] = [Checkbox(False, indent=False, name=k_name,
                                                        description=prop_name) for prop_name in v[1]]
                elif MetaPropertiesType.CONTINUE.value == v[0].value:
                    self.bounded_text[k_name] = BoundedFloatText(value=v[1][0], min=v[1][0], max=v[1][1])

                elif MetaPropertiesType.TEXT.value == v[0].value:
                    self.box_text[k_name] = Textarea(disabled=False)
                    
        for key in self.radiobuttons.keys():
                self.radiobuttons[key].value = None
                    
        return labels
    
    
    
    def _set_check_radio_boxes_layout(self, labels):
        """
        Initialize the widget for the annotations
        Parameters
        ----------
        labels
            meta-properties names

        Returns
        -------
            output_layout
        """
        
        output_layout = []
        
        if self.annotate_meta_properties:
            for rb_k, rb_v in self.radiobuttons.items():
                if rb_k == "categories":
                    continue
                rb_v.layout.width = '180px'
                rb_v.observe(self._checkbox_changed)
                rb_v.add_class(rb_k)
                html_title = HTML(value="<b>" + labels[rb_k] + "</b>")
                self.check_radio_boxes_layout[rb_k] = VBox([rb_v])
                output_layout.append(VBox([html_title, self.check_radio_boxes_layout[rb_k]]))

            for cb_k, cb_i in self.checkboxes.items():
                if cb_k == "categories":
                    continue
                for cb in cb_i:
                    cb.layout.width = '180px'
                    cb.observe(self._checkbox_changed)
                    cb.add_class(cb_k)
                html_title = HTML(value="<b>" + labels[cb_k] + "</b>")
                self.check_radio_boxes_layout[cb_k] = VBox(children=[cb for cb in cb_i])
                output_layout.append(VBox([html_title, self.check_radio_boxes_layout[cb_k]]))

            for bf_k, bf in self.bounded_text.items():
                bf.layout.width = '80px'
                bf.layout.height = '35px'
                bf.observe(self._checkbox_changed)
                bf.add_class(bf_k)
                html_title = HTML(value="<b>" + labels[bf_k] + "</b>")
                self.check_radio_boxes_layout[bf_k] = VBox([bf])
                output_layout.append(VBox([html_title, self.check_radio_boxes_layout[bf_k]]))

            for tb_k, tb_i in self.box_text.items():
                tb_i.layout.width = '500px'
                tb_i.observe(self._checkbox_changed)
                tb_i.add_class(tb_k)
                html_title = HTML(value="<b>" + labels[tb_k] + "</b>")
                self.check_radio_boxes_layout[tb_k] = VBox([tb_i])
                output_layout.append(VBox([html_title, self.check_radio_boxes_layout[tb_k]]))
                
        return output_layout

    def __change_check_radio_boxes_value(self):
        """
        update the values of the widget based on the current annotation
        Parameters
        ----------
        current_ann: dict
            current annotation
        """
        
        if self.selected_ann_id is None:
            return
        current_ann = self.mapping["annotations"][self.selected_ann_id]
        
        print(current_ann)

       
        if self.annotate_meta_properties:
            for m_k, m_v in self.properties_and_values.items():
                if m_v[0].value == MetaPropertiesType.UNIQUE.value:  # radiobutton
                    self.radiobuttons[m_k].unobserve(self._checkbox_changed)
                    self.radiobuttons[m_k].value = current_ann[m_k] if m_k in current_ann.keys() else None
                    self.radiobuttons[m_k].disabled = False
                    self.radiobuttons[m_k].observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.COMPOUND.value:  # checkbox
                    for cb_i, cb_v in enumerate(self.checkboxes[m_k]):
                        cb_v.unobserve(self._checkbox_changed)
                        if m_k in current_ann.keys():
                            if cb_v.description in current_ann[m_k].keys():
                                cb_v.value = current_ann[m_k][cb_v.description]
                            else:
                                cb_v.value = False
                        else:
                            cb_v.value = False
                        cb_v.disabled = False
                        cb_v.observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.CONTINUE.value:  # textbound
                    self.bounded_text[m_k].unobserve(self._checkbox_changed)
                    self.bounded_text[m_k].value = float(current_ann[m_k]) if m_k in current_ann.keys() else \
                        self.bounded_text[m_k].min
                    self.bounded_text[m_k].disabled = False
                    self.bounded_text[m_k].observe(self._checkbox_changed)
                elif m_v[0].value == MetaPropertiesType.TEXT.value:  # text
                    self.box_text[m_k].unobserve(self._checkbox_changed)
                    self.box_text[m_k].value = current_ann[m_k] if m_k in current_ann.keys() else ""
                    self.box_text[m_k].disabled = False
                    self.box_text[m_k].observe(self._checkbox_changed)

    
    def _checkbox_changed(self, b):
               
        if b['name'] == '_dom_classes':
            property_name = b['new'][0]
            selected_name = b['owner'].value
            
            self.metaproperties_status[property_name] = selected_name
            
        elif b['name'] == 'label':
            property_name = b['owner']._dom_classes[0]
            selected_name = b['new']
            
            self.metaproperties_status[property_name] = selected_name
            
        return b
            

    def _on_next_clicked(self, btn):
        self._pos += self._n_data_update
        self._update_plot()

    def _on_previous_clicked(self, btn):
        self._pos -= self._n_data_update
        self._update_plot()

    def _set_update_steps(self, change):
        self._n_data_update = change['new']

    def _set_n_data_visualization(self, change):
        self._n_data = change['new']
        self.__n_data_update.max = self._n_data
        self._update_plot()

    def _on_add_clicked(self, btn):
        
        self.anomalies = pd.concat([self.anomalies, pd.DataFrame({"start_date": pd.to_datetime([self._start_time]), 
                                                                  "end_date": pd.to_datetime([self._end_time])})], ignore_index=True)
        self.anomalies.sort_values(by="start_date", inplace=True)
        self.anomalies.reset_index(inplace=True, drop=True)
        
        # Metaproperties
        status_timestamp = {'timestamp': self._start_time}
        status_timestamp = {**status_timestamp, **self.metaproperties_status} 
        self.metaproperties = self.metaproperties.append(status_timestamp, ignore_index = True)
                
        self._update_plot()
    
    def _on_remove_clicked(self, btn):
        self.anomalies = self.anomalies.drop(self.anomalies.loc[(self.anomalies["start_date"] == pd.to_datetime(self._start_time)) & (self.anomalies["end_date"] == pd.to_datetime(self._end_time))].index.values)
        
        print(self._start_time)
        self.metaproperties["timestamp"]
        
        self.metaproperties = self.metaproperties.drop(self.metaproperties.loc[(pd.to_datetime(self.metaproperties["timestamp"]) == pd.to_datetime(self._start_time))].index.values)
                                                                          
        self.anomalies.sort_values(by="start_date", inplace=True)
        self.anomalies.reset_index(inplace=True, drop=True)
        self.__ann_label.value = ""

        self._update_plot()
    
    def _on_update_clicked(self, btn):
        
        # Remove old status (i.e., old combination of meta-properties)
        self.metaproperties = self.metaproperties.drop(self.metaproperties.loc[(pd.to_datetime(self.metaproperties["timestamp"]) == pd.to_datetime(self._start_time))].index.values)
        
        # Add new combination of metaproperties
        # Metaproperties
        status_timestamp = {'timestamp': self._start_time}
        status_timestamp = {**status_timestamp, **self.metaproperties_status} 
        self.metaproperties = self.metaproperties.append(status_timestamp, ignore_index = True)
        
        self.anomalies.sort_values(by="start_date", inplace=True)
        self.anomalies.reset_index(inplace=True, drop=True)

        self._update_plot()

    def produce_length_based_output(self, start_date, end_date):
        if start_date == end_date: # single point
            return start_date.strftime('%Y-%m-%d %H:%M:%S')
        else: # multiple points
            return [start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S')]
        
        
    def _on_export_clicked(self, btn):
        
        # Export anomalies
        out_json = dict()
        self.anomalies["out"] = self.anomalies.apply(lambda x: self.produce_length_based_output(x.start_date, x.end_date), axis = 1)
        out_json["anomalies"] = [an[0] for an in self.anomalies[["out"]].to_numpy().tolist()]
        with open(self.anomaly_path, 'w') as outfile:
            json.dump(out_json, outfile)
            
        # Export manually created meta-annotations as .csv
        out_metaproperties = self.metaproperties.set_index("timestamp")
        out_metaproperties.to_csv(self.metaproperties_path)
        
        self.__ann_label.value = "Anomalies successfully exported at: '{}'. Total anomalies annotated: {}".format(self.anomaly_path, len(self.anomalies))
        
        for key in self.radiobuttons.keys():
                self.radiobuttons[key].value = None

    def _init_plot(self):
        self._fig = go.FigureWidget([go.Scatter(x=self._data_to_show.index[self._pos:self._pos+self._n_data], 
                                                y=self._data_to_show[self.feature_name].values[self._pos:self._pos+self._n_data],
                                                mode="lines+markers", 
                                                selected=dict(marker=dict(color=DEFAULT_PLOTLY_COLORS[1])))], 
                                    layout=dict(dragmode="select"))
        self._fig.update_layout(modebar={"remove": ['zoom', 'lasso']}, width=700, height=300, showlegend=False, margin=dict(l=0, r=20, t=20, b=20))
        self._scatter = self._fig.data[0]

        self.__new_annotation_drawn = False
    
        self._scatter.on_selection(self.handle_click)
    
    def handle_click(self, trace, points, state):
        
        self.metaproperties = self.metaproperties.where(pd.notnull(self.metaproperties), None)
        
        if len(points.point_inds) == 0:
            return
        trace.selectedpoints = None
        pts = trace.x[points.point_inds[0]:points.point_inds[-1]+1]
        tmp = self.anomalies.loc[(self.anomalies["start_date"].isin(pd.to_datetime(pts))) | (self.anomalies["end_date"].isin(pd.to_datetime(pts)) |
                                ((self.anomalies["start_date"] < pd.to_datetime(pts[0])) & (self.anomalies["end_date"] > pd.to_datetime(pts[-1]))))]
        if tmp.empty:
            self._start_time, self._end_time = pts[0], pts[-1]
            self.__add_ann_btn.disabled = False
            self.__remove_ann_btn.disabled = True
            self.__update_ann_btn.disabled = True
            
            for key in self.radiobuttons.keys():
                self.radiobuttons[key].value = None
                 
        else:
            self._start_time, self._end_time = pd.to_datetime(tmp["start_date"].values[0]), pd.to_datetime(tmp["end_date"].values[0])
            self.__remove_ann_btn.disabled = False
            self.__add_ann_btn.disabled = True
            self.__update_ann_btn.disabled = False
            
            for col_name in self.metaproperties.columns:
                if col_name != "timestamp" and col_name in self.radiobuttons.keys():
                    col_value = self.metaproperties.loc[(pd.to_datetime(self.metaproperties["timestamp"]) == pd.to_datetime(self._start_time))].iloc[0][col_name]
                    self.radiobuttons[col_name].value = col_value

            
        y_max = max(trace.y)+1
        y_min = min(trace.y)-1
        self.__ann_label.value = "Start time: {} | End time: {}".format(self._start_time, self._end_time)
        
        
        
        
        with self._fig.batch_update():
            if self.__new_annotation_drawn:
                self._fig.data = self._fig.data[:-1]
            self._fig.add_scatter(x=[self._start_time, self._start_time, self._end_time, self._end_time], y=[y_min, y_max, y_max, y_min], fillcolor="red", fill="toself", mode="lines+markers", opacity=0.75)
            self.__new_annotation_drawn = True
            
    
    def _update_plot(self):
        if self._pos <= 0:
            self._pos = 0
        elif self._pos + self._n_data > self._n_max_data:
            self._pos = self._n_max_data - self._n_data
        
        self.__previous_btn.disabled = (self._pos == 0)
        self.__next_btn.disabled = (self._pos >= self._n_max_data - self._n_data)

        self.__position_label.value = "Datapoint {}/{}".format(self._pos+self._n_data, self._n_max_data)
        self.__ann_label.value = ""
        
        self.__add_ann_btn.disabled = True
        self.__remove_ann_btn.disabled = True
        self.__update_ann_btn.disabled = True
        self.__new_annotation_drawn = False

        annotated_anomalies = self.anomalies.loc[(self.anomalies["start_date"].isin(self._data_to_show.index[self._pos:self._pos+self._n_data])) | 
                                                 (self.anomalies["end_date"].isin(self._data_to_show.index[self._pos:self._pos+self._n_data]) |
                                                 ((self.anomalies["start_date"] < self._data_to_show.index[self._pos]) & (self.anomalies["end_date"] > self._data_to_show.index[self._pos+self._n_data])))]

        with self._fig.batch_update():
            self._fig.data = self._fig.data[:1]
            self._scatter.x = self._data_to_show.index[self._pos:self._pos+self._n_data]
            self._scatter.y = self._data_to_show[self.feature_name].values[self._pos:self._pos+self._n_data]

            tmp_values = self._data_to_show[self.feature_name].values[self._pos:self._pos+self._n_data]
            y_max = max(tmp_values) + 1
            y_min = min(tmp_values) - 1
            for i, r in annotated_anomalies.iterrows():
                self._fig.add_scatter(x=[r["start_date"], r["start_date"], r["end_date"], r["end_date"]], y=[y_min, y_max, y_max, y_min], fillcolor="orange", fill="toself", mode="lines+markers", opacity=0.75, name="anomaly")
                
        for key in self.radiobuttons.keys():
                self.radiobuttons[key].value = None
    
    
    def start_annotation(self, feature_name):
        self.feature_name = feature_name
        self._init_plot()
        display(self._create_widgets())
        self._update_plot()

        
    def print_statistics(self):
        pass

    def print_results(self):
        pass

    def save_state(self):
        pass

    def _show_image(self):
        pass

    def _create_buttons(self):
        pass

    def _create_results_dict(self):
        return {}, {}

    def _set_display_function(self, custom_display_function):
        pass
    
    def _on_save_clicked(self, b):
        pass

    def _on_reset_clicked(self, b):
        pass

   
    def _perform_action(self):
        pass

    def _execute_validation(self):
        pass