import pandas as pd
from IPython.core.display import display, clear_output
from ipywidgets import Button, Dropdown, IntSlider, FloatSlider, Checkbox, Output, VBox, HBox, Layout
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from odin.classes.timeseries import TSProposalsType
from odin.classes.timeseries.visualizer_ts_interface import VisualizerTimeSeriesInterface


class VisualizerTSAnomalyDetection(VisualizerTimeSeriesInterface):

    def __init__(self, dataset, analyzers=None):

        analyzers_dict = {}
        if analyzers is not None:
            for a in analyzers:
                analyzers_dict[a.model_name] = a

        super().__init__(dataset, analyzers_dict)
        self._n_max_data = len(self.dataset.get_observations().index) - 1

        self._data = self.dataset.get_observations()

        self._data_to_show = self._data.copy()

        self._pos = 0
        self._data_aggregation = 'default'
        self._feature = None
        self._n_data = self._n_max_data
        self._n_data_update = round(self._n_max_data/2)
        self._show_anomalies = True
        self._show_anomaly_windows = False
        self._show_proposals = self._init_proposals()

        self.all_widgets = self._create_widgets()

    def _init_proposals(self):
        colors = ['purple', 'olive', 'grey', 'brown', 'pink']
        proposals = {}
        for i, a in enumerate(self.analyzers.keys()):
            y_score = self.analyzers[a]._get_y_score(self._data, self.dataset.get_proposals(a))
            p = pd.DataFrame(data={'y_score': y_score},
                             index=self._data.index)
            threshold = self.analyzers[a]._threshold
            proposals[a] = {'y_score': p,
                            'threshold': threshold,
                            'max_value': max(y_score),
                            'show': False,
                            'show_error': False,
                            'error_distance': 20,
                            'color': colors[i % len(colors)]}
        return proposals

    def _create_widgets(self):
        self.__next_btn = Button(description='next', disabled=True)
        self.__next_btn.on_click(self._on_next_clicked)

        self.__previous_btn = Button(description='previous', disabled=True)
        self.__previous_btn.on_click(self._on_previous_clicked)

        self.__data_aggregation = Dropdown(options=[('Default', 'default'),
                                                    ('Second', 'S'),
                                                    ('Minute', 'T'),
                                                    ('Hour', 'H'),
                                                    ('Day', 'D'),
                                                    ('Week', 'W'),
                                                    ('Month', 'M'),
                                                    ('Year', 'A')],
                                           value='default',
                                           description='Data aggregation')

        self.__data_aggregation.observe(self._set_data_aggregation, names='value')

        features = list(set(self._data.columns) - {'anomaly', 'anomaly_window'})
        self._feature = features[0]
        self.__feature = Dropdown(options=features,
                                  value=features[0],
                                  description='Features')
        self.__feature.observe(self._set_feature, names='value')

        self.__n_data_visualization = IntSlider(value=self._n_max_data,
                                                min=1,
                                                max=self._n_max_data,
                                                description='Data points',
                                                continuous_update=False)
        self.__n_data_visualization.observe(self._set_n_data_visualization, names='value')

        self.__n_data_update = IntSlider(value=self._n_data_update,
                                         min=2,
                                         max=self._n_max_data,
                                         description='Update steps',
                                         continuous_update=False)
        self.__n_data_update.observe(self._set_update_steps, names='value')

        self.__anomalies_cbox = Checkbox(value=True,
                                         description='Show anomalies')
        self.__anomalies_cbox.observe(self._set_show_anomalies, names='value')

        self.__a_windows_cbox = Checkbox(value=False,
                                         description='Show anomaly windows')
        self.__a_windows_cbox.observe(self._set_show_anomaly_windows, names='value')

        self.__out = Output()

        proposals = []
        for a in self.analyzers.keys():
            tmp_cbox = Checkbox(value=False,
                                description=a)

            tmp_cbox.observe(self._set_proposals_visibility, names='value')

            tmp_slider = FloatSlider(value=self._show_proposals[a]['threshold'],
                                     min=0.0,
                                     max=self._show_proposals[a]['max_value'],
                                     step=0.05,
                                     description='threshold',
                                     description_tooltip=a,
                                     continuous_update=False,
                                     disabled=True)
            tmp_slider.observe(self._set_proposals_threshold, names='value')

            tmp_error_cbox = Checkbox(value=False,
                                      description='show error types',
                                      description_tooltip=a,
                                      disabled=True)
            tmp_error_cbox.observe(self._set_show_errors, names='value')

            tmp_error_distance = IntSlider(value=20,
                                           min=0,
                                           max=self._n_data,
                                           description='affected distance',
                                           description_tooltip=a,
                                           continuous_update=False,
                                           disabled=True)
            tmp_error_distance.observe(self._set_distance_errors, names='value')

            self._show_proposals[a]['checkbox'] = tmp_cbox
            self._show_proposals[a]['slider'] = tmp_slider
            self._show_proposals[a]['error_checkbox'] = tmp_error_cbox
            self._show_proposals[a]['error_slider'] = tmp_error_distance

            proposals.append(HBox([tmp_cbox, tmp_slider, tmp_error_cbox, tmp_error_distance]))

        layout = Layout(display='flex', justify_content='center')

        widgets = VBox([HBox([self.__feature, self.__data_aggregation], layout=layout),
                        HBox([self.__n_data_visualization, self.__n_data_update], layout=layout),
                        HBox([self.__anomalies_cbox, self.__a_windows_cbox], layout=layout),
                        VBox(proposals, layout=layout),
                        HBox([self.__previous_btn, self.__next_btn], layout=layout),
                        HBox([self.__out], layout=layout)])
        return widgets

    def _on_next_clicked(self, btn):
        self._pos += self._n_data_update
        self._update_plot()

    def _on_previous_clicked(self, btn):
        self._pos -= self._n_data_update
        self._update_plot()

    def _set_proposals_visibility(self, change):
        self._show_proposals[change['owner'].description]['show'] = change['new']
        self._show_proposals[change['owner'].description]['slider'].disabled = (not change['new'])
        self._show_proposals[change['owner'].description]['error_checkbox'].disabled = (not change['new'])

        if change['new']:
            for m in self._show_proposals.keys() - {change['owner'].description}:
                if self._show_proposals[m]['show_error']:
                    self._show_proposals[m]['error_checkbox'].value = False

        self._update_plot()

    def _set_show_errors(self, change):
        self._show_proposals[change['owner'].description_tooltip]['show_error'] = change['new']
        self._show_proposals[change['owner'].description_tooltip]['error_slider'].disabled = (not change['new'])
        # disable other models
        if change['new']:
            for m in self._show_proposals.keys() - {change['owner'].description_tooltip}:
                if self._show_proposals[m]['show']:
                    self._show_proposals[m]['checkbox'].value = False

        self._update_plot()

    def _set_distance_errors(self, change):
        self._show_proposals[change['owner'].description_tooltip]['error_distance'] = change['new']
        self._update_plot()

    def _set_proposals_threshold(self, change):
        self._show_proposals[change['owner'].description_tooltip]['threshold'] = change['new']
        self._update_plot()

    def _set_data_aggregation(self, change):
        if change['new'] == 'default':
            self._data_to_show = self._data
        else:
            self._data_to_show = self._data.resample(change['new']).mean().pad()

        self._n_max_data = len(self._data_to_show.index) - 1
        self.__n_data_visualization.max = self._n_max_data
        self.__n_data_update.max = self._n_max_data
        self._update_plot()

    def _set_update_steps(self, change):
        self._n_data_update = change['new']

    def _set_n_data_visualization(self, change):
        self._n_data = change['new']
        self.__n_data_update.max = self._n_data
        self._update_plot()

    def _set_show_anomalies(self, change):
        self._show_anomalies = change['new']
        self._update_plot()

    def _set_show_anomaly_windows(self, change):
        self._show_anomaly_windows = change['new']
        self._update_plot()

    def _set_feature(self, change):
        self._feature = change['new']
        self._update_plot()

    def _setup_plot(self):
        plt.ioff()
        plt.figure(figsize=(15, 8))
        self._ax = plt.gca()

    def _update_plot(self):
        self._ax.clear()

        if self._pos <= 0:
            self._pos = 0
        elif self._pos + self._n_data > self._n_max_data:
            self._pos = self._n_max_data - self._n_data

        self.__previous_btn.disabled = (self._pos == 0)
        self.__next_btn.disabled = (self._pos >= self._n_max_data - self._n_data)

        # PLOT TimeSeries
        x = self._data_to_show.index[self._pos:self._pos+self._n_data]
        y = self._data_to_show[self._feature].values[self._pos:self._pos+self._n_data]
        self._ax.plot(x, y, label='TimeSeries', color='skyblue')

        # PLOT Anomalies
        if self._show_anomalies:
            anomalies = self._data_to_show.loc[
                (self._data_to_show.index.isin(x)) & (self._data_to_show['anomaly'] == 1)]
            self._ax.plot(anomalies.index, anomalies[self._feature].values, color='red', label='Anomaly',
                          linestyle='none', marker='X', markersize=10)

        # PLOT Anomaly Windows
        if self._show_anomaly_windows:
            min_v, max_v = min(y), max(y)
            for s, e in self.dataset.nab_config['windows_index']:
                i = np.where((x >= s) & (x <= e))[0]
                if len(i) == 0:
                    continue
                start = x[i[0]]
                end = x[i[-1]]

                self._ax.fill_betweenx([min_v, max_v], start, end, color=(1, 0, 0, 0.25))

        # PLOT Proposals
        for a in self._show_proposals.keys():
            if not self._show_proposals[a]['show']:
                continue
            proposals = self._show_proposals[a]['y_score'].copy()

            max_value, step_value = (1, 0.1) if self.analyzers[a].proposals_type == TSProposalsType.LABEL else (
            max(proposals['y_score'].values), max(proposals['y_score'].values) / 10)
            bins = np.arange(0, max_value, step_value).round(2)

            proposals = proposals.loc[proposals.index.isin(x) &
                                      (proposals['y_score'] >= self._show_proposals[a]['threshold'])]

            for i, b in enumerate(bins[1:]):
                if b < self._show_proposals[a]['threshold']:
                    continue
                tmp = proposals.loc[(proposals['y_score'] > bins[i]) & (proposals['y_score'] <= b)]
                tmp = self._data_to_show.loc[self._data_to_show.index.isin(tmp.index)]
                self._ax.plot(tmp.index, tmp[self._feature].values,
                              linestyle='none', color=to_rgba(self._show_proposals[a]['color'], 0.1*(i+1)), marker='X')

            tmp = proposals.loc[proposals['y_score'] > bins[-1]]
            tmp = self._data_to_show.loc[self._data_to_show.index.isin(tmp.index)]
            self._ax.plot(tmp.index, tmp[self._feature].values, label=a,
                          linestyle='none', color=self._show_proposals[a]['color'],  marker='X')

            # PLOT Errors
            if self._show_proposals[a]['show_error']:
                proposals = self._data_to_show.loc[self._data_to_show.index.isin(proposals.index)]
                _, _, errors = self.analyzers[a].analyze_false_positive_errors(distance=self._show_proposals[a]['error_distance'],
                                                                               threshold=self._show_proposals[a]['threshold'],
                                                                               show=False)
                for e, color in zip(errors.keys(), ['darkorange', 'forestgreen', 'royalblue']):
                    if errors[e]:
                        tmp = proposals.loc[proposals.index.isin(errors[e])]
                        self._ax.plot(tmp.index, tmp[self._feature].values, label=e,
                                      linestyle='none', color=color, marker='X')

        self._ax.legend(loc='upper right')
        with self.__out:
            clear_output(wait=True)
            display(self._ax.figure)

    def show(self):
        self._setup_plot()
        display(self.all_widgets)
        self._update_plot()


