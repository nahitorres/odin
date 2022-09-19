import pandas as pd
from IPython.core.display import display, clear_output
from ipywidgets import Button, Dropdown, IntSlider, FloatSlider, Checkbox, Output, VBox, HBox, Layout, Label
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from odin.classes.timeseries import TSProposalsType
from odin.classes.timeseries.visualizer_ts_interface import VisualizerTimeSeriesInterface


class VisualizerTSPredictiveMaintenance(VisualizerTimeSeriesInterface):
    __matching_types = {'label': TSProposalsType.LABEL,
                      'RUL': TSProposalsType.REGRESSION}

    def __init__(self, dataset, analyzers=None):

        analyzers_dict = {}
        if analyzers is not None:
            for a in analyzers:
                analyzers_dict[a.model_name] = a

        super().__init__(dataset, analyzers_dict)
        self._data = self.dataset.get_observations()

        unit_id = list(self._data.index.get_level_values('unit_id').unique())[0]
        self._data_to_show = self._data.loc[self._data.index.get_level_values('unit_id') == unit_id]

        self._n_max_data = len(self._data_to_show.index) - 1

        self._pos = 0
        self._features = {}
        self._n_data = self._n_max_data
        self._n_data_update = round(self._n_max_data/2)

        self._show_pm = False
        self._pm_type = None
        self._show_proposals = self._init_proposals()

        self.all_widgets = self._create_widgets()

        self._ax = []

    def _init_proposals(self):
        colors = ['purple', 'olive', 'grey', 'brown', 'pink']
        proposals = {}
        for i, a in enumerate(self.analyzers.keys()):
            proposals[a] = {'proposals': self.dataset.get_proposals(self.analyzers[a].model_name),
                            'threshold': self.analyzers[a]._threshold,
                            'show': False,
                            'type': self.analyzers[a].proposals_type,
                            'color': colors[i % len(colors)]}
        return proposals

    def _create_widgets(self):
        self.__next_btn = Button(description='next', disabled=True)
        self.__next_btn.on_click(self._on_next_clicked)

        self.__previous_btn = Button(description='previous', disabled=True)
        self.__previous_btn.on_click(self._on_previous_clicked)

        unit_ids = list(self._data.index.get_level_values('unit_id').unique())
        self.__unit_id = Dropdown(options=unit_ids,
                                  value=unit_ids[0],
                                  description='Unit ID')

        self.__unit_id.observe(self._set_unit_id, names='value')

        self.__show_pm = Checkbox(value=False,
                                  description='Show PM',
                                  disabled=(not self.dataset._analysis_available))
        self.__show_pm.observe(self._set_show_pm, names='value')

        pm_types = list({'label', 'RUL'}.intersection(set(self._data.columns)))
        self.__pm_type = Dropdown(options=pm_types,
                                  value=pm_types[0],
                                  description='PM type',
                                  disabled=True)
        self._pm_type = pm_types[0]
        self.__pm_type.observe(self._set_pm_type, names='value')

        features = sorted(list(set(self._data.columns) - {'label', 'RUL'}))
        features_checkbox = []
        for f in features:
            self._features[f] = False
            tmp_c = Checkbox(value=False,
                             description=f)
            tmp_c.observe(self._set_feature, names='value')
            features_checkbox.append(tmp_c)

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

        self.__out = Output()

        proposals = []
        if self.analyzers:
            proposals.append(Label(value='Models'))

        for a in self.analyzers.keys():
            tmp_cbox = Checkbox(value=False,
                                description=a,
                                disabled=True)

            tmp_cbox.observe(self._set_proposals_visibility, names='value')

            self._show_proposals[a]['checkbox'] = tmp_cbox
            if self.analyzers[a].proposals_type == TSProposalsType.LABEL:
                tmp_slider = FloatSlider(value=self._show_proposals[a]['threshold'],
                                         min=0.0,
                                         max=1,
                                         step=0.05,
                                         description='threshold',
                                         description_tooltip=a,
                                         continuous_update=False,
                                         disabled=True)
                tmp_slider.observe(self._set_proposals_threshold, names='value')

                self._show_proposals[a]['slider'] = tmp_slider
                proposals.append(HBox([tmp_cbox, tmp_slider]))
            else:
                proposals.append(HBox([tmp_cbox]))

        layout = Layout(display='flex', justify_content='center')

        widgets = VBox([HBox([self.__n_data_visualization, self.__n_data_update], layout=layout),
                        HBox([self.__unit_id], layout=layout),
                        HBox([self.__show_pm, self.__pm_type], layout=layout),
                        Label(value='Features'),
                        HBox(features_checkbox, layout=Layout(display='inline-flex', flex_flow='row wrap', justify_content='center')),
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

    def _set_unit_id(self, change):
        self._data_to_show = self._data.loc[self._data.index.get_level_values('unit_id') == change['new']]
        self._n_max_data = len(self._data_to_show.index) - 1
        self.__n_data_visualization.max = self._n_max_data
        self.__n_data_update.max = self._n_max_data
        self._update_plot()

    def _set_show_pm(self, change):
        self._show_pm = change['new']
        self.__pm_type.disabled = not self._show_pm
        for k in self._show_proposals.keys():
            self._show_proposals[k]['checkbox'].disabled = ((self.__matching_types[self._pm_type] != self._show_proposals[k]['type']) or not self._show_pm)

        self._reset_plot()
        self._update_plot()

    def _set_pm_type(self, change):
        self._pm_type = change['new']
        for k in self._show_proposals.keys():
            self._show_proposals[k]['checkbox'].disabled = (self.__matching_types[self._pm_type] != self._show_proposals[k]['type'])

        self._update_plot()

    def _set_proposals_visibility(self, change):
        self._show_proposals[change['owner'].description]['show'] = change['new']
        if 'slider' in self._show_proposals[change['owner'].description].keys():
            self._show_proposals[change['owner'].description]['slider'].disabled = (not change['new'])

        self._update_plot()

    def _set_proposals_threshold(self, change):
        self._show_proposals[change['owner'].description_tooltip]['threshold'] = change['new']
        self._update_plot()

    def _set_update_steps(self, change):
        self._n_data_update = change['new']

    def _set_n_data_visualization(self, change):
        self._n_data = change['new']
        self.__n_data_update.max = self._n_data
        self._update_plot()

    def _set_feature(self, change):
        self._features[change['owner'].description] = change['new']
        self._reset_plot()
        self._update_plot()

    def _reset_plot(self):
        with self.__out:
            self.__out.clear_output()

        n_subplots = 0 if not self._show_pm else 1
        for f in self._features.keys():
            if self._features[f]:
                n_subplots += 1

        plt.ioff()
        self._figure, self._ax = plt.subplots(n_subplots, 1, figsize=(15, 4*n_subplots))
        if n_subplots == 1:
            self._ax = [self._ax]

    def __update_pm_plot(self):
        ax = self._ax[0]
        ax.clear()

        x = self._data_to_show.index.get_level_values(self.dataset._index_gt).values[self._pos:self._pos + self._n_data]
        y = self._data_to_show[self._pm_type].values[self._pos:self._pos + self._n_data]
        ax.plot(x, y, label=self._pm_type, color='skyblue')
        for k in self._show_proposals.keys():
            if not self._show_proposals[k]['show']:
                continue
            proposals = self._show_proposals[k]['proposals'].loc[self._show_proposals[k]['proposals'].index.get_level_values(self.dataset._index_proposals).isin(x)]
            if proposals.empty:
                continue
            if self._show_proposals[k]['type'] == TSProposalsType.LABEL:
                v = np.where(proposals['confidence'] >= self._show_proposals[k]['threshold'], 1, 0)
                ax.plot(proposals.index.get_level_values(self.dataset._index_proposals), v, label=k, color=self._show_proposals[k]['color'])

            else:
                v = proposals['RUL'].values
                ax.plot(proposals.index.get_level_values(self.dataset._index_proposals), v, label=k,
                        color=self._show_proposals[k]['color'])
        ax.legend()

        self.__update_features_plot(self._ax[1:])

    def __update_features_plot(self, axs):
        i = 0
        for f in self._features.keys():
            if not self._features[f]:
                continue

            ax = axs[i]
            ax.clear()

            x = self._data_to_show.index.get_level_values(self.dataset._index_gt).values[self._pos:self._pos + self._n_data]
            y = self._data_to_show[f].values[self._pos:self._pos + self._n_data]
            ax.plot(x, y, label=f)
            ax.legend()
            i += 1

    def _update_plot(self):
        if self._pos <= 0:
            self._pos = 0
        elif self._pos + self._n_data > self._n_max_data:
            self._pos = self._n_max_data - self._n_data

        self.__previous_btn.disabled = (self._pos == 0)
        self.__next_btn.disabled = (self._pos >= self._n_max_data - self._n_data)

        if self._show_pm:
            self.__update_pm_plot()
        else:
            self.__update_features_plot(self._ax)

        plt.close('all')
        if len(self._ax) > 0:
            with self.__out:
                clear_output(wait=True)
                display(self._figure)

    def show(self):
        display(self.all_widgets)


