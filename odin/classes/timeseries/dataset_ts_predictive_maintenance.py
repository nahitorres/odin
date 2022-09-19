import os
import pandas as pd

from odin.classes import TaskType
from odin.classes.strings import err_type, err_value
from odin.classes.timeseries import DatasetTimeSeriesInterface, TSProposalsType
from odin.utils import get_root_logger


class DatasetTSPredictiveMaintenance(DatasetTimeSeriesInterface):

    def __init__(self,
                 dataset_path,
                 ts_type,
                 proposals_paths=None,
                 properties_path=None,
                 csv_separator=',',
                 index_gt='observation_id',
                 index_proposals='observation_id',
                 result_saving_path='./result',
                 save_graphs_as_png=False):

        if not isinstance(index_gt, str):
            raise TypeError(err_type.format('index_gt'))

        if not isinstance(index_proposals, str):
            raise TypeError(err_type.format('index_proposals'))

        self._index_gt = index_gt
        self._index_proposals = index_proposals

        super().__init__(dataset_path, TaskType.TS_PREDICTIVE_MAINTENANCE, ts_type, proposals_paths, properties_path,
                         csv_separator, result_saving_path, save_graphs_as_png)

    def get_proposals_type(self, model_name):
        if model_name not in self.proposals.keys():
            get_root_logger().error(err_value.format('model_name', list(self.proposals.keys())))
            return -1
        return self.proposals[model_name][1]

    def _load_gt(self, force_loading=False):
        if not force_loading and self.observations is not None:
            return

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError('Invalid data set path:', self.dataset_path)

        print('Loading data set...')

        data = pd.read_csv(self.dataset_path, sep=self.csv_separator)

        if self._index_gt not in data.columns:
            raise ValueError("'{}' column not present. Unable to index data.".format(self._index_gt))
        if 'unit_id' not in data.columns:
            raise ValueError("'unit_id' column not present. Unable to index data.")

        data = data.set_index([self._index_gt, 'unit_id'])
        data = data.sort_index()

        if 'label' in data.columns or 'RUL' in data.columns:
            self._analysis_available = True

        self.observations = data

        print('Done!')

    def _load_proposals(self, model_name, path, proposals_type, csv_sep=','):
        print("Loading proposals of {} model...".format(model_name))

        if not os.path.exists(path):
            raise FileNotFoundError('Invalid proposals path for {} model: {}'.format(model_name, path))

        proposals_names = self.observations.index.get_level_values('unit_id').unique()

        all_data = []
        for filename in proposals_names:
            file_path = os.path.join(path, f'{filename}.csv')
            if not os.path.exists(file_path):
                get_root_logger().warning('Invalid proposals file path for {} model: {}'.format(model_name, file_path))
                continue

            data = pd.read_csv(file_path, sep=self.csv_separator)
            data['unit_id'] = filename

            if self._index_proposals not in data.columns:
                raise ValueError("'{}' column not present. Unable to index data for {} model".format(self._index_proposals, model_name))

            # if self._time_index:
            #     data[self._index_proposals] = pd.to_datetime(data[self._index_proposals])

            if proposals_type == TSProposalsType.LABEL:
                if 'confidence' not in data.columns:
                    raise ValueError("'confidence' column not present for {} model".format(model_name))
            elif 'RUL' not in data.columns:
                raise ValueError("'RUL' column not present for {} model".format(model_name))

            all_data.append(data)

        data = pd.concat(all_data)
        data = data.set_index([self._index_proposals, 'unit_id'])
        data = data.sort_index()

        print('Done!')

        return data, proposals_type

