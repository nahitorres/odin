from numbers import Number

from odin.classes import Metrics, TaskType, DatasetLocalization, AnalyzerLocalization, Curves
from odin.classes.comparator_interface import ComparatorInterface
from odin.classes.strings import err_type, err_value
from odin.utils.env import is_notebook


class ComparatorLocalization(ComparatorInterface):

    __localization_tasks = {TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION}
    _valid_metrics = [Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE,
                      Metrics.AVERAGE_PRECISION_SCORE, Metrics.ROC_AUC, Metrics.PRECISION_RECALL_AUC, Metrics.F1_AUC,
                      Metrics.AVERAGE_PRECISION_INTERPOLATED]

    _valid_curves = [Curves.PRECISION_RECALL_CURVE, Curves.F1_CURVE]

    def __init__(self,
                 dataset_gt_param,
                 task_type,
                 multiple_proposals_path,
                 result_saving_path='./results/',
                 properties_file='properties.json',
                 use_normalization=True,
                 norm_factor_categories=None,
                 norm_factors_properties=None,
                 iou=None,
                 iou_weak=None,
                 conf_thresh=None,
                 metric=Metrics.AVERAGE_PRECISION_SCORE,
                 similar_classes=None,
                 load_properties=True,
                 match_on_filename=False,
                 save_graph_as_png=True,
                 ignore_proposals_threshold=0.01
                 ):
        """
        The ComparatorLocalization class can be used to perform comparison of localization models, such as object detection and instance segmentation.

        Parameters
        ----------
        dataset_gt_param: str
            Path of the ground truth .json file.
        task_type: TaskType
            Problem task type. It can be: TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION.
        multiple_proposals_path: list of list
            List of the models proposals path. For each model, it must be specified the model name and the model proposals path directory; optionally can be specified also the default confidence threshold. Example: multiple_proposals_path = [['model_a', 'model_a_path', 0.65], ['model_b', 'model_b_path']]
        result_saving_path: str, optional
            Path used to save results. (default is './results/')
        properties_file: str, optional
            The name of the file used to store the names of and values of the properties and the names of the categories. (default is 'properties.json')
        use_normalization: bool, optional
            Indicates whether normalisation should be used. (default is False)
        norm_factor_categories: float, optional
            Normalisation factor for the categories. If not specified, the default value is 1/number_of_categories. (default is None)
        norm_factors_properties: list of pairs, optional
            Normalization factor for the properties. Each pair (property_name, value) specifies the normalisation factor to be applied to a specific property. If not specified, for each property the default value is 1/number_of_property_values. (default is None)
        iou: float, optional
            Intersection Over Union threshold. All the predictions with a iou value less than the threshold are considered False Positives. If not specified, the default value is 0.5. (default is None)
        iou_weak: float, optional
            Intersection Over Union weak threshold. Used for the identification of the localization errors. If not specified, the default value is 0.1. (default is None)
        conf_thresh: float, optional
            Confidence threshold. All the predictions with a confidence value less than the threshold are ignored. If not specified the default value is 0.5. (default is None)
        metric: Metrics, optional
            The evaluation metric that will be used as default. (default is Metrics.AVERAGE_PRECISION_SCORE)
        similar_classes: list of list, optional
            List of groups of ids of categories which are similar to each other. (default is None)
        load_properties: bool, optional
            Indicates whether the properties should be loaded. (default is True)
        match_on_filename: bool, optional
            Indicates whether the predictions refer to the ground truth by file_name (set to True) or by id (set to False). (default is False)
        save_graph_as_png: bool, optional
            Indicates whether plots should be saved as .png images. (default is True)
        ignore_proposals_threshold: float, optional
            All the proposals with a confidence score lower than the threshold are not loaded. (Default is 0.01)
        """
        if not isinstance(dataset_gt_param, str):
            raise TypeError(err_type.format("dataset_gt_param"))

        if not isinstance(task_type, TaskType):
            raise TypeError(err_type.format("task_type"))
        elif task_type not in self.__localization_tasks:
            raise Exception(err_value.format("task_type", self.__localization_tasks))

        if not isinstance(multiple_proposals_path, list) or not all(
                isinstance(v[0], str) and isinstance(v[1], str) and len(v) > 1 for v in multiple_proposals_path):
            raise TypeError(err_type.format("multiple_proposals_path"))

        if not isinstance(result_saving_path, str):
            raise TypeError(err_type.format("result_saving_path"))

        if not isinstance(properties_file, str):
            raise TypeError(err_type.format("properties_file"))

        if not isinstance(use_normalization, bool):
            raise TypeError(err_type.format("use_normalization"))

        if norm_factor_categories is not None and not isinstance(norm_factor_categories, float):
            raise TypeError(err_type.format("norm_factor_categories"))

        if norm_factors_properties is not None and (not isinstance(norm_factors_properties, list) or not (
        all(isinstance(item, tuple) and len(item) == 2 for item in norm_factors_properties))):
            raise TypeError(err_type.format("norm_factors_properties"))

        if iou is None:
            iou = 0.5
        elif not isinstance(iou, Number):
            raise TypeError(err_type.format("iou"))
        elif iou < 0 or iou > 1:
            raise ValueError(err_value.format("iou", "0 <= x >= 1"))

        if iou_weak is None:
            iou_weak = 0.1
        elif not isinstance(iou_weak, Number):
            raise TypeError(err_type.format("iou_weak"))
        elif iou_weak < 0 or iou_weak > 1:
            raise ValueError(err_value.format("iou_weak", "0 <= x >= 1"))
        elif iou_weak >= iou:
            raise ValueError(err_value.format("iou_weak", "iou_weak < iou"))

        if conf_thresh is not None:
            if not isinstance(conf_thresh, Number):
                raise TypeError(err_type.format("conf_thresh"))
            if conf_thresh < 0 or conf_thresh > 1:
                raise ValueError(err_value.format("conf_thresh", "0 <= x >= 1"))

        if not isinstance(metric, Metrics):
            raise TypeError(err_type.format("metric"))
        elif metric not in self._valid_metrics:
            raise Exception(err_value.format("metric", self._valid_metrics))

        if not isinstance(match_on_filename, bool):
            raise TypeError(err_type.format("match_on_filename"))

        if not isinstance(load_properties, bool):
            raise TypeError(err_type.format("load_properties"))

        if not isinstance(save_graph_as_png, bool):
            raise TypeError(err_type.format("save_graphs_as_png"))

        proposals_paths = [(m[0], m[1]) for m in multiple_proposals_path]
        self._default_dataset = DatasetLocalization(dataset_gt_param,
                                                    task_type,
                                                    proposals_paths=proposals_paths,
                                                    result_saving_path=result_saving_path,
                                                    properties_file=properties_file,
                                                    similar_classes=similar_classes,
                                                    for_analysis=False,
                                                    match_on_filename=match_on_filename,
                                                    save_graphs_as_png=save_graph_as_png,
                                                    ignore_proposals_threshold=ignore_proposals_threshold)
        self.iou = iou
        self.iou_weak = iou_weak
        super().__init__(task_type, multiple_proposals_path, result_saving_path, use_normalization,
                         norm_factor_categories, norm_factors_properties, conf_thresh, metric,
                         similar_classes, match_on_filename, save_graph_as_png)
        if not load_properties:
            self._load_all_models_proposals(load_properties=False)
            return

        if is_notebook():
            self._default_dataset._load_or_create_properties_notebook(self._default_dataset.annotations,
                                                                      self._default_dataset.common_properties,
                                                                      self._load_all_models_proposals)
        else:
            self._default_dataset._load_or_create_properties(self._default_dataset.annotations,
                                                             self._default_dataset.common_properties)
            self._load_all_models_proposals()

    def _load_all_models_proposals(self, load_properties=True):
        """
        Loads the proposals of all the models
        """
        self._default_dataset.for_analysis = True

        if load_properties:
            self._default_dataset._set_analyses_with_properties_available()
            self._default_dataset.load_proposals()
        else:
            self._default_dataset.load(load_properties=False)

        for p in self.proposals_path:
            model_name = p[0]
            threshold = p[2] if len(p) > 2 else self.conf_thresh
            tmp_analyzer = AnalyzerLocalization(model_name,
                                                self._default_dataset,
                                                use_normalization=self.use_normalization,
                                                norm_factor_categories=self.norm_factor_categories,
                                                norm_factors_properties=self.norm_factors_properties,
                                                iou=self.iou,
                                                iou_weak=self.iou_weak,
                                                conf_thresh=threshold,
                                                metric=self.metric,
                                                save_graphs_as_png=False)
            self.models[model_name] = {"dataset": self._default_dataset,
                                       "analyzer": tmp_analyzer}

        self._allow_analyses = True
