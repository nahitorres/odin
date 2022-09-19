import copy
from numbers import Number

from odin.classes import Metrics, TaskType, DatasetClassification, AnalyzerClassification, Curves
from odin.classes.comparator_interface import ComparatorInterface
from odin.classes.strings import err_type, err_value
from odin.utils.draw_utils import plot_models_comparison_on_tp_fp_fn_tn
from odin.utils.env import is_notebook, get_root_logger

logger = get_root_logger()


class ComparatorClassification(ComparatorInterface):

    __classification_tasks = {TaskType.CLASSIFICATION_BINARY, TaskType.CLASSIFICATION_SINGLE_LABEL,
                              TaskType.CLASSIFICATION_MULTI_LABEL}
    _valid_metrics = [Metrics.ACCURACY, Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE,
                      Metrics.AVERAGE_PRECISION_SCORE, Metrics.ROC_AUC, Metrics.ERROR_RATE,
                      Metrics.PRECISION_RECALL_AUC, Metrics.F1_AUC]

    _valid_curves = [Curves.PRECISION_RECALL_CURVE, Curves.ROC_CURVE, Curves.F1_CURVE]

    __valid_cams_metrics = [Metrics.CAM_BBOX_COVERAGE, Metrics.CAM_COMPONENT_IOU, Metrics.CAM_GLOBAL_IOU,
                            Metrics.CAM_IRRELEVANT_ATTENTION]

    def __init__(self,
                 dataset_gt_param,
                 task_type,
                 multiple_proposals_path,
                 result_saving_path='./results/',
                 properties_file='properties.json',
                 use_normalization=False,
                 norm_factor_categories=None,
                 norm_factors_properties=None,
                 conf_thresh=None,
                 metric=Metrics.F1_SCORE,
                 similar_classes=None,
                 load_properties=True,
                 match_on_filename=False,
                 save_graph_as_png=True
                 ):
        """
        The ComparatorClassification class can be used to perform comparison of classification models.

        Parameters
        ----------
        dataset_gt_param: str
            Path of the ground truth .json file.
        task_type: TaskType
            Problem task type. It can be: TaskType.CLASSIFICATION_BINARY, TaskType.CLASSIFICATION_SINGLE_LABEL, TaskType.CLASSIFICATION_MULTI_LABEL.
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
        conf_thresh: float, optional
            Confidence threshold. All the predictions with a confidence value less than the threshold are ignored. If not specified, for single-label classification problems, the default value is 0. If not specified, for binary and multi-label classification problems the default value is 0.5. (default is None)
        metric: Metrics, optional
            The evaluation metric that will be used as default. (default is Metrics.F1_SCORE)
        similar_classes: list of list, optional
            List of groups of ids of categories which are similar to each other. (default is None)
        load_properties: bool, optional
            Indicates whether the properties should be loaded. (default is True)
        match_on_filename: bool, optional
            Indicates whether the predictions refer to the ground truth by file_name (set to True) or by id (set to False). (default is False)
        save_graph_as_png: bool, optional
            Indicates whether plots should be saved as .png images. (default is True)
        """

        if not isinstance(dataset_gt_param, str):
            raise TypeError(err_type.format("dataset_gt_param"))

        if not isinstance(task_type, TaskType):
            raise TypeError(err_type.format("task_type"))
        elif task_type not in self.__classification_tasks:
            raise ValueError(err_value.format("task_type", self.__classification_tasks))

        if not isinstance(multiple_proposals_path, list) or not all(isinstance(v[0], str) and (isinstance(v[1], str) or isinstance(v[1], list)) and len(v) > 1 for v in multiple_proposals_path):
            raise TypeError(err_type.format("multiple_proposals_path"))

        if not isinstance(result_saving_path, str):
            raise TypeError(err_type.format("result_saving_path"))

        if not isinstance(properties_file, str):
            raise TypeError(err_type.format("properties_file"))

        if not isinstance(use_normalization, bool):
            raise TypeError(err_type.format("use_normalization"))

        if norm_factor_categories is not None and not isinstance(norm_factor_categories, float):
            raise TypeError(err_type.format("norm_factor_categories"))

        if norm_factors_properties is not None and (not isinstance(norm_factors_properties, list) or not (all(isinstance(item, tuple) and len(item) == 2 for item in norm_factors_properties))):
            raise TypeError(err_type.format("norm_factors_properties"))

        if conf_thresh is not None:
            if not isinstance(conf_thresh, Number):
                raise TypeError(err_type.format("conf_thresh"))
            if conf_thresh < 0 or conf_thresh > 1:
                raise ValueError(err_value.format("conf_thresh", "0 <= x >= 1"))

        if not isinstance(metric, Metrics):
            raise TypeError(err_type.format("metric"))
        elif metric not in self._valid_metrics and (self._default_dataset is not None and metric not in self.__valid_cams_metrics):
            raise Exception(f"Invalid metric: {metric}")

        if not isinstance(load_properties, bool):
            raise TypeError(err_type.format("load_properties"))

        if not isinstance(match_on_filename, bool):
            raise TypeError(err_type.format("match_on_filename"))

        if not isinstance(save_graph_as_png, bool):
            raise TypeError(err_type.format("save_graphs_as_png"))

        if self._default_dataset is None:
            proposals_paths = [(m[0], m[1]) for m in multiple_proposals_path]
            self._default_dataset = DatasetClassification(dataset_gt_param,
                                                          task_type,
                                                          proposals_paths=proposals_paths,
                                                          result_saving_path=result_saving_path,
                                                          properties_file=properties_file,
                                                          similar_classes=similar_classes,
                                                          for_analysis=False,
                                                          match_on_filename=match_on_filename,
                                                          save_graphs_as_png=save_graph_as_png)
        super().__init__(task_type, multiple_proposals_path, result_saving_path, use_normalization,
                         norm_factor_categories, norm_factors_properties, conf_thresh, metric,
                         similar_classes, match_on_filename, save_graph_as_png)

        if not load_properties:
            self._load_all_models_proposals(load_properties=False)
            return

        if is_notebook():
            self._default_dataset._load_or_create_properties_notebook(self._default_dataset.observations,
                                                                      self._default_dataset.common_properties,
                                                                      self._load_all_models_proposals)
        else:
            self._default_dataset._load_or_create_properties(self._default_dataset.observations,
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
            tmp_analyzer = AnalyzerClassification(model_name,
                                                  self._default_dataset,
                                                  use_normalization=self.use_normalization,
                                                  norm_factor_categories=self.norm_factor_categories,
                                                  norm_factors_properties=self.norm_factors_properties,
                                                  conf_thresh=threshold,
                                                  metric=self.metric,
                                                  save_graphs_as_png=False)
            self.models[model_name] = {"dataset": self._default_dataset,
                                       "analyzer": tmp_analyzer}
        self._allow_analyses = True

    def show_true_negative_distribution(self, categories=None, models=None, show=True):
        """
        It compares the true negatives of the models.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        models: list, optional
            List of models on which to perform the analysis. If not specified, all models are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if not self._allow_analyses:
            logger.error("Please select the properties first")
            return -1

        if self._is_binary:
            logger.error("Not supported for binary classification.")
            return -1

        if categories is None:
            categories = self._default_dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self._default_dataset.are_valid_categories(categories):
            return -1

        if models is None:
            models = self.models.keys()
        elif not isinstance(models, list):
            logger.error(err_type.format("models"))
            return -1
        elif any(m not in self.models for m in models):
            logger.error(err_value.format("models", list(self.models.keys())))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        results = {}
        if len(models) == 1:
            results[models[0]] = self.models[models[0]]["analyzer"].show_true_negative_distribution(categories, show=show)
            if show:
                return
        else:
            for model in models:
                results[model] = self.models[model]["analyzer"].show_true_negative_distribution(categories, show=False)

        if not show:
            return results

        labels = [self._default_dataset.get_display_name_of_category(cat) for cat in categories]
        plot_models_comparison_on_tp_fp_fn_tn(results, labels, "True Negative comparison", "Categories", self.save_graph_as_png,
                                              self.result_saving_path)

    def show_true_negative_distribution_for_categories_for_property(self, property_name, property_values=None, categories=None, models=None, show=True):
        """
        It compares the true negative distribution of the property values for each category of the models.

        Parameters
        ----------
        property_name: str
            Name of the property to be analyzed.
        property_values: list
            Values of the property to be included in the analysis. If not specified, all the values are included. (default is None)
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        models: list, optional
            List of models on which to perform the analysis. If not specified, all models are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if not self._allow_analyses:
            logger.error("Please select the properties first")
            return -1

        if not self._default_dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset")
            return -1

        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        elif not self._default_dataset.are_valid_properties([property_name]):
            return -1

        if property_values is None:
            property_values = self._default_dataset.get_values_for_property(property_name)
        elif not isinstance(property_values, list):
            logger.error(err_type.format("property_values"))
            return -1
        elif not self._default_dataset.is_valid_property(property_name, property_values):
            return -1

        if categories is None:
            categories = [self._default_dataset.get_category_name_from_id(1)] if self._default_dataset.task_type == TaskType.CLASSIFICATION_BINARY else self._default_dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self._default_dataset.are_valid_categories(categories):
            return -1

        if models is None:
            models = self.models.keys()
        elif not isinstance(models, list):
            logger.error(err_type.format("models"))
            return -1
        elif any(m not in self.models for m in models):
            logger.error(err_value.format("models", list(self.models.keys())))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        results = {}
        tmp_results = {}
        if len(models) == 1:
            tmp_results[models[0]] = self.models[models[0]][
                "analyzer"].show_true_negative_distribution_for_categories_for_property(property_name, property_values,
                                                                                        categories=categories,
                                                                                        show=show)
            if show:
                return
        else:
            for model in models:
                tmp_results[model] = self.models[model]["analyzer"].show_true_negative_distribution_for_categories_for_property(property_name, property_values, categories=categories, show=False)

        for c in categories:
            results[c] = {}
            for model in models:
                results[c][model] = tmp_results[model][c]

        if not show:
            return results

        p_label = self._default_dataset.get_display_name_of_property(property_name)
        labels = [self._default_dataset.get_display_name_of_property_value(property_name, p_v) for p_v in
                  self._default_dataset.get_values_for_property(property_name)]
        for c in categories:
            c_label = self._default_dataset.get_display_name_of_category(c)
            plot_models_comparison_on_tp_fp_fn_tn(results[c], labels, f"True Negative comparison of {p_label} for {c_label}", "Property values", self.save_graph_as_png,
                                                  self.result_saving_path)

    def show_false_positive_distribution_for_categories_for_property(self, property_name, property_values=None, categories=None, models=None, show=True):
        """
        It compares the false positive distribution of the property values for each category of the models.

        Parameters
        ----------
        property_name: str
            Name of the property to be analyzed.
        property_values: list
            Values of the property to be included in the analysis. If not specified, all the values are included. (default is None)
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        models: list, optional
            List of models on which to perform the analysis. If not specified, all models are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if not self._allow_analyses:
            logger.error("Please select the properties first")
            return -1

        if not self._default_dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset")
            return -1

        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        elif not self._default_dataset.are_valid_properties([property_name]):
            return -1

        if property_values is None:
            property_values = self._default_dataset.get_values_for_property(property_name)
        elif not isinstance(property_values, list):
            logger.error(err_type.format("property_values"))
            return -1
        elif not self._default_dataset.is_valid_property(property_name, property_values):
            return -1

        if categories is None:
            categories = [self._default_dataset.get_category_name_from_id(1)] if self._default_dataset.task_type == TaskType.CLASSIFICATION_BINARY else self._default_dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self._default_dataset.are_valid_categories(categories):
            return -1

        if models is None:
            models = self.models.keys()
        elif not isinstance(models, list):
            logger.error(err_type.format("models"))
            return -1
        elif any(m not in self.models for m in models):
            logger.error(err_value.format("models", list(self.models.keys())))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        results = {}
        tmp_results = {}
        if len(models) == 1:
            tmp_results[models[0]] = self.models[models[0]][
                "analyzer"].show_false_positive_distribution_for_categories_for_property(property_name, property_values,
                                                                                         categories=categories,
                                                                                         show=show)
            if show:
                return
        else:
            for model in models:
                tmp_results[model] = self.models[model]["analyzer"].show_false_positive_distribution_for_categories_for_property(property_name, property_values, categories=categories, show=False)

        for c in categories:
            results[c] = {}
            for model in models:
                results[c][model] = tmp_results[model][c]

        if not show:
            return results

        p_label = self._default_dataset.get_display_name_of_property(property_name)
        labels = [self._default_dataset.get_display_name_of_property_value(property_name, p_v) for p_v in
                  self._default_dataset.get_values_for_property(property_name)]
        for c in categories:
            c_label = self._default_dataset.get_display_name_of_category(c)
            plot_models_comparison_on_tp_fp_fn_tn(results[c], labels, f"False Positive comparison of {p_label} for {c_label}", "Property values", self.save_graph_as_png,
                                                  self.result_saving_path)
