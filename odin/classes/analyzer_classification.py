import math
from collections import defaultdict
from numbers import Number
from statistics import mean

import numpy as np
import pandas as pd

from sklearn.metrics import auc, multilabel_confusion_matrix, confusion_matrix

from odin.classes import DatasetClassification, TaskType, Metrics, Curves
from odin.classes.analyzer_interface import AnalyzerInterface
from odin.classes.dataset_cams import DatasetCAMs
from odin.classes.strings import err_type, err_value
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_false_positive_errors, plot_reliability_diagram, \
    display_confusion_matrix, display_confusion_matrix_categories, plot_class_distribution, display_top1_top5_error, \
    pie_plot, plot_false_positive_trend

logger = get_root_logger()


class AnalyzerClassification(AnalyzerInterface):

    _SAVE_PNG_GRAPHS = True  # if set to false graphs will just be displayed

    __normalized_number_of_observations = 1000

    _valid_metrics = [Metrics.ACCURACY, Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE,
                      Metrics.AVERAGE_PRECISION_SCORE, Metrics.ROC_AUC, Metrics.ERROR_RATE,
                      Metrics.PRECISION_RECALL_AUC, Metrics.F1_AUC]

    _valid_curves = [Curves.PRECISION_RECALL_CURVE, Curves.ROC_CURVE, Curves.F1_CURVE]

    _valid_cams_metrics = []

    matching_dict = {}

    def __init__(self,
                 classifier_name,
                 dataset,
                 result_saving_path='./results/',
                 use_normalization=False,
                 norm_factor_categories=None,
                 norm_factors_properties=None,
                 conf_thresh=None,
                 metric=Metrics.F1_SCORE,
                 save_graphs_as_png=True):
        """
        The AnalyzerClassification class can be used to perform diagnostics for classification models.

        Parameters
        ----------
        classifier_name: str
            Name of the classifier. It is used as folder to save results.
        dataset: DatasetClassification
            Dataset used to perform the analysis.
        result_saving_path: str, optional
            Path used to save results. (default is './results/')
        use_normalization: bool, optional
            Indicates whether normalisation should be used. (default is False)
        norm_factor_categories: float, optional
            Normalisation factor for the categories. If not specified, the default value is 1/number_of_categories. (default is None)
        norm_factors_properties: list of pair, optional
            Properties normalization factors.

            Each pair (property name, normalization factor value) specifies the normalization factor to apply to a
            specific property.
            The default value for each property is 1/number of property values
        conf_thresh: float, optional
            Confidence threshold. Predictions with a confidence value less than the threshold are ignored.

            For single-label problem the default value is 0.
            For binary and multi-label problems the default value is 0.5
        metric: Metrics, optional
            The evaluation metric that will be used as default. (default is Metrics.F1_SCORE)
        save_graphs_as_png: bool, optional
            Indicates whether plots should be saved as .png images. (default is True)
        """
        if not isinstance(classifier_name, str):
            raise TypeError(err_type.format("classifier_name"))

        if type(dataset) is not DatasetCAMs:
            if type(dataset) is not DatasetClassification:
                raise TypeError(err_type.format("dataset"))
            if classifier_name not in dataset.proposals:
                loaded_models = list(dataset.proposals.keys())
                if len(loaded_models) == 1 and "model" in loaded_models:
                    dataset.proposals[classifier_name] = dataset.proposals["model"]
                    del dataset.proposals["model"]
                else:
                    raise Exception(
                        "No proposals. Unable to perform any type of analysis. "
                        "Please make sure to load the proposals to the dataset for {} model".format(classifier_name))

        if not dataset.are_analyses_without_properties_available():
            raise Exception("Please complete the properties selection first")
        if not dataset.are_analyses_with_properties_available():
            logger.warning("No properties available")

        if not isinstance(result_saving_path, str):
            raise TypeError(err_type.format("result_saving_path"))

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

        if not isinstance(save_graphs_as_png, bool):
            raise TypeError(err_type.format("save_graphs_as_png"))

        if metric not in self._valid_metrics and metric not in self._valid_cams_metrics:
            raise Exception(f"Invalid metric: {metric}")

        if conf_thresh is None:
            conf_thresh = 0 if dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL else 0.5

        self._SAVE_PNG_GRAPHS = save_graphs_as_png
        self.matching_dict = {}

        super().__init__(classifier_name, dataset, result_saving_path, use_normalization, norm_factor_categories,
                         norm_factors_properties, conf_thresh, metric)

    def analyze_reliability(self, num_bins=10, show=True):
        """
        It provides the reliability analysis by showing the distribution of the proposals among different confidence values and the corresponding confidence calibration.

        Parameters
        ----------
        num_bins: int, optional
            The number of bins the confidence values are split into. (default is 10)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not isinstance(num_bins, int):
            logger.error(err_type.format("num_bins"))
            return -1
        if num_bins < 2 or num_bins > 50:
            logger.error(err_value.format("num_bins", "2 <= x >= 50"))
            return -1

        if "reliability" not in self.saved_analyses:
            self.saved_analyses["reliability"] = {}

        if "overall" not in self.saved_analyses["reliability"]:
            self.saved_analyses["reliability"]["overall"] = {}

        if str(num_bins) not in self.saved_analyses["reliability"]["overall"]:
            if "all" not in self.matching_dict:
                self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
            matching = self.matching_dict["all"].copy()

            y_true, y_pred, y_score = self.__convert_input_reliability(matching)
            result = self._calculate_reliability(y_true, y_pred, y_score, num_bins)
            self.saved_analyses["reliability"]["overall"][str(num_bins)] = result
        else:
            result = self.saved_analyses["reliability"]["overall"][str(num_bins)]

        if not show:
            return result

        plot_reliability_diagram(result, self._SAVE_PNG_GRAPHS, self.result_saving_path, is_classification=True)

    def analyze_reliability_for_categories(self, categories=None, num_bins=10, show=True):
        """
        It provides the reliability analysis by showing the distribution of the proposals among different confidence values and the corresponding confidence calibration.

        Parameters
        ----------
        categories: list, optional
            If not specified, it performs the analysis on the entire data set, otherwise it performs a per-category analysis. (default is None)
        num_bins: int, optional
            The number of bins the confidence values are split into. (default is 10)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            logger.error("Analysis not supported for binary classification")
            return -1

        if not isinstance(num_bins, int):
            logger.error(err_type.format("num_bins"))
            return -1
        if num_bins < 2 or num_bins > 50:
            logger.error(err_value.format("num_bins", "2 <= x >= 50"))
            return -1

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        if "reliability" not in self.saved_analyses:
            self.saved_analyses["reliability"] = {}

        results = {}

        if "all" not in self.matching_dict:
            self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
        matching = self.matching_dict["all"].copy()

        for category in categories:
            if category not in self.saved_analyses["reliability"]:
                self.saved_analyses["reliability"][category] = {}
            if str(num_bins) not in self.saved_analyses["reliability"][category]:
                y_true, y_pred, y_score = self.__convert_input_reliability(matching, category)
                result = self._calculate_reliability(y_true, y_pred, y_score, num_bins, category=category)
                self.saved_analyses["reliability"][category][str(num_bins)] = result
            else:
                result = self.saved_analyses["reliability"][category][str(num_bins)]

            results[category] = result

            if show:
                plot_reliability_diagram(result, self._SAVE_PNG_GRAPHS, self.result_saving_path,
                                         is_classification=True,
                                         category=self.dataset.get_display_name_of_category(category))
        if not show:
            return results

    def analyze_false_positive_errors_for_category(self, category, metric=None, show=True):
        """
        It analyzes the false positives for a specific category, by identifying the type of the errors and shows the gain that the model could achieve by removing all the false positives of each type.

        Parameters
        ----------
        category: str
            Name of the category to be analyzed.
        metric: Metrics, optional
            Evaluation metric used for the analysis. If not specified, the default one is used. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1
        elif not self.dataset.is_valid_category(category):
            logger.error(err_value.format("category", self.dataset.get_categories_names()))
            return -1

        if metric is None:
            metric = self.metric
        elif not isinstance(metric, Metrics):
            logger.error(err_type.format("metric"))
            return -1
        elif not self._is_valid_metric(metric):
            logger.error(err_value.format("metric", self._valid_metrics))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        if "false_positive_analysis" not in self.saved_analyses:
            self.saved_analyses["false_positive_analysis"] = {}

        if metric not in self.saved_analyses["false_positive_analysis"]:
            self.saved_analyses["false_positive_analysis"][metric] = {}

        if category not in self.saved_analyses["false_positive_analysis"][metric]:
            threshold = self._conf_thresh if metric in self._get_metrics_with_threshold() else 0

            error_dict_total, _ = self._analyze_false_positive_errors([category], threshold)
            error_dict = error_dict_total[category]
            values = self._calculate_metric_for_category(category, metric)

            category_metric_value = values['value']
            errors = ["similar", "background", "other"] if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL else ["similar", "other"]
            error_values = []

            for error in errors:
                if len(error_dict[error]) == 0:
                    error_values.append([category_metric_value, 0])
                    continue

                observations = self.dataset.get_all_observations()
                proposals = self.dataset.get_proposals(self._model_name)
                props = proposals.loc[~proposals["id"].isin(error_dict[error])]

                matching = self._match_classification_with_ground_truth(observations, props)

                y_true, y_score = self.__convert_input_format_for_category(matching, category)
                self._set_normalized_number_of_observations_for_categories()
                metric_value, _ = self._compute_metric(y_true, y_score, metric, None)

                count_error = len(error_dict[error])
                error_values.append([metric_value, count_error])

            self.saved_analyses["false_positive_analysis"][metric][category] = {"error_values": error_values,
                                                                        "errors": errors,
                                                                        "category_metric_value": category_metric_value}

        error_values = self.saved_analyses["false_positive_analysis"][metric][category]["error_values"]
        errors = self.saved_analyses["false_positive_analysis"][metric][category]["errors"]
        category_metric_value = self.saved_analyses["false_positive_analysis"][metric][category]["category_metric_value"]

        if not show:
            return error_values, errors, category_metric_value

        plot_false_positive_errors(error_values, errors, category_metric_value,
                                   self.dataset.get_display_name_of_category(category), metric,
                                   self.result_saving_path, self._SAVE_PNG_GRAPHS)

    def analyze_false_positive_trend_for_category(self, category, include_correct_predictions=True, show=True):
        """
        It analyzes the trend of the false positives by indicating the percentage of each error type.

        Parameters
        ----------
        category: str
            Name of the category to be analyzed.
        include_correct_predictions: bool, optional
            Indicates whether the correct detections should be included in the trend analysis or not. (default is True)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            logger.error("Not supported for binary classification")
            return -1

        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1
        elif not self.dataset.is_valid_category(category):
            logger.error(err_value.format("category", self.dataset.get_categories_names()))
            return -1

        if not isinstance(include_correct_predictions, bool):
            logger.error(err_type.format("include_correct_detections"))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        if "false_positive_trend" not in self.saved_analyses:
            self.saved_analyses["false_positive_trend"] = {}

        if category not in self.saved_analyses["false_positive_trend"]:
            self.saved_analyses["false_positive_trend"][category] = {}

        if str(include_correct_predictions) not in self.saved_analyses["false_positive_trend"][category]:
            error_dict_total, _ = self._analyze_false_positive_errors([category], self._conf_thresh)
            error_dict = error_dict_total[category]

            if "all" not in self.matching_dict:
                self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
            matching = self.matching_dict["all"].copy()

            fp = self.__get_fp_dataframe_for_category_from_matching(matching, self.dataset.get_category_id_from_name(category), self._conf_thresh)

            labels = list(error_dict.keys())

            if include_correct_predictions:
                tp = self.__get_tp_dataframe_for_category_from_matching(matching, self.dataset.get_category_id_from_name(category))
                match = pd.concat([tp, fp])
                match = match.sort_values("confidence", ascending=False)
                error_trend = [np.cumsum(np.where(match["id_y"].isin(tp["id_y"]), 1, 0))]
                error_trend.extend([np.cumsum(np.where(match["id_y"].isin(error_dict[error]), 1, 0)) for error in error_dict])
                error_trend = np.array(error_trend)
                labels.insert(0, "correct")
            else:
                fp = fp.sort_values("confidence", ascending=False)
                error_trend = np.array([np.cumsum(np.where(fp["id_y"].isin(error_dict[error]), 1, 0)) for error in error_dict])

            error_sum = np.sum(error_trend, axis=0)
            error_trend = np.divide(error_trend, error_sum)

            result = {}
            for i, error in enumerate(labels):
                if error == "background" and self.dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
                    continue
                result[error] = error_trend[i, :]

            self.saved_analyses["false_positive_trend"][category][str(include_correct_predictions)] = result

        result = self.saved_analyses["false_positive_trend"][category][str(include_correct_predictions)]

        if not show:
            return result

        plot_false_positive_trend(result, "False Positive Trend - " + self.dataset.get_display_name_of_category(category), self._SAVE_PNG_GRAPHS, self.result_saving_path)

    def analyze_false_negative_errors_for_category(self, category, show=True):
        """
        It analyzes the false negative errors for a specific category, by identifying the type of the errors.
        Parameters
        ----------
        category: str
            Name of the category to be included in the analysis.
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            logger.error("Analysis not supported for binary classification")
            return -1

        if not isinstance(category, str):
            logger.error(err_type.format("category"))
            return -1
        elif not self.dataset.is_valid_category(category):
            logger.error(err_value.format("category", self.dataset.get_categories_names()))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        if "false_negative_errors" not in self.saved_analyses:
            self.saved_analyses["false_negative_errors"] = {}

        if category not in self.saved_analyses["false_negative_errors"]:
            self.saved_analyses["false_negative_errors"][category] = {}

            _, fn_ids = self._analyze_false_negative_for_categories([category])

            observations = self.dataset.get_observations_from_ids(fn_ids[category]["gt"])
            proposals = self.dataset.get_proposals(self._model_name).copy()
            cat_id = self.dataset.get_category_id_from_name(category)

            if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
                proposals = proposals.loc[proposals["confidence"] >= self._conf_thresh]
                proposals = proposals.groupby(self.dataset.match_param_props)["category_id"].apply(list).reset_index(name="cat_props")
                matching = pd.merge(observations, proposals, how="left", left_on=self.dataset.match_param_gt,
                                    right_on=self.dataset.match_param_props).replace(np.nan, 0)
                matching["fn_categorization"] = np.where(matching.apply(lambda x: self.__is_false_negative_ml_similar_classes(x["categories"], x["cat_props"]),
                                                                        axis=1),
                                                         1, 2)

            else:  # SINGLE_LABEL
                proposals = proposals.sort_values(by="confidence", ascending=False).groupby(self.dataset.match_param_props).first().reset_index()
                proposals = proposals.loc[proposals["confidence"] >= self._conf_thresh]

                matching = pd.merge(observations, proposals, how="left", left_on=self.dataset.match_param_gt,
                                    right_on=self.dataset.match_param_props).replace(np.nan, 0)
                matching["fn_categorization"] = np.where(
                    matching["category_id"].apply(lambda x: self.dataset.is_similar(cat_id, x)), 1, 2)

            self.saved_analyses["false_negative_errors"][category]["similar"] = len(matching.loc[matching["fn_categorization"] == 1].index)
            self.saved_analyses["false_negative_errors"][category]["other"] = len(matching.loc[matching["fn_categorization"] == 2].index)

        if not show:
            return self.saved_analyses["false_negative_errors"][category]

        display_name = self.dataset.get_display_name_of_category(category)
        pie_plot(list(self.saved_analyses["false_negative_errors"][category].values()),
                 list(self.saved_analyses["false_negative_errors"][category].keys()),
                 "False Negative categorization - {}".format(display_name),
                 self.result_saving_path, self._SAVE_PNG_GRAPHS, colors=['lightskyblue', 'orchid'])

    def show_confusion_matrix(self, categories=None, properties_names=None, properties_values=None, show=True):
        """
        It shows the confusion matrix of the model. The confusion matrix can be performed for the entire data set or for a subset with a specific property value.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        properties_names: list, optional
            List of properties to be included in the analysis. If not specified, all the properties are included. (default is None)
        properties_values: list of list, optional
            Properties values to be considered in te analysis. If not specified, all the values are considered. (default is None)
            The index of the properties values must be the same of the properties names. Example: properties_names=['name1', 'name2'] properties_values=[['value1_of_name1', 'value2_of_name1'], ['value1_of_name2']]
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            categories = [self.dataset.get_categories_names()[0]]
        elif categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        if not self.dataset.are_valid_categories(categories):
            return -1

        if properties_names is not None:
            if not self.dataset.are_analyses_with_properties_available():
                logger.error("No properties available. Please make sure to load the properties to the dataset.")
                return -1

            if not isinstance(properties_names, list):
                logger.error(err_type.format("properties_names"))
                return -1
            if not self.dataset.are_valid_properties(properties_names):
                return -1
            if properties_values is not None:
                if not isinstance(properties_values, list) or not all(isinstance(item, list) for item in properties_values):
                    logger.error(err_type.format("properties_values"))
                    return -1
                if len(properties_names) != len(properties_values):
                    logger.error("Inconsistency between properties_names and properties_values")
                    return -1
                for i, p in enumerate(properties_names):
                    if not self.dataset.is_valid_property(p, properties_values[i]):
                        return -1
        elif properties_values is not None:
            logger.error("properties_names not specified. Please be sure to specify the properties_names in order to "
                         "apply the properties_values filter")
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format(bool))
            return -1

        properties_filter = self.__get_properties_filter_from_names_and_values(properties_names, properties_values)

        if "all" not in self.matching_dict:
            self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
        matching = self.matching_dict["all"].copy()

        y_true, y_pred, labels, cat_ids = self.__convert_input_confusion_matrix(matching, categories, properties_filter)
        if len(y_true) == 0:
            logger.warning("No observations found")
            return
        result_ml = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        cats_match_label = self.dataset.get_categories_names_from_ids(cat_ids)
        cats_labels = [self.dataset.get_display_name_of_category(cat) for cat in cats_match_label]
        if show:
            display_confusion_matrix(result_ml, cats_labels, properties_filter, self._SAVE_PNG_GRAPHS,
                                     self.result_saving_path)

        result_sl = None
        if self.dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL and len(categories) > 1:
            if len(cats_labels) < len(self.dataset.get_categories_names()):
                labels.append(-1)
                cats_labels.append('other')
            result_sl = confusion_matrix(y_true, y_pred, labels=labels)
            if show:
                display_confusion_matrix_categories(result_sl, cats_labels, properties_filter, self._SAVE_PNG_GRAPHS,
                                                    self.result_saving_path)
        if not show:
            if result_sl is None:
                return result_ml
            return result_ml, result_sl

    def analyze_top1_top5_error(self, properties=None, metric=Metrics.ERROR_RATE, show=True):
        """
        It analyzes the model performances by considering the top-1 and the top-5 classification predictions. The analysis can be performed for the entire data set or for a subset with a specific property value.

        Parameters
        ----------
        properties: list, optional
            If not specified, it performs the analysis on the entire data set, otherwise it performs a per-property analysis. (default is None)
        metric: Metrics, optional
            Evaluation metric used for the analysis. (default is Metrics.ERROR_RATE)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.dataset.task_type != TaskType.CLASSIFICATION_SINGLE_LABEL:
            logger.error(f"Analysis not supported for task type: {self.dataset.task_type}")
            return -1

        if not isinstance(metric, Metrics):
            logger.error(err_type.format("metric"))
            return -1
        if metric not in [Metrics.ERROR_RATE, Metrics.ACCURACY]:
            logger.error(f"Analysis not supported for metric: {metric}")
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        title = "Error Rate" if metric == Metrics.ERROR_RATE else "Accuracy"
        if properties is None:
            observations = self.dataset.get_all_observations().copy()
            proposals = self.dataset.get_proposals(self._model_name).copy()
            results = list(self.__calculate_top1_top5_error(observations, proposals, metric))

            if not show:
                return results

            display_top1_top5_error([results], ["Dataset"], f"{title} analysis", title, self.result_saving_path, self._SAVE_PNG_GRAPHS)

        else:
            if not self.dataset.are_analyses_with_properties_available():
                logger.error("No properties available. Please make sure to load the properties to the dataset.")
                return -1

            if not isinstance(properties, list):
                logger.error(err_type.format("properties"))
                return -1
            if not self.dataset.are_valid_properties(properties):
                return -1

            proposals = self.dataset.get_proposals(self._model_name)
            all_results = {}
            for p in properties:
                all_results[p] = {}
                p_values = self.dataset.get_values_for_property(p)
                p_name = self.dataset.get_display_name_of_property(p)
                results = []
                p_value_names = []
                for p_value in p_values:
                    observations = self.dataset.get_observations_from_property(p, p_value).copy()
                    res = list(self.__calculate_top1_top5_error(observations, proposals.copy(), metric))
                    results.append(res)
                    p_value_names.append(self.dataset.get_display_name_of_property_value(p, p_value))
                    all_results[p][p_value] = res
                if show:
                    display_top1_top5_error(results, p_value_names, f"{title} analysis for the property: {p_name}", title, self.result_saving_path, self._SAVE_PNG_GRAPHS)

            if not show:
                return all_results

    def base_report(self, metrics=None, categories=None, properties=None, show_categories=True, show_properties=True, include_reliability=True):
        """
        It summarizes all the performance scores of the model at all levels of granularity: it provides the overall scores, the per-category scores and the per-property scores.

        Parameters
        ----------
        metrics: list, optional
            List of evaluation metrics to be included in the analysis. If not specified, all the evaluation metrics are included. (default is None)
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        properties: list, optional
            List of properties to be included in the analysis. If not specified, all the properties are included. (default is None)
        show_categories: bool, optional
            Indicates whether the categories should be included in the report. (default is True)
        show_properties: bool, optional
            Indicates whether the properties should be included in the report. (default is True)
        include_reliability: bool, optional
            Indicates whether the 'ece' and 'mce' should be included in the report. (default is True)
        Returns
        -------
        pandas.DataFrame
        """
        if show_properties and not self.dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset or set 'show_properties=False'")
            return -1
        default_metrics = [Metrics.ACCURACY, Metrics.ERROR_RATE, Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE,
                           Metrics.F1_SCORE, Metrics.AVERAGE_PRECISION_SCORE]
        return self._get_report_results(default_metrics, metrics, categories, properties, show_categories,
                                        show_properties, include_reliability)

    def show_true_negative_distribution(self, categories=None, show=True):
        """
        It provides the true negative distribution among the categories.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            logger.error("Not supported for binary classification")
            return -1

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        tn_classes, _ = self._analyze_true_negative_for_categories(categories)

        if not show:
            return tn_classes

        labels = [self.dataset.get_display_name_of_category(cat) for cat in tn_classes.keys()]
        plot_class_distribution(tn_classes, labels, self.result_saving_path, self._SAVE_PNG_GRAPHS, "True Negative distribution")

    def show_true_negative_distribution_for_categories_for_property(self, property_name, property_values=None, categories=None, show=True):
        """
        It provides the true negative distribution of the property values for each category.

        Parameters
        ----------
        property_name: str
            Name of the property to be analyzed.
        property_values: str or Number
            List of the property values to be included in the analysis. If not specified, all the values are included. (default is None)
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not self.dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset.")
            return -1

        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        elif not self.dataset.are_valid_properties([property_name]):
            return -1

        if property_values is None:
            property_values = self.dataset.get_values_for_property(property_name)
        elif not isinstance(property_values, list):
            logger.error(err_type.format("property_values"))
            return -1
        elif not self.dataset.is_valid_property(property_name, property_values):
            return -1

        if categories is None:
            categories = [self.dataset.get_category_name_from_id(1)] if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY else self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type, "categories")
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        results = {}
        p_label = self.dataset.get_display_name_of_property(property_name)
        labels = [self.dataset.get_display_name_of_property_value(property_name, p_v) for p_v in property_values]
        for c in categories:
            fp_p, _ = self._analyze_true_negative_for_category_for_property(c, property_name, property_values)
            results[c] = fp_p

            if show:
                c_label = self.dataset.get_display_name_of_category(c)
                plot_class_distribution(fp_p, labels, self.result_saving_path, self._SAVE_PNG_GRAPHS, f"True Negative distribution of {p_label} for {c_label}")

        if not show:
            return results

    def show_false_positive_distribution_for_categories_for_property(self, property_name, property_values=None, categories=None, show=True):
        """
        It provides the false positive distribution of the property values for each category.

        Parameters
        ----------
        property_name: str
            Name of the property to be analyzed.
        property_values: str or Number
            List of the property values to be included in the analysis. If not specified, all the values are included. (default is None)
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if not self.dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset.")
            return -1

        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        elif not self.dataset.are_valid_properties([property_name]):
            logger.error(err_value.format("property_name", list(self.dataset.get_property_keys())))
            return -1

        if property_values is None:
            property_values = self.dataset.get_values_for_property(property_name)
        elif not isinstance(property_values, list):
            logger.error(err_type.format("property_values"))
            return -1
        elif not self.dataset.is_valid_property(property_name, property_values):
            return -1

        if categories is None:
            categories = [self.dataset.get_category_name_from_id(1)] if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY else self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type, "categories")
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        results = {}
        p_label = self.dataset.get_display_name_of_property(property_name)
        labels = [self.dataset.get_display_name_of_property_value(property_name, p_v) for p_v in property_values]
        for c in categories:
            fp_p, _ = self._analyze_false_positive_for_category_for_property(c, property_name, property_values)
            results[c] = fp_p

            if show:
                c_label = self.dataset.get_display_name_of_category(c)
                plot_class_distribution(fp_p, labels, self.result_saving_path, self._SAVE_PNG_GRAPHS, f"False Positive distribution of {p_label} for {c_label}")

        if not show:
            return results

    def get_true_negative_ids(self, categories):
        """
        Returns the gt and proposals ids of the true negative predictions
        Parameters
        ----------
        categories: list
            Categories to include

        Returns
        -------
            dict
        """
        if not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1

        _, ids = self._analyze_true_negative_for_categories(categories)
        return ids

    # --- EVALUATION METRICS --- #

    def _compute_metric(self, gt, detections, metric, matching, is_micro_required=False):
        """
        Method used to call the metric that is used to calculate the performances

        Parameters
        ----------
        gt: array-like
            Ground Truth
        detections: array-like
            Detections
        metric: Metrics
            Metric selected for the computation
        matching:
            Ground Truth and Proposals matching. In classification is always None
        is_micro_required: bool, optional
            If True it is not a single class analysis
        Returns
        -------
        dict containing the metric value and the standard error
        """
        if metric == Metrics.ACCURACY:
            return self._compute_metric_accuracy(gt, detections)
        elif metric == Metrics.ERROR_RATE:
            return self._compute_metric_error_rate(gt, detections)
        elif metric == Metrics.PRECISION_SCORE:
            return self._compute_metric_precision_score(gt, detections, matching)
        elif metric == Metrics.RECALL_SCORE:
            return self._compute_metric_recall_score(gt, detections, matching)
        elif metric == Metrics.F1_SCORE:
            return self._compute_metric_f1_score(gt, detections, matching)
        elif metric == Metrics.ROC_AUC:
            return self._compute_roc_auc_curve(gt, detections)
        elif metric == Metrics.PRECISION_RECALL_AUC:
            return self._compute_precision_recall_auc_curve(gt, detections, matching)
        elif metric == Metrics.F1_AUC:
            return self._compute_f1_auc_curve(gt, detections, matching)
        elif metric == Metrics.AVERAGE_PRECISION_SCORE:
            return self._compute_average_precision_score(gt, detections, matching)
        else:
            custom_metrics = self.get_custom_metrics()
            if metric not in custom_metrics:
                raise NotImplementedError(f"Not supported metric : {metric}")
            return custom_metrics[metric].evaluate_metric(gt, detections, matching, is_micro_required)

    def _compute_metric_error_rate(self, gt, detections):
        """
        Calculates the error rate

        Parameters
        ----------
        gt: array-like
            Ground Truth
        detections: array-like
            Predictions scores

        Returns
        -------
        error_rate, None (standard error)
        """
        # binary
        if np.array(gt).ndim == 1 and np.array_equal(np.array(gt), np.array(gt).astype(bool)) and \
                np.array(detections).ndim == 1:
            gt_ord, det_ord, tp, _, fp = self._support_metric(gt, detections, None)
            tp, tp_norm, fp, tn = self._support_metric_threshold(sum(np.array(gt_ord) == 1),
                                                                 self._get_normalized_number_of_observations(),
                                                                 gt_ord, det_ord, tp, fp, self._conf_thresh)
            fn = np.sum(np.array(gt_ord) == 1) - tp
            n_tot = len(gt)
            error_rate = (fp + fn) / n_tot if n_tot > 0 else 0
            tn_fp_fn = len(gt) - tp
            error_rate_norm = (fp + fn) / (tp_norm + tn_fp_fn) if tn_fp_fn > 0 else 0

        # single-label and multi-label
        else:
            y_true, y_pred, _, _ = self.__convert_input_ml_sl(gt, detections)
            error_rate = np.mean(np.logical_not(np.logical_and.reduce(np.array(y_true) == np.array(y_pred), axis=-1)))
            error_rate_norm = None

        if np.isnan(error_rate):
            error_rate = 0
            error_rate_norm = 0

        if self._use_normalization:
            return error_rate_norm, None
        else:
            return error_rate, None

    def _compute_metric_accuracy(self, gt, detections):
        """
        Calculates the accuracy score
        Parameters
        ----------
        gt: array-like
            Ground Truth
        detections: array-like
            Predictions scores
        Returns
        -------
        accuracy, None (the standard error)
        """
        np.warnings.filterwarnings('ignore')
        # binary
        if np.array(gt).ndim == 1 and np.array_equal(np.array(gt), np.array(gt).astype(bool)) and \
                np.array(detections).ndim == 1:
            gt_ord, det_ord, tp, _, fp = self._support_metric(gt, detections, None)
            tp, tp_norm, fp, tn = self._support_metric_threshold(sum(np.array(gt_ord) == 1),
                                                                 self._get_normalized_number_of_observations(),
                                                                 gt_ord, det_ord, tp, fp, self._conf_thresh)
            n_tot = len(gt)
            tn_fp_fn = n_tot - tp
            accuracy = (tp + tn) / n_tot if n_tot > 0 else 0
            accuracy_norm = (tp_norm + tn) / (tp_norm + tn_fp_fn) if (tn_fp_fn > 0 or tp_norm > 0) else 0

        # single-label and multi-label
        else:
            y_true, y_pred, _, _ = self.__convert_input_ml_sl(gt, detections)
            accuracy = np.mean(np.logical_and.reduce(np.array(y_true) == np.array(y_pred), axis=-1))
            accuracy_norm = None

        if np.isnan(accuracy):
            accuracy = 0
            accuracy_norm = 0

        if self._use_normalization:
            return accuracy_norm, None
        else:
            return accuracy, None

    def _compute_metric_precision_score(self, gt, detections, matching):
        """
        Calculates the precision score

        Parameters
        ----------
        gt: array-like, or label indicator array
            Ground Truth
        detections: array-like
            Detections
        matching:
            Not used in classification problems

        Returns
        -------
        precision_score, None (the standard error)
        """
        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)
        if is_single_label:
            return self._compute_metric_accuracy(gt, detections)
        tp, tp_norm, fp, _ = self._support_metric_threshold(sum(np.array(gt_ord) == 1),
                                                            self._get_normalized_number_of_observations(),
                                                            gt_ord, det_ord, tp, fp, self._conf_thresh)

        precision, precision_norm = self._support_precision_score(tp, tp_norm, fp)

        if self._use_normalization:
            return precision_norm, None
        else:
            return precision, None

    def _compute_metric_recall_score(self, gt, detections, matching):
        """
        Calculates the recall score

        Parameters
        ----------
        gt: array-like, or label indicator array
            Ground Truth
        detections: array-like
            Detections
        matching:
            Not used in classification problems

        Returns
        -------
        recall_score, None (the standard error)
        """
        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)
        if is_single_label:
            return self._compute_metric_accuracy(gt, detections)

        tp, tp_norm, fp, tn = self._support_metric_threshold(sum(np.array(gt_ord) == 1),
                                                             self._get_normalized_number_of_observations(),
                                                             gt_ord, det_ord, tp, fp, self._conf_thresh)
        fn = np.sum(np.array(gt_ord) == 1) - tp

        recall, recall_norm = self._support_recall_score(tp, tp_norm, fn)
        if self._use_normalization:
            return recall_norm, None
        else:
            return recall, None

    def _compute_metric_f1_score(self, gt, detections, matching):
        """
        Calculates the F1 score

        Parameters
        ----------
        gt: array-like, or label indicator array
            Ground Truth
        detections: array-like
            Detections
        matching:
            Not used in classification problems

        Returns
        -------
        f1_score, None (the standard error)

        """
        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)
        if is_single_label:
            return self._compute_metric_accuracy(gt, detections)

        tp, tp_norm, fp, tn = self._support_metric_threshold(sum(np.array(gt_ord) == 1),
                                                             self._get_normalized_number_of_observations(),
                                                             gt_ord, det_ord, tp, fp, self._conf_thresh)
        fn = np.sum(np.array(gt_ord) == 1) - tp

        f1, f1_norm = self._support_f1_score(tp, tp_norm, fp, fn)

        if self._use_normalization:
            return f1_norm, None
        else:
            return f1, None

    def _compute_average_precision_score(self, gt, detections, matching):
        """
        Calculates the average precision score

        Parameters
        ----------
        gt: array-like
            Ground Truth
        detections: array-like
            Predictions scores
        matching
            Not used in classification problems

        Returns
        -------
        metric_value, std_err
        """
        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)
        if is_single_label:
            logger.warning("Single-label not supported for average_precision_score metric")
            return 0, 0

        metric_value, std_err = self._support_average_precision(len(gt_ord),
                                                                sum(np.array(gt_ord) == 1),
                                                                self._get_normalized_number_of_observations(),
                                                                np.array(det_ord), tp, fp, True)
        return metric_value, std_err

    def _compute_roc_auc_curve(self, gt, detections):
        """
        Calculates the area under the ROC curve
        Parameters
        ----------
        gt:  array-like
            Ground Truth
        detections: array-like
            Detections confidence scores

        Returns
        -------
        auc, standard_error
        """
        fpr, tpr = self._compute_roc_curve(gt, detections)
        std_err = np.std(tpr) / np.sqrt(tpr.size)
        return auc(fpr, tpr), std_err

    def _compute_precision_recall_auc_curve(self, gt, detections, matching):
        """
        Calculates the area under the Precision-Recall curve
        Parameters
        ----------
        gt:  array-like
            Ground Truth
        detections: array-like
            Detections confidence scores

        Returns
        -------
        auc, standard_error

        """
        recall, precision = self._compute_precision_recall_curve(gt, detections)
        std_err = np.std(precision) / np.sqrt(precision.size)
        return auc(recall, precision), std_err

    def _compute_f1_auc_curve(self, gt, detections, matching):
        """
        Calculates the area under the F1 curve

        Parameters
        ----------
        gt:  array-like
            Ground Truth
        detections: array-like
            Detections confidence scores

        Returns
        -------
        auc, standard_error

        """
        thresholds, f1 = self._compute_f1_curve(gt, detections)
        std_err = np.std(f1) / np.sqrt(f1.size)
        return auc(thresholds, f1), std_err

    # -- Evaluation metrics support functions -- #

    def _support_metric(self, gt, detections, matching):
        """
        Sorts by confidence in descending order the detections and the ground truth and calculates the True Positive,
        False Positive and True Negative
        Parameters
        ----------
        gt: array-like
            Ground Truth
        detections: array-like
            Proposals confidence scores
        matching
            Not used in classification

        Returns
        -------
        gt_ord, det_ord, tp, tn, fp
        """
        si = sorted(range(len(detections)), key=lambda k: detections[k], reverse=True)
        gt_ord, det_ord = [], []
        for i in si:
            gt_ord.append(gt[i])
            det_ord.append(detections[i])

        tp, tn, fp = [], [], []
        for i, pos_class in enumerate(gt_ord):
            # tp
            tp.append(1) if pos_class and det_ord[i] > 0 else tp.append(0)
            # fp
            fp.append(1) if not pos_class and det_ord[i] > 0 else fp.append(0)
            # tn
            tn.append(1) if not pos_class and det_ord[i] == 0 else tn.append(0)

        tp = np.cumsum(tp)
        tn = np.sum(tn)
        fp = np.cumsum(fp)
        return gt_ord, det_ord, tp, tn, fp

    def _support_metric_threshold(self, n_true_gt, n_normalized, gt_ord, det_ord, tp, fp, threshold):
        """
        Calculates the True Positive, True Positive Normalized, False Positive and True Negative based on the threshold

        Parameters
        ----------
        n_true_gt: int
            Number of observations considered
        n_normalized: float
            Number of observations normalized
        gt_ord: array-like
            Ordered ground truth
        det_ord: array-like
            Predictions confidences ordered in descending order
        tp: array-like
            Cumsum of True Positive
        fp: array-like
            Cumsum of False Positive
        threshold: float
            Threshold value

        Returns
        -------
        tp, tp_norm, fp, tn
        """
        det_ord = np.array(det_ord)
        tp[det_ord < threshold] = 0
        fp[det_ord < threshold] = 0

        tp = tp[tp > 0]
        tp = tp[-1] if len(tp) > 0 else 0

        fp = fp[fp > 0]
        fp = fp[-1] if len(fp) > 0 else 0

        det_ord[det_ord < threshold] = 0
        det_ord[det_ord > 0] = 1
        tn = np.sum(np.array(gt_ord) == det_ord) - tp
        tp_norm = tp * n_normalized / n_true_gt if n_true_gt > 0 else 0

        return tp, tp_norm, fp, tn

    def _convert_input_for_micro_evaluation(self, matching):
        """
        Converts the dataframe into two lists for micro evaluation
        Parameters
        ----------
        matching: pandas.Dataframe

        Returns
        -------
        y_true, y_score
        """
        cat_names = self.dataset.get_categories_names()
        cat_ids = self.dataset.get_categories_id_from_names(cat_names)

        if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            y_true, y_score = self.__convert_input_format_for_category(matching, cat_names[0])

        elif self.dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            matching["micro"] = matching.apply(lambda x: self.__match_id_confidence_sl(x["category_id"], x["confidence"],
                                                                                 len(cat_ids), np.array(cat_ids)), axis=1)
            y_true = matching["category"].tolist()
            y_score = matching["micro"].tolist()

        else:
            matching_group = matching.groupby(self.dataset.match_param_props)
            props_categories = matching_group["category_id"].apply(list).reset_index(name="categories_prop")
            props_confidences = matching_group["confidence"].apply(list).reset_index(name="confidences_prop")
            gt_categories = matching_group.first().reset_index()

            df = pd.merge(props_categories, props_confidences, on=self.dataset.match_param_props)
            group_param = "id_x" if self.dataset.match_param_gt == "id" else self.dataset.match_param_gt
            matching = pd.merge(gt_categories, df, how="left", left_on=group_param,
                                right_on=self.dataset.match_param_props)
            matching["micro"] = matching.apply(
                lambda x: self.__match_id_confidence_ml(x["categories_prop"], x["confidences_prop"], len(cat_ids),
                                                        np.array(cat_ids)), axis=1)

            y_true = (gt_categories["categories"].apply(lambda x: [1 if i in x else 0 for i in cat_ids])).tolist()
            y_score = matching["micro"].tolist()

        return y_true, y_score

    def __check_and_parse_input(self, gt, detections):
        """
        Checks the input type (binary, single-label, multi-label) and calculates the True Positive, the True Negative
        and the False Positive

        Parameters
        ----------
        gt:
            Ground Truth
        detections:
            Predictions scores

        Returns
        -------
        gt_ord, det_ord, tp, tn, fp, is_single_label
        """
        is_single_label = False
        # binary
        if np.array(gt).ndim == 1 and np.array_equal(np.array(gt), np.array(gt).astype(bool)) and \
                np.array(detections).ndim == 1:
            gt_ord, det_ord, tp, tn, fp = self._support_metric(gt, detections, None)
        # single-label
        elif np.array(gt).ndim == 1:
            is_single_label = True
            return None, None, None, None, None, is_single_label
        # multi-label
        else:
            _, _, y_true_all, y_score_all = self.__convert_input_ml_sl(gt, detections)
            gt_ord, det_ord, tp, tn, fp = self._support_metric(y_true_all, y_score_all, None)
        return gt_ord, det_ord, tp, tn, fp, is_single_label

    def __convert_input_ml_sl(self, gt, detections):
        """
        Converts the input for single-label and multi-label problems
        Parameters
        ----------
        gt:
            Single-label: array-like
            Multi-label: label indicator array
        detections
            Single-label: 2D array-like with prediction scores for each sample
            Multi-label: 2D array with prediction scores for each sample

        Returns
        -------
        y_true: 2d array with label indicator for each sample ([1, 1, 0, 0], [0, 1, 0, 1], ...)
        y_pred: 2d array with label indicator for each sample based on the threshold ([1, 1, 0, 0], [0, 1, 0, 1], ...)
        y_true_all: list of all the categories samples (binary)
        y_score_all: list of all samples scores
        """
        categories_index = np.array(self.dataset.get_categories_id_from_names(self.dataset.get_categories_names()))
        n_categories = categories_index.size
        y_true, y_true_all, y_pred, y_score_all = [], [], [], []
        for i, v in enumerate(gt):
            if type(v) == int:
                im_true = np.zeros(n_categories)
                im_true[np.where(categories_index == v)] = 1
                y_true.append(im_true)
            else:
                y_true.append(v)
                y_true_all.extend(v)
                y_score_all.extend(detections[i])
            im_score = np.array(detections[i])
            im_score[im_score < self._conf_thresh] = 0
            im_score[im_score > 0] = 1
            y_pred.append(list(im_score))
        return y_true, y_pred, y_true_all, y_score_all

    def __support_roc_auc(self, n_true_imgs, n_normalized, det_ord, tp, tn, fp):
        """
        Calculates the False Positive Rate and the True Positive Rate in order to plot the ROC curve

        Parameters
        ----------
        n_true_imgs: int
            Number of positive observations considered
        n_normalized: float
            Normalized number of observations
        det_ord: array-like
            Detections ordered by confidence in descending order
        tp: array-like
            Cumsum of True Positive
        tn: int
            Number of True Negative
        fp: array-like
            Cumsum of False Positive

        Returns
        -------
        fpr, tpr, tpr_norm
        """
        fn = n_true_imgs - tp[-1]
        tp_norm = np.multiply(tp, n_normalized) / n_true_imgs
        tpr = np.true_divide(tp, n_true_imgs)
        tpr_norm = np.true_divide(tp_norm, tp_norm + fn)

        thresholds = np.unique(det_ord)
        rel_indexes = []
        for t in thresholds:
            indexes = np.where(np.array(det_ord) == t)[0]
            for i in indexes:
                fp[i] = fp[indexes[-1]]
                tpr[i] = tpr[indexes[-1]]
                tpr_norm[i] = tpr_norm[indexes[-1]]
            rel_indexes.append(indexes[0])
        tn_tmp = -(fp - fp[-1])
        tn = np.add(tn_tmp, tn)

        fpr = np.divide(fp, np.add(fp, tn))
        np.nan_to_num(fpr, copy=False)
        np.nan_to_num(tpr, copy=False)
        np.nan_to_num(tpr_norm, copy=False)


        one = np.ones(1)
        zero = np.zeros(1)
        rel_indexes = np.sort(rel_indexes)
        # add (0, 0) and (1, 1)
        tpr = np.concatenate([zero, tpr[rel_indexes], one])
        tpr_norm = np.concatenate([zero, tpr_norm[rel_indexes], one])
        fpr = np.concatenate([zero, fpr[rel_indexes], one])

        indexes = []
        v_fpr = -1
        v_tpr = -1

        for i, (f, t) in enumerate(zip(fpr, tpr)):
            if f == v_fpr and t == v_tpr:
                continue
            v_fpr = f
            v_tpr = t
            indexes.append(i)

        fpr = fpr[indexes]
        tpr = tpr[indexes]
        tpr_norm = tpr_norm[indexes]

        return fpr, tpr, tpr_norm

    # --- CURVES --- #

    def _compute_curve_for_categories(self, categories, curve):
        """
        Calculates the x-values, y-values and the area under the curve of the specified curve for each category

        Parameters
        ----------
        categories: array-like
            Categories to consider
        curve: Curves
            Curve to consider

        Returns
        -------
        dict containing the values for each category
        """
        if curve.value not in self.saved_analyses:
            self.saved_analyses[curve.value] = {}

        if "all" not in self.matching_dict:
            self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
        matching = self.matching_dict["all"].copy()
        results = {}
        self._set_normalized_number_of_observations_for_categories()
        for category in categories:

            if category not in self.saved_analyses[curve.value]:
                y_true, y_score = self.__convert_input_format_for_category(matching, category)
                x_values, y_values = self._compute_curve(y_true, y_score, curve)

                auc_value = auc(x_values, y_values)
                self.saved_analyses[curve.value][category] = {'auc': auc_value,
                                                              'x': x_values,
                                                              'y': y_values}

            results[category] = self.saved_analyses[curve.value][category]
        return results

    def _compute_curve_overall(self, curve):
        """
        Computes the overall performance curve
        Parameters
        ----------
        curve: Curves
            Curve used for the analysis

        Returns
        -------
            dict
        """

        if curve.value not in self.saved_analyses:
            self.saved_analyses[curve.value] = {}

        if "overall" not in self.saved_analyses[curve.value]:

            if "all" not in self.matching_dict:
                self.matching_dict["all"] = self._match_classification_with_all_ground_truth()

            matching = self.matching_dict["all"].copy()

            user_normalization = self._use_normalization
            self._use_normalization = False
            y_true, y_score = self._convert_input_for_micro_evaluation(matching)
            x_values, y_values = self._compute_curve(y_true, y_score, curve)

            self._use_normalization = user_normalization
            auc_value = auc(x_values, y_values)
            self.saved_analyses[curve.value]["overall"] = {'overall': {'auc': auc_value,
                                                                       'x': x_values,
                                                                       'y': y_values}}
        results = self.saved_analyses[curve.value]["overall"]
        return results

    def _compute_curve(self, gt, proposals, curve):
        """
        Computes the selected curve values
        Parameters
        ----------
        gt: pandas.DataFrame
            Ground truth annotations
        proposals: pandas.DataFrame
            Predictions dataframe
        curve: Curves
            Curve used for the computation

        Returns
        -------
            x_values, y_values
        """
        if curve == Curves.PRECISION_RECALL_CURVE:
            return self._compute_precision_recall_curve(gt, proposals)
        elif curve == Curves.ROC_CURVE:
            return self._compute_roc_curve(gt, proposals)
        elif curve == Curves.F1_CURVE:
            return self._compute_f1_curve(gt, proposals)
        else:
            raise ValueError("Invalid curve name.")

    def _compute_roc_curve(self, gt, detections):
        """
        Calculates the False Positive Rate (x-values) and True Positive Rate (y-values) of the ROC curve
        Parameters
        ----------
        gt: array-like
            Ground Truth
        detections: array-like
            Detections confidence scores

        Returns
        -------
        fpr, tpr
        """

        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)

        if is_single_label:
            raise ValueError("Not supported for single-label task.")

        fpr, tpr, tpr_norm = self.__support_roc_auc(sum(np.array(gt_ord) == 1),
                                                    self._get_normalized_number_of_observations(),
                                                    det_ord, tp, tn, fp)
        if self._use_normalization:
            return fpr, tpr_norm
        else:
            return fpr, tpr

    def _compute_precision_recall_curve(self, gt, detections):
        """
        Calculates the Recall (x-values) and Precision (y-values) of the Precision-Recall curve

        Parameters
        ----------
        gt:  array-like
            Ground Truth
        detections: array-like
            Detections confidence scores

        Returns
        -------
        recall, precision

        """

        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)

        if is_single_label:
            raise ValueError("Not supported for single-label task.")

        precision, precision_norm, recall, recall_norm = self._support_precision_recall_auc(
            len(gt_ord), sum(np.array(gt_ord) == 1), self._get_normalized_number_of_observations(),
            np.array(det_ord), tp, fp, True)
        if self._use_normalization:
            return recall_norm, precision_norm
        else:
            return recall, precision

    def _compute_f1_curve(self, gt, detections):
        """
        Calculates the F1 scores for different thresholds

        gt:  array-like
            Ground Truth
        detections: array-like
            Detections confidence scores

        Returns
        -------
        thresholds, f1_scores

        """

        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)

        if is_single_label:
            raise ValueError("Not supported for single-label task.")

        precision, precision_norm, recall, recall_norm, rel_indexes = self._support_precision_recall(
            sum(np.array(gt_ord) == 1), self._get_normalized_number_of_observations(), np.array(det_ord), tp, fp)

        return self._support_f1_curve(np.array(det_ord), precision, precision_norm, recall, recall_norm, rel_indexes)

    # -- Per-category and per-property evaluation support functions -- #

    def _calculate_metric_for_category(self, category, metric):
        """
        Calculates the metric for a specific category
        Parameters
        ----------
        category: str
            Category considered
        metric: Metrics
            Metric used

        Returns
        -------
        dict containing the metric value and the standard error
        """

        if metric in self.saved_results and category in self.saved_results[metric] and 'all' in self.saved_results[metric][category]:
            return self.saved_results[metric][category]["all"]

        if metric not in self.saved_results:
            self.saved_results[metric] = {}
        if category not in self.saved_results[metric]:
            self.saved_results[metric][category] = {}

        if "all" not in self.matching_dict:
            self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
        matching = self.matching_dict["all"].copy()

        y_true, y_score = self.__convert_input_format_for_category(matching, category)
        self._set_normalized_number_of_observations_for_categories()
        result, std_err = self._compute_metric(y_true, y_score, metric, None)

        value = {"value": result, "std": std_err, "matching": None}
        self.saved_results[metric][category]["all"] = value

        return value

    def _calculate_metric_for_properties_of_category(self, category_name, category_id, property_name,
                                                     possible_values,
                                                     matching, metric):
        """
        Calculates the metric for a specific category and per-property value

        Parameters
        ----------
        category_name: str
            Name of the category to consider
        category_id: int
            Id of the category to consider
        property_name:
            Property to consider in the analysis
        possible_values:
            Property values that are used to filter the dataset
        matching:
            Not used in classifications problems
        metric: Metrics
            Metric used

        Returns
        -------
        dict containing the metric value and the standard error
        """

        if metric not in self.saved_results:
            self.saved_results[metric] = {}
        if category_name not in self.saved_results[metric]:
            self.saved_results[metric][category_name] = {}
        if property_name not in self.saved_results[metric][category_name]:
            self.saved_results[metric][category_name][property_name] = {}

        properties_results = {}

        if self._use_new_normalization:
            self._set_normalized_number_of_observations_for_property_for_categories(property_name)
        else:
            self._set_normalized_number_of_observations_for_categories()

        for value in possible_values:
            if value in self.saved_results[metric][category_name][property_name].keys():
                result = self.saved_results[metric][category_name][property_name][value]
            else:
                if property_name not in self.matching_dict:
                    self.matching_dict[property_name] = {}
                if value not in self.matching_dict[property_name]:
                    self.matching_dict[property_name][value] = self._match_classification_with_ground_truth_for_property_value(property_name, value)
                matching = self.matching_dict[property_name][value]
                y_true, y_score = self.__convert_input_format_for_category(matching, category_name)
                metricvalue, std_err = self._compute_metric(y_true, y_score, metric, None)
                if math.isnan(metricvalue):
                    metricvalue = 0
                result = {"value": metricvalue, "std": std_err}

                self.saved_results[metric][category_name][property_name][value] = result

            properties_results[value] = result
        return properties_results

    def __convert_input_format_for_category(self, matching, category):
        """
        Matches the proposals and the observations of a specific category

        Parameters
        ----------
        matching: pandas.DataFrame
        category: str

        Returns
        -------
        y_true, y_scores: y_true is a binary array indicating the presence of the category or not, y_scores is an array
        indicating the confidence of the prediction corresponding to the y_true
        """
        cat_id = self.dataset.get_category_id_from_name(category)

        matching["cat_conf"] = np.where(matching['category_id'] == cat_id, matching['confidence'], 0)

        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            matching = matching.assign(binary=np.where(matching['categories'].apply(lambda x: cat_id in x), 1, 0))
            group_param = "id_x" if self.dataset.match_param_gt == "id" else self.dataset.match_param_gt
            matching = matching.sort_values(by="cat_conf", ascending=False).groupby(
                group_param).first()
        else:
            matching = matching.assign(binary=np.where(matching['category'] == cat_id, 1, 0))
        y_true = matching["binary"].tolist()
        y_scores = matching["cat_conf"].tolist()

        return y_true, y_scores

    # -- Distribution support functions -- #

    def _analyze_false_negative_for_categories(self, categories):
        """
        Computes the False Negative for each category

        Parameters
        ----------
        categories: array-like
            Categories considered in the analysis

        Returns
        -------
        dict containing the number of False Negative for each class
        dict containing the ids of the False Negative for each class
        """
        if "fn" not in self.saved_analyses:
            self.saved_analyses["fn"] = {"classes": {},
                                         "ids": {}}
        classes, ids = {}, {}
        for category in categories:
            if category not in self.saved_analyses["fn"]["classes"] or category not in self.saved_analyses["fn"]["ids"]:
                cat_id = self.dataset.get_category_id_from_name(category)

                if "all" not in self.matching_dict:
                    self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
                matching = self.matching_dict["all"].copy()

                fn_matching = self.__get_fn_dataframe_for_category_from_matching(matching, cat_id)

                self.saved_analyses["fn"]["ids"][category] = {
                    "gt": fn_matching["id_x"].tolist() if not fn_matching.empty else [],
                    "props": []}
                self.saved_analyses["fn"]["classes"][category] = len(fn_matching.index)

            classes[category] = self.saved_analyses["fn"]["classes"][category]
            ids[category] = self.saved_analyses["fn"]["ids"][category]

        return classes, ids

    def _analyze_true_positive_for_categories(self, categories):
        """
        Computes the True Positive for each category

        Parameters
        ----------
        categories: array-like
            Categories considered in the analysis

        Returns
        -------
        dict containing the number of True Positive for each class
        dict containing the ids of the True Positive for each class
        """

        if "tp" not in self.saved_analyses:
            self.saved_analyses["tp"] = {"classes": {},
                                         "ids": {}}

        classes, ids = {}, {}
        for category in categories:

            if category not in self.saved_analyses["tp"]["classes"] or category not in self.saved_analyses["tp"]["ids"]:
                cat_id = self.dataset.get_category_id_from_name(category)

                if "all" not in self.matching_dict:
                    self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
                matching = self.matching_dict["all"].copy()

                matching = self.__get_tp_dataframe_for_category_from_matching(matching, cat_id)

                self.saved_analyses["tp"]["classes"][category] = len(matching.index)
                self.saved_analyses["tp"]["ids"][category] = {"gt": matching["id_x"].tolist() if not matching.empty else [],
                                    "props": matching["id_y"].tolist() if not matching.empty else []}
            classes[category] = self.saved_analyses["tp"]["classes"][category]
            ids[category] = self.saved_analyses["tp"]["ids"][category]
        return classes, ids

    def _analyze_true_negative_for_categories(self, categories):
        """
        Computes the True Negative for each category

        Parameters
        ----------
        categories: array-like
            Categories considered in the analysis

        Returns
        -------
        dict containing the number of True Negative for each class
        dict containing the ids of the True Negative for each class

        """
        if "tn" not in self.saved_analyses:
            self.saved_analyses["tn"] = {"classes": {},
                                         "ids": {}}

        classes, ids = {}, {}
        for category in categories:
            if category not in self.saved_analyses["tn"]["classes"] or category not in self.saved_analyses["tn"]["ids"]:
                cat_id = self.dataset.get_category_id_from_name(category)

                if "all" not in self.matching_dict:
                    self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
                matching = self.matching_dict["all"].copy()

                tn_matching = self.__get_tn_dataframe_for_category_from_matching(matching, cat_id)

                self.saved_analyses["tn"]["classes"][category] = len(tn_matching.index)
                self.saved_analyses["tn"]["ids"][category] = {"gt": tn_matching["id_x"].tolist() if not tn_matching.empty else [],
                                                              "props": []}

            classes[category] = self.saved_analyses["tn"]["classes"][category]
            ids[category] = self.saved_analyses["tn"]["ids"][category]
        return classes, ids

    def _analyze_false_positive_for_categories(self, categories):
        """
        Computes the False Positive for each category

        Parameters
        ----------
        categories: array-like
            Categories considered in the analysis

        Returns
        -------
        dict containing the number of False Positive for each class
        dict containing the ids of the False Positive for each class

        """
        if "fp" not in self.saved_analyses:
            self.saved_analyses["fp"] = {"classes": {},
                                         "ids": {}}

        classes, ids = {}, {}
        for category in categories:
            if category not in self.saved_analyses["fp"]["classes"] or category not in self.saved_analyses["fp"]["ids"]:
                cat_id = self.dataset.get_category_id_from_name(category)

                if "all" not in self.matching_dict:
                    self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
                matching = self.matching_dict["all"].copy()

                fp_matching = self.__get_fp_dataframe_for_category_from_matching(matching, cat_id, self._conf_thresh)
                self.saved_analyses["fp"]["classes"][category] = len(fp_matching.index)
                self.saved_analyses["fp"]["ids"][category] = {"gt": fp_matching["id_x"].tolist() if not fp_matching.empty else [],
                                                              "props": fp_matching["id_y"].tolist() if not fp_matching.empty else []}

            classes[category] = self.saved_analyses["fp"]["classes"][category]
            ids[category] = self.saved_analyses["fp"]["ids"][category]

        return classes, ids

    def _analyze_false_positive_errors(self, categories, threshold):
        """
        Analyzes the False Positive for each category by dividing the errors in three tags: similar (if the error is
        due to a similarity with another category), no_ground_truth (if the observation considered hasn't a category),
        other.
        Parameters
        ----------
        categories: list
            Categories to be included.
        threshold: float
            Confidence threshold value.

        Returns
        -------
        dict,
        dict containing the ids of the False Positive for each class
        """

        if "false_positive_errors" not in self.saved_analyses:
            self.saved_analyses["false_positive_errors"] = {}
        if str(threshold) not in self.saved_analyses["false_positive_errors"]:
            self.saved_analyses["false_positive_errors"][str(threshold)] = {"errors": {},
                                                                            "ids": {}}

        fp_errors, fp_ids = {}, {}

        for category in categories:

            if category not in self.saved_analyses["false_positive_errors"][str(threshold)]["errors"]:

                cat_id = self.dataset.get_category_id_from_name(category)

                if "all" not in self.matching_dict:
                    self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
                matching = self.matching_dict["all"].copy()

                similar_indexes, without_gt_indexes, other_indexes = [], [], []
                fp_ids_cat = {}

                matching = self.__get_fp_dataframe_for_category_from_matching(matching, cat_id, threshold)

                if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
                    match_no_gt = matching.loc[matching["categories"].str.len() == 0]
                    if not match_no_gt.empty:
                        without_gt_indexes = match_no_gt["id_y"].tolist()
                        fp_ids_cat["background"] = {"gt": match_no_gt["id_x"].tolist(),
                                                          "props": match_no_gt["id_y"].tolist()}
                    matching = matching.loc[matching["categories"].str.len() != 0]
                    matching["similar"] = np.where(matching["categories"].apply(
                        lambda x: self.dataset.is_similar(cat_id, x)), 1, 0)

                else:
                    matching["similar"] = np.where(matching["category"].apply(
                        lambda x: self.dataset.is_similar(cat_id, x)), 1, 0)

                fp_ids_cat["similar"] = {"gt": matching.loc[matching["similar"] == 1]["id_x"].tolist(),
                                               "props": matching.loc[matching["similar"] == 1]["id_y"].tolist()}
                fp_ids_cat["other"] = {"gt": matching[matching["similar"] == 0]["id_x"].tolist(),
                                             "props": matching.loc[matching["similar"] == 0]["id_y"].tolist()}
                similar_indexes = matching[matching["similar"] == 1]["id_y"].tolist()
                other_indexes = matching[matching["similar"] == 0]["id_y"].tolist()

                self.saved_analyses["false_positive_errors"][str(threshold)]["errors"][category] = {"similar": similar_indexes,
                                                                                     "background": without_gt_indexes,
                                                                                     "other": other_indexes}
                self.saved_analyses["false_positive_errors"][str(threshold)]["ids"][category] = fp_ids_cat

            fp_errors[category] = self.saved_analyses["false_positive_errors"][str(threshold)]["errors"][category]
            fp_ids[category] = self.saved_analyses["false_positive_errors"][str(threshold)]["ids"][category]

        return fp_errors, fp_ids

    def __get_tp_dataframe_for_category_from_matching(self, matching, cat_id):
        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            matching = matching.loc[(matching["category_id"] == cat_id) &
                                    (matching["confidence"] >= self._conf_thresh) &
                                    (matching["categories"].apply(lambda x: cat_id in x))]

        else:
            matching = matching[(matching["category_id"] == cat_id) &
                                (matching["confidence"] >= self._conf_thresh) &
                                (matching["category"] == cat_id)]
        return matching

    def __get_fp_dataframe_for_category_from_matching(self, matching, cat_id, threshold):
        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            group_param_gt = "id_x" if self.dataset.match_param_gt == "id" else self.dataset.match_param_gt
            matching = matching[(matching["category_id"] == cat_id) &
                                (matching["confidence"] >= threshold) &
                                (matching["categories"].apply(lambda x: cat_id not in x))].drop_duplicates(
                group_param_gt)
        else:
            matching = matching[(matching["category_id"] == cat_id) &
                                (matching["confidence"] >= threshold) &
                                (matching["category"] != cat_id)]
        return matching

    def __get_fn_dataframe_for_category_from_matching(self, matching, cat_id,):
        category_predictions = matching.loc[
            (matching["category_id"] == cat_id) & (matching["confidence"] >= self._conf_thresh)]
        group_param_gt = "id_x" if self.dataset.match_param_gt == "id" else self.dataset.match_param_gt

        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            matching = matching.loc[
                (~matching[group_param_gt].isin(category_predictions[self.dataset.match_param_props])) &
                (matching["categories"].apply(lambda x: cat_id in x))].drop_duplicates(group_param_gt)
        else:
            matching = matching.loc[
                (~matching[group_param_gt].isin(category_predictions[self.dataset.match_param_props])) &
                (matching["category"] == cat_id)]
        return matching

    def __get_tn_dataframe_for_category_from_matching(self, matching, cat_id):
        category_predictions = matching.loc[
            (matching["category_id"] == cat_id) & (matching["confidence"] >= self._conf_thresh)]
        group_param_gt = "id_x" if self.dataset.match_param_gt == "id" else self.dataset.match_param_gt

        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            matching = matching.loc[
                (~matching[group_param_gt].isin(category_predictions[self.dataset.match_param_props])) &
                (matching["categories"].apply(lambda x: cat_id not in x))].drop_duplicates(group_param_gt)
        else:
            matching = matching.loc[
                (~matching[group_param_gt].isin(category_predictions[self.dataset.match_param_props])) &
                (matching["category"] != cat_id)]
        return matching

    def _analyze_true_positive_for_category_for_property(self, category, property_name, property_values):
        """
        Computes the true positive for each category for each property value
        Parameters
        ----------
        category: str
            Category name
        property_name: str
            Property name
        property_values: list
            Property values

        Returns
        -------
        tp_p_values: dict, tp_ids: dict
        """
        tp_p_values = defaultdict(int)
        tp_ids = {}
        cat_id = self.dataset.get_category_id_from_name(category)

        for p_value in property_values:

            if property_name not in self.matching_dict:
                self.matching_dict[property_name] = {}
            if p_value not in self.matching_dict[property_name]:
                self.matching_dict[property_name][p_value] = self._match_classification_with_ground_truth_for_property_value(property_name, p_value)
            matching = self.matching_dict[property_name][p_value].copy()

            matching = self.__get_tp_dataframe_for_category_from_matching(matching, cat_id)

            tp_p_values[p_value] = len(matching.index)
            tp_ids[p_value] = {"gt": matching["id_x"].tolist() if not matching.empty else [],
                               "props": matching["id_y"].tolist() if not matching.empty else []}

        return tp_p_values, tp_ids

    def _analyze_false_negative_for_category_for_property(self, category, property_name, property_values):
        """
        Computes the false negative for each category for each property value
        Parameters
        ----------
        category: str
            Category name
        property_name: str
            Property name
        property_values: list
            Property values

        Returns
        -------
        fn_p_values: dict, fn_ids: dict
        """

        fn_p_values = defaultdict(int)
        fn_ids = {}
        cat_id = self.dataset.get_category_id_from_name(category)

        for p_value in property_values:
            if property_name not in self.matching_dict:
                self.matching_dict[property_name] = {}
            if p_value not in self.matching_dict[property_name]:
                self.matching_dict[property_name][p_value] = self._match_classification_with_ground_truth_for_property_value(property_name, p_value)
            matching = self.matching_dict[property_name][p_value].copy()

            matching = self.__get_fn_dataframe_for_category_from_matching(matching, cat_id)

            fn_p_values[p_value] = len(matching.index)
            fn_ids[p_value] = {"gt": matching["id_x"].tolist() if not matching.empty else [],
                               "props": matching["id_y"].tolist() if not matching.empty else []}

        return fn_p_values, fn_ids

    def _analyze_false_positive_for_category_for_property(self, category, property_name, property_values):
        """
        Computes the false positive for each category for each property value
        Parameters
        ----------
        category: str
            Category name
        property_name: str
            Property name
        property_values: list
            Property values

        Returns
        -------
        fp_p_values: dict, fp_ids: dict
        """

        fp_p_values = defaultdict(int)
        fp_ids = {}
        cat_id = self.dataset.get_category_id_from_name(category)

        for p_value in property_values:
            if property_name not in self.matching_dict:
                self.matching_dict[property_name] = {}
            if p_value not in self.matching_dict[property_name]:
                self.matching_dict[property_name][p_value] = self._match_classification_with_ground_truth_for_property_value(property_name, p_value)
            matching = self.matching_dict[property_name][p_value].copy()

            matching = self.__get_fp_dataframe_for_category_from_matching(matching, cat_id, self._conf_thresh)

            fp_p_values[p_value] = len(matching.index)
            fp_ids[p_value] = {"gt": matching["id_x"].tolist() if not matching.empty else [],
                               "props": matching["id_y"].tolist() if not matching.empty else []}

        return fp_p_values, fp_ids

    def _analyze_true_negative_for_category_for_property(self, category, property_name, property_values):
        """
        Computes the true negative for each category for each property value
        Parameters
        ----------
        category: str
            Category name
        property_name: str
            Property name
        property_values: list
            Property values

        Returns
        -------
        tn_p_values: dict, tn_ids: dict
        """

        tn_p_values = defaultdict(int)
        tn_ids = {}
        cat_id = self.dataset.get_category_id_from_name(category)

        for p_value in property_values:
            if property_name not in self.matching_dict:
                self.matching_dict[property_name] = {}
            if p_value not in self.matching_dict[property_name]:
                self.matching_dict[property_name][p_value] = self._match_classification_with_ground_truth_for_property_value(property_name, p_value)
            matching = self.matching_dict[property_name][p_value].copy()

            matching = self.__get_tn_dataframe_for_category_from_matching(matching, cat_id)

            tn_p_values[p_value] = len(matching.index)
            tn_ids[p_value] = {"gt": matching["id_x"].tolist() if not matching.empty else [],
                               "props": []}

        return tn_p_values, tn_ids

    # -- Reliability support functions -- #

    def _calculate_reliability(self, y_true, y_pred, y_score, num_bins, category=None):
        """
        Calculates the reliability

        Parameters
        ----------
        y_true:  array-like
            Ground Truth
        y_pred:  array-like
            Predictions label
        y_score: array-like
            Predictions scores
        num_bins: int
            Number of bins used to split the confidence values

        Returns
        -------
        dict : {'values': bin_accuracies, 'gaps': gaps, 'counts': bin_counts, 'bins': bins,
                  'avg_value': avg_acc, 'avg_conf': avg_conf, 'ece': ece, 'mce': mce}
        """
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        indices = np.digitize(y_score, bins, right=True)

        bin_accuracies = np.zeros(num_bins, dtype=float)
        bin_confidences = np.zeros(num_bins, dtype=float)
        bin_counts = np.zeros(num_bins, dtype=int)

        np_y_true = np.array(y_true)
        np_y_pred = np.array(y_pred)
        np_y_score = np.array(y_score)

        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(np_y_true[selected] == np_y_pred[selected])
                bin_confidences[b] = np.mean(np_y_score[selected])
                bin_counts[b] = len(selected)

        if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            category = self.dataset.get_categories_names()[0]
            avg_acc = self._calculate_metric_for_category(category, Metrics.ACCURACY)["value"]
        else:
            if category is None:
                accuracies = []
                for c in self.dataset.get_categories_names():
                    accuracies.append(self._calculate_metric_for_category(c, Metrics.ACCURACY)["value"])
                avg_acc = np.mean(accuracies)
            else:
                avg_acc = self._calculate_metric_for_category(category, Metrics.ACCURACY)["value"]

        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)
        if np.isnan(avg_conf):
            avg_conf = 0

        gaps = bin_confidences - bin_accuracies
        ece = np.sum(np.abs(gaps) * bin_counts) / np.sum(bin_counts)
        if np.isnan(ece):
            ece = 0
        mce = np.max(np.abs(gaps))
        if np.isnan(mce):
            mce = 0

        result = {'values': bin_accuracies, 'gaps': gaps, 'counts': bin_counts, 'bins': bins,
                  'avg_value': avg_acc, 'avg_conf': avg_conf, 'ece': ece, 'mce': mce}
        return result

    def __convert_input_reliability(self, matching, category=None):
        """
        Converts the input DataFrames in array-like type for the _calculate_reliability function

        Parameters
        ----------
        matching: pandas.DataFrame
        category: str, optional
            A specific category to consider. If None consider all

        Returns
        -------
        y_true, y_pred, y_score
        """

        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:

            if category is None:
                matching = matching.assign(is_correct=[1 if a in b else 0 for a, b in zip(matching["category_id"],
                                                                                    matching["categories"])])
            else:
                cat_id = self.dataset.get_category_id_from_name(category)
                matching["is_correct"] = np.where((matching["categories"].apply(lambda x: cat_id in x)) &
                                                  (matching["category_id"] == cat_id), 1, 0)
            y_pred = matching["is_correct"].tolist()
            y_true = np.ones(len(y_pred))

        else:
            if category is not None:
                cat_id = self.dataset.get_category_id_from_name(category)
                matching = matching.loc[matching["category_id"] == cat_id]
            y_pred = matching["category_id"].tolist()
            y_true = matching["category"].tolist()

        y_score = matching["confidence"].tolist()

        return y_true, y_pred, y_score

    # -- Confusion matrix support functions -- #

    def __convert_input_confusion_matrix(self, matching, categories, properties_filter):
        """
        Converts the input DataFrames in array-like type for the 'multilabel_confusion_matrix' and the
        'confusion_matrix' functions

        Parameters
        ----------
        matching: pandas.DataFrame
        categories: array-like
            Categories to consider
        properties_filter: dict
            Properties filter dict

        Returns
        -------
        y_true, y_pred, labels, cat_ids
        """

        cat_ids = self.dataset.get_categories_id_from_names(categories)

        if properties_filter is not None:
            observations = self.dataset.get_all_observations().copy()
            for p_name in properties_filter.keys():
                observations = observations.loc[observations.index.get_level_values(p_name).isin(properties_filter[p_name])]
            if observations.empty:
                return [], [], cat_ids
            group_param_gt = "id_x" if self.dataset.match_param_gt == "id" else self.dataset.match_param_gt
            matching = matching.loc[matching[group_param_gt].isin(observations[self.dataset.match_param_gt])]

        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            all_cat_ids = self.dataset.get_categories_id_from_names(self.dataset.get_categories_names())

            proposals = matching.loc[matching["confidence"] >= self._conf_thresh].groupby(self.dataset.match_param_props)["category_id"].apply(list).reset_index(name="categories_prop")
            gt = matching.groupby(self.dataset.match_param_props).first().reset_index()
            group_param_gt = "id_x" if self.dataset.match_param_gt == "id" else self.dataset.match_param_gt
            matching = pd.merge(gt, proposals, how="left", left_on=group_param_gt, right_on=self.dataset.match_param_props)
            no_proposals = np.zeros(len(all_cat_ids)).tolist()
            y_true = gt["categories"].apply(lambda x: [1 if i in x else 0 for i in all_cat_ids]).tolist()
            y_pred = matching["categories_prop"].apply(lambda x: no_proposals if type(x) == float else [1 if i in x else 0 for i in all_cat_ids]).tolist()
            labels = []
            for id in cat_ids:
                labels.append(np.where(np.array(all_cat_ids) == id)[0][0])

        elif self.dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            matching["confusion_id"] = np.where((matching["confidence"] >= self._conf_thresh) &
                                                (matching["category_id"].isin(cat_ids)), matching["category_id"], -1)
            y_true = matching["category"].tolist()
            y_pred = matching["confusion_id"].tolist()
            labels = cat_ids
        else:
            cat_id = cat_ids[0]
            matching[cat_id] = np.where(matching['category'] == cat_id, 1, 0)
            matching["confusion_id"] = np.where(matching["confidence"] >= self._conf_thresh, 1, 0)
            y_true = matching[cat_id].tolist()
            y_pred = matching["confusion_id"].tolist()
            labels = cat_ids

        return y_true, y_pred, labels, cat_ids

    def __get_properties_filter_from_names_and_values(self, properties_names, properties_values):
        """
        Creates a dict containing for each property the corresponding values to consider in a future analysis

        Parameters
        ----------
        properties_names: array-like
            Properties to filter
        properties_values: array-like
            Properties values to filter

        Returns
        -------
        dict
        """
        properties_filter = {}
        if properties_values is None:
            if properties_names is not None:
                for p in properties_names:
                    properties_filter[p] = self.dataset.get_values_for_property(p)
            else:
                properties_filter = None
        else:
            for i, p in enumerate(properties_names):
                properties_filter[p] = properties_values[i]
        return properties_filter

    # -- Base report support functions -- #

    def _get_input_report(self, properties, show_properties_report):
        """
        Creates a dict containing all the ground truth and the corresponding predictions used to calculate the base
        report
        Parameters
        ----------
        properties: array-like
            Properties to consider
        show_properties_report: bool
            If False don't consider the properties

        Returns
        -------
        dict
        """
        input_report = {}

        if "all" not in self.matching_dict:
            self.matching_dict["all"] = self._match_classification_with_all_ground_truth()
        matching = self.matching_dict["all"].copy()

        y_true_micro, y_score_micro = self._convert_input_for_micro_evaluation(matching)

        input_report["total"] = {'all': {"y_true": y_true_micro,
                                         "y_score": y_score_micro}}

        return input_report

    def _calculate_report_for_metric(self, input_report, categories, properties, show_categories, show_properties,
                                     metric):
        """
        Calculates the metric value for the report. The metric is calculated for the entire dataset with the micro and
        macro averaging and then is calculated for each category and for each property value

        Parameters
        ----------
        input_report: dict
            Dict created with the '_get_input_report'
        categories: array-like
            Categories to consider in the report
        properties: array-like
            Properties to consider in the report
        show_categories: bool
            If False don't consider the categories
        show_properties: bool
            If False don't consider the properties
        metric: Metrics
            Metric used in the analysis

        Returns
        -------
        dict containing all the results
        """
        warn_metrics = [Metrics.AVERAGE_PRECISION_SCORE]
        results = {}
        cat_metric_values = {}

        # total
        self._set_normalized_number_of_observations_for_categories()
        user_normalization = self._use_normalization
        if not self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            self._use_normalization = False
        if self.dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL and metric in warn_metrics:
            micro_value = "not supported"
        else:
            micro_value, _ = self._compute_metric(input_report["total"]["all"]["y_true"],
                                                  input_report["total"]["all"]["y_score"],
                                                  metric, None, is_micro_required=True)
        self._use_normalization = user_normalization
        if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            results["total"] = micro_value
        else:
            tot_value = []
            for cat in self.dataset.get_categories_names():
                value = self._calculate_metric_for_category(cat, metric)['value']
                tot_value.append(value)
                cat_metric_values[cat] = value
            results["avg macro"] = mean(tot_value)
            results["avg micro"] = micro_value

        # categories
        if show_categories:
            for cat in categories:
                results[cat] = cat_metric_values[cat]

        # properties
        if show_properties:
            for prop in properties:
                if self._use_new_normalization:
                    self._set_normalized_number_of_observations_for_property_for_categories(prop)
                else:
                    self._set_normalized_number_of_observations_for_categories()
                p_values = self.dataset.get_values_for_property(prop)

                categories = self.dataset.get_categories_names()
                if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
                    categories = [categories[0]]
                for p_value in p_values:
                    tot_value = []
                    for cat in categories:
                        cat_id = self.dataset.get_category_id_from_name(cat)
                        value = self._calculate_metric_for_properties_of_category(cat, cat_id, prop, [p_value], None, metric)[p_value]['value']
                        tot_value.append(value)
                    p_value = prop + "_" + "{}".format(p_value)
                    results[p_value] = mean(tot_value)
        return results

    def __match_id_confidence_sl(self, cat_id, confidence, size, cats_ids):
        """
        Creates an array of size = 'size' and assigns to the 'cat_id' position the confidence value.
        Parameters
        ----------
        cat_id: int
            Category id
        confidence: float
            Confidence value of the prediction that refers to the cat_id
        size: int
            Length of the array
        cats_ids: array-like
            Array containing all the categories ids

        Returns
        -------
        index: array-like
        """
        if np.isnan(cat_id) or np.isnan(confidence):
            return list(np.zeros(size))
        else:
            cat_id = int(cat_id)
            index = np.zeros(size)
            index[np.where(cats_ids == cat_id)] = confidence
            return list(index)

    def __match_id_confidence_ml(self, cat_ids, confidences, size, cats_ids):
        """
        Creates an array of size = 'size' and assigns to each category id position the corresponding confidence value.

        Parameters
        ----------
        cat_ids: array-like
            Category ids
        confidences: array-like
            Confidence values of the predictions that refer to the cat_ids
        size: int
            Length of the array
        cats_ids: array-like
            Array containing all the categories ids

        Returns
        -------
        index: array-like
        """
        if np.any(np.isnan(cats_ids)) or np.any(np.isnan(confidences)):
            return list(np.zeros(size))
        else:
            index = np.zeros(size)
            for i, cat_id in enumerate(cat_ids):
                cat_id = int(cat_id)
                index[np.where(cats_ids == cat_id)] = confidences[i]
            return list(index)

    # -- Top1 top5 support functions -- #

    def __support_top1_top5_error_rate(self, obs_cat_id, props_scores, cats_ids):
        """
        Support function to prepare input for top-1 and top-5 error rate calculation.

        Parameters
        ----------
        obs_cat_id: int
            Single observation category id
        props_scores: array-like
            Scores of all the categories referring to the specific observation. The scores are ordered by category id
        cats_ids: array-like
            List containing all the categories ids

        Returns
        -------
        [top_1, top_5]: list containing 1 or 0. top_1 is 1 if the top-1 classification is correct, otherwise 0.
        top_5 is 1 if the top-5 classification is correct, otherwise 0.
        """
        pos = np.where(cats_ids == obs_cat_id)[0]
        top_1 = 1 if pos == np.argmax(props_scores) else 0
        top_5 = 1 if len(props_scores) < 6 or pos in np.argsort(props_scores)[-5:] else 0
        return [top_1, top_5]

    def __calculate_top1_top5_error(self, observations, proposals, metric):
        """
        Calculates the top-1 and top-5 accuracy/error rate.

        Parameters
        ----------
        observations: pandas.DataFrame
            Ground Truth observations
        proposals: pandas.DataFrame
            Proposals
        metric: Metrics
            Metric used for the evaluation

        Returns
        -------
        top_1_error: float
        top_5_error: float
        """
        proposals = proposals.sort_values(by="category_id", ascending=True)
        proposals = proposals.groupby(self.dataset.match_param_props).agg({'confidence': list}).reset_index()
        matching = pd.merge(observations, proposals, how="left", left_on=self.dataset.match_param_gt,
                            right_on=self.dataset.match_param_props)
        cats_ids = self.dataset.get_categories_id_from_names(self.dataset.get_categories_names())
        matching["top1_top5"] = matching.apply(
            lambda x: self.__support_top1_top5_error_rate(x["category"], x["confidence"], np.array(cats_ids)), axis=1)

        props_top1_top5 = np.array(list(matching["top1_top5"]))
        y_true = np.ones(len(props_top1_top5))
        if metric == Metrics.ERROR_RATE:
            top_1_error, _ = self._compute_metric_error_rate(list(y_true), list(props_top1_top5[:, 0]))
            top_5_error, _ = self._compute_metric_error_rate(list(y_true), list(props_top1_top5[:, 1]))
        else:
            top_1_error, _ = self._compute_metric_accuracy(list(y_true), list(props_top1_top5[:, 0]))
            top_5_error, _ = self._compute_metric_accuracy(list(y_true), list(props_top1_top5[:, 1]))
        return top_1_error, top_5_error

    # -- False negative support function -- #

    def __is_false_negative_ml_similar_classes(self, gt_categories, proposals_categories):
        if not isinstance(proposals_categories, list):
            return False
        for cat_id in gt_categories:
            if cat_id in proposals_categories:
                proposals_categories.remove(cat_id)
                continue
            if self.dataset.is_similar(cat_id, proposals_categories):
                return True
        return False

    # -- Normalization support functions -- #

    def _get_normalized_number_of_observations(self):
        """
        Returns
        -------
        normalized_number_of_observations: float
            Number of observations normalized
        """
        return self.__normalized_number_of_observations

    def _set_normalized_number_of_observations_for_categories(self):
        """
        Normalizes the number of observations considering only the categories
        """
        self.__normalized_number_of_observations = self._norm_factors["categories"] * \
                                                   self.dataset.get_number_of_observations()

    def _set_normalized_number_of_observations_for_property_for_categories(self, property_name):
        """
        Normalizes the number of observations considering the categories and the property

        Parameters
        ----------
        property_name:
            Property to consider in the normalization
        """
        if property_name not in self._norm_factors:
            self.update_property_normalization_factor(property_name)
        self.__normalized_number_of_observations = self._norm_factors[property_name] * \
                                                   self._norm_factors["categories"] * \
                                                   self.dataset.get_number_of_observations()

    # -- Matching support functions -- #

    def _match_classification_with_all_ground_truth(self):
        return self._match_classification_with_ground_truth(self.dataset.get_all_observations(),
                                                            self.dataset.get_proposals(self._model_name))

    def _match_classification_with_ground_truth_for_property_value(self, property_name, value):
        observations = self.dataset.get_observations_from_property(property_name, value)
        proposals = self.dataset.get_proposals(self._model_name)

        return self._match_classification_with_ground_truth(observations, proposals)

    def _match_classification_with_ground_truth(self, observations, proposals):
        if self.dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            proposals = proposals.sort_values(by="confidence", ascending=False).groupby(
                self.dataset.match_param_props).first()
            proposals = proposals.reset_index()
        elif self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            category_id = self.dataset.get_category_id_from_name(self.dataset.get_categories_names()[0])
            proposals = proposals.loc[proposals["category_id"] == category_id]

        matching = pd.merge(observations, proposals, how="left", left_on=self.dataset.match_param_gt,
                            right_on=self.dataset.match_param_props).replace(np.nan, 0)

        return matching
