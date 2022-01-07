import copy
import math
from collections import defaultdict
from numbers import Number
from statistics import mean

import numpy as np
import pandas as pd

from sklearn.metrics import auc
from tqdm import tqdm, tqdm_notebook

from odin.classes import DatasetLocalization, Metrics, Curves, TaskType
from odin.classes.analyzer_interface import AnalyzerInterface
from odin.classes.strings import err_type, err_value
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_reliability_diagram, plot_false_positive_errors, plot_iou_analysis, pie_plot, \
    plot_false_positive_trend
from odin.utils.env import is_notebook
from odin.utils.utils import sg_intersection_over_union, bb_intersection_over_union

logger = get_root_logger()


class AnalyzerLocalization(AnalyzerInterface):

    matching_dict = {}

    _iou_thresh_weak = 0.1  # intersection/union threshold
    _iou_thresh_strong = 0.5  # intersection/union threshold

    _SAVE_PNG_GRAPHS = True

    __normalized_number_of_images = 1000

    _valid_metrics = [Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE,
                      Metrics.AVERAGE_PRECISION_SCORE, Metrics.PRECISION_RECALL_AUC, Metrics.F1_AUC,
                      Metrics.AVERAGE_PRECISION_INTERPOLATED]

    _valid_curves = [Curves.PRECISION_RECALL_CURVE, Curves.F1_CURVE]

    _valid_cams_metrics = []

    def __init__(self,
                 detector_name,
                 dataset,
                 result_saving_path='./results/',
                 use_normalization=True,
                 norm_factor_categories=None,
                 norm_factors_properties=None,
                 iou=None,
                 iou_weak=None,
                 conf_thresh=None,
                 metric=Metrics.AVERAGE_PRECISION_SCORE,
                 save_graphs_as_png=True):
        """
        The AnalyzerLocalization class can be used to perform diagnostics for localization models, such as object detection and instance segmentation.

        Parameters
        ----------
        detector_name: str
            Name of the detector. It is used as folder to save results.
        dataset: DatasetLocalization
            Dataset used to perform the analysis.
        result_saving_path: str, optional
            Path used to save results. (default is './results/')
        use_normalization: bool, optional
            Indicates whether normalisation should be used. (default is False)
        norm_factor_categories: float, optional
            Normalisation factor for the categories. If not specified, the default value is 1/number_of_categories. (default is None))
        norm_factors_properties: list of pair, optional
            Properties normalization factors.

            Each pair (property name, normalization factor value) specifies the normalization factor to apply to a
            specific property.
            If not specified, for each property the default value is 1/number_of_property_values. default is None)
        iou: float, optional
            Intersection Over Union threshold. All the predictions with a iou value less than the threshold are considered False Positives. If not specified, the default value is 0.5. (default is None)
        iou_weak: float, optional
            Intersection Over Union weak threshold. Used for the identification of the localization errors. If not specified, the default value is 0.1. (default is None)
        conf_thresh: float, optional
            Confidence threshold. All the predictions with a confidence value less than the threshold are ignored. If not specified, the default value is 0.5. (default is None)
        metric: Metrics
            The evaluation metric that will be used as default. (default is Metrics.Metrics.AVERAGE_PRECISION_SCORE)
        save_graphs_as_png: bool
            Specifies whether or not to save the graphs as .png images (default is True)
        """
        if not isinstance(detector_name, str):
            raise TypeError(err_type.format("detector_name"))

        if type(dataset) is not DatasetLocalization:
            raise TypeError(err_type.format("dataset"))

        if detector_name not in dataset.proposals:
            loaded_models = list(dataset.proposals.keys())
            if len(loaded_models) == 1 and "model" in loaded_models:
                dataset.proposals[detector_name] = dataset.proposals["model"]
                del dataset.proposals["model"]
            else:
                raise Exception(
                    "No proposals. Unable to perform any type of analysis. "
                    "Please make sure to load the proposals to the dataset for {} model".format(detector_name))

        if not isinstance(result_saving_path, str):
            raise TypeError(err_type.format("result_saving_path"))

        if not isinstance(use_normalization, bool):
            raise TypeError(err_type.format("use_normalization"))

        if norm_factor_categories is not None and not isinstance(norm_factor_categories, Number):
            raise TypeError(err_type.format("norm_factor_categories"))

        if norm_factors_properties is not None and (not isinstance(norm_factors_properties, list) or
                                                    not (all(isinstance(item, tuple) and len(item) == 2
                                                             for item in norm_factors_properties))):
            raise TypeError(err_type.format("norm_factors_properties"))

        if iou is None:
            iou = 0.5
        elif not isinstance(iou, Number):
            raise TypeError(err_type.format("iou"))
        elif iou < 0 or iou > 1:
            raise ValueError(err_value.format("iou", "0 <= x <= 1"))

        if iou_weak is None:
            iou_weak = 0.1
        elif not isinstance(iou_weak, Number):
            raise TypeError(err_type.format("iou_weak"))
        elif iou_weak < 0 or iou_weak > 1:
            raise ValueError(err_value.format("iou_weak", "0 <= x <= 1"))
        elif iou_weak >= iou:
            raise ValueError(err_value.format("iou_weak", "iou_weak < iou"))

        self._iou_thresh_strong = iou
        self._iou_thresh_weak = iou_weak

        if conf_thresh is not None:
            if not isinstance(conf_thresh, Number):
                raise TypeError(err_type.format("conf_thresh"))
            if conf_thresh < 0 or conf_thresh > 1:
                raise ValueError(err_value.format("conf_thresh", "0 <= x >= 1"))

        if not isinstance(metric, Metrics):
            raise TypeError(err_type.format("metric"))

        if not isinstance(save_graphs_as_png, bool):
            raise TypeError(err_type.format("save_graphs_as_png"))

        if not dataset.are_analyses_without_properties_available():
            raise Exception("Please complete the properties selection first")

        if not dataset.are_analyses_with_properties_available():
            logger.warning("No properties available")

        if metric not in self._valid_metrics:
            raise Exception(f"Unsupported metric: {metric}")

        if conf_thresh is None:
            conf_thresh = 0.5

        self._SAVE_PNG_GRAPHS = save_graphs_as_png
        self.matching_dict = {}

        self.__tqdm = tqdm_notebook if is_notebook() else tqdm

        super().__init__(detector_name, dataset, result_saving_path, use_normalization, norm_factor_categories,
                         norm_factors_properties, conf_thresh, metric)

    def analyze_intersection_over_union(self, categories=None, metric=None, show=True):
        """
        It provides a per-category analysis of the model performances at different Intersection Over Union (IoU) thresholds.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        metric: Metrics, optional
            Evaluation metric used for the analysis. If not specified, the default one is used. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            logger.error(err_value.format("categories", self.dataset.get_categories_names()))
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

        results, display_names = {}, {}
        for category in categories:
            results[category] = self.analyze_intersection_over_union_for_category(category, metric, show=False)
            display_names[category] = self.dataset.get_display_name_of_category(category)

        if not show:
            return results

        plot_iou_analysis(results, metric, display_names, self._SAVE_PNG_GRAPHS, self.result_saving_path)

    def analyze_intersection_over_union_for_category(self, category, metric=None, show=True):
        """
        It provides a per-category analysis of the model performances at different Intersection Over Union (IoU) thresholds.

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

        if "iou" not in self.saved_analyses:
            self.saved_analyses["iou"] = {}

        if category not in self.saved_analyses["iou"]:
            proposals = self.dataset.get_proposals(self._model_name)
            if not self.matching_dict or "iou" not in self.matching_dict:
                annotations = self.dataset.get_annotations()
                self.matching_dict["iou"] = self._match_detection_with_ground_truth(annotations, proposals, 0).copy()

            matching = self.matching_dict["iou"].copy()

            cat_id = self.dataset.get_category_id_from_name(category)
            matching = matching.loc[matching["category_det"] == cat_id]

            category_anns = self.dataset.get_anns_for_category(cat_id)

            ious_unique = np.arange(0, 1.001, 0.05).round(5)
            metric_values = []
            for iou in ious_unique:
                matching['label'] = np.where((matching['label'] == 1) & (matching['iou'] < iou), -1, matching['label'])
                value, _ = self._compute_metric(category_anns, proposals, matching, metric)
                metric_values.append(value)
            result = {"iou": ious_unique, "metric_values": metric_values}
            self.saved_analyses[category] = result
        else:
            result = self.saved_analyses[category]

        if not show:
            return result

        plot_iou_analysis({category: result}, metric, {category: self.dataset.get_display_name_of_category(category)},
                          self._SAVE_PNG_GRAPHS, self.result_saving_path)

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
        elif num_bins < 2 or num_bins > 50:
            logger.error(err_value.format("num_bins", "2 <= x >= 50"))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        if "reliability" not in self.saved_analyses:
            self.saved_analyses["reliability"] = {}

        if "overall" not in self.saved_analyses["reliability"]:
            self.saved_analyses["reliability"]["overall"] = {}

        if str(num_bins) not in self.saved_analyses["reliability"]["overall"]:
            if not self.matching_dict or "all" not in self.matching_dict.keys():
                anns = self.dataset.get_annotations()
                proposals = self.dataset.get_proposals(self._model_name)
                self.matching_dict["all"] = self._match_detection_with_ground_truth(anns, proposals,
                                                                                    self._iou_thresh_strong).copy()

            matching = self.matching_dict["all"].copy()

            numpy_confidence, numpy_label = self.__support_reliability(matching)
            result = self._calculate_reliability(numpy_label, numpy_confidence, num_bins)
            self.saved_analyses["reliability"]["overall"][str(num_bins)] = result
        else:
            result = self.saved_analyses["reliability"]["overall"][str(num_bins)]

        if not show:
            return result

        plot_reliability_diagram(result, self._SAVE_PNG_GRAPHS, self.result_saving_path, is_classification=False)

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

        if not isinstance(num_bins, int):
            logger.error(err_type.format("num_bins"))
            return -1
        elif num_bins < 2 or num_bins > 50:
            logger.error(err_value.format("num_bins", "2 <= x >= 50"))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            logger.error(err_value.format("categories", self.dataset.get_categories_names()))
            return -1

        if "reliability" not in self.saved_analyses:
            self.saved_analyses["reliability"] = {}

        if not self.matching_dict or "all" not in self.matching_dict.keys():
            anns = self.dataset.get_annotations()
            proposals = self.dataset.get_proposals(self._model_name)
            self.matching_dict["all"] = self._match_detection_with_ground_truth(anns, proposals,
                                                                                self._iou_thresh_strong).copy()

        matching = self.matching_dict["all"].copy()

        results = {}
        for category in categories:
            if category not in self.saved_analyses["reliability"]:
                self.saved_analyses["reliability"][category] = {}

            if str(num_bins) not in self.saved_analyses["reliability"][category]:
                cat_id = self.dataset.get_category_id_from_name(category)
                numpy_confidence, numpy_label = self.__support_reliability(matching, cat_id)
                result = self._calculate_reliability(numpy_label, numpy_confidence, num_bins)
                self.saved_analyses["reliability"][category][str(num_bins)] = result
            else:
                result = self.saved_analyses["reliability"][category][str(num_bins)]

            results[category] = result

            if show:
                plot_reliability_diagram(result, self._SAVE_PNG_GRAPHS, self.result_saving_path,
                                         is_classification=False,
                                         category=self.dataset.get_display_name_of_category(category))

        if not show:
            return results

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

            fp_detection_ids = []
            for error in error_dict:
                fp_detection_ids.extend(error_dict[error])

            labels = list(error_dict.keys())

            if include_correct_predictions:
                matching = self.matching_dict["all"].loc[self.matching_dict["all"]["category_det"] == self.dataset.get_category_id_from_name(category)]
                error_trend = [np.cumsum(np.where((matching["label"] == 1) & (matching["confidence"] >= self._conf_thresh), 1, 0))]
                error_trend.extend([np.cumsum(np.where(matching["det_id"].isin(error_dict[error]), 1, 0)) for error in error_dict])
                error_trend = np.array(error_trend)
                labels.insert(0, "correct")
            else:
                matching = self.matching_dict["all"].loc[self.matching_dict["all"]["det_id"].isin(fp_detection_ids)]
                error_trend = np.array([np.cumsum(np.where(matching["det_id"].isin(error_dict[error]), 1, 0)) for error in error_dict])

            error_sum = np.sum(error_trend, axis=0)
            error_trend = np.divide(error_trend, error_sum)

            result = {}
            for i, error in enumerate(labels):
                result[error] = error_trend[i, :]

            self.saved_analyses["false_positive_trend"][category][str(include_correct_predictions)] = result

        result = self.saved_analyses["false_positive_trend"][category][str(include_correct_predictions)]

        if not show:
            return result

        plot_false_positive_trend(result, "False Positive Trend - " + self.dataset.get_display_name_of_category(category), self._SAVE_PNG_GRAPHS, self.result_saving_path)

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
            matching = values["matching"]

            errors = ["localization", "similar", "other", "background"]
            error_values = []

            for error in errors:
                if len(error_dict[error]) == 0:
                    error_values.append([category_metric_value, 0])
                    continue

                matching_tmp = matching[~matching["det_id"].isin(error_dict[error])]
                list_ids = matching_tmp["det_id"].tolist()

                det = self.dataset.get_proposals_by_category_and_ids(category, list_ids, self._model_name)
                anns = self.dataset.get_anns_for_category(self.dataset.get_category_id_from_name(category))

                self._set_normalized_number_of_images_for_categories()

                metric_value, _ = self._compute_metric(anns, det, matching_tmp, metric)

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
            cat_id = self.dataset.get_category_id_from_name(category)

            if not self.matching_dict or "all" not in self.matching_dict.keys():
                anns = self.dataset.get_annotations()
                proposals = self.dataset.get_proposals(self._model_name)
                matching = self._match_detection_with_ground_truth(anns, proposals, self._iou_thresh_strong)
                self.matching_dict["all"] = matching.copy()
            else:
                matching = self.matching_dict["all"].copy()

            matching = matching.loc[matching["ann_id"].isin(fn_ids[category]["gt"])]
            self.saved_analyses["false_negative_errors"][category]["localization"] = len(matching.loc[(matching["category_det"] == cat_id) &
                                                                         (matching["iou"] >= self._iou_thresh_weak)].index)

            self.saved_analyses["false_negative_errors"][category]["similar"] = len(matching.loc[(matching["category_det"] != cat_id) &
                                                                    (matching["iou"] >= self._iou_thresh_weak) &
                                                                    (matching["category_det"].apply(lambda x: self.dataset.is_similar(cat_id, x)))].index)

            self.saved_analyses["false_negative_errors"][category]["no_prediction"] = len(matching.loc[matching["iou"] < self._iou_thresh_weak].index)

            self.saved_analyses["false_negative_errors"][category]["other"] = len(matching.index) - \
                                                                              (self.saved_analyses["false_negative_errors"][category]["localization"] +
                                                                               self.saved_analyses["false_negative_errors"][category]["similar"] +
                                                                               self.saved_analyses["false_negative_errors"][category]["no_prediction"])

        if not show:
            return self.saved_analyses["false_negative_errors"][category]

        display_name = self.dataset.get_display_name_of_category(category)
        pie_plot(list(self.saved_analyses["false_negative_errors"][category].values()),
                 list(self.saved_analyses["false_negative_errors"][category].keys()),
                 "False Negative categorization - {}".format(display_name),
                 self.result_saving_path, self._SAVE_PNG_GRAPHS, colors=['lightskyblue', 'lightgreen', 'coral', 'orchid'])

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
            If False don't show the properties in the report (default is True)
        include_reliability: bool, optional
            Indicates whether the 'ece' and 'mce' should be included in the report. (default is True)
        Returns
        -------
        pandas.DataFrame
        """
        if show_properties and not self.dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset or set 'show_properties=False'")
            return -1

        default_metrics = [Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE,
                           Metrics.AVERAGE_PRECISION_SCORE, Metrics.AVERAGE_PRECISION_INTERPOLATED]
        return self._get_report_results(default_metrics, metrics, categories, properties, show_categories,
                                        show_properties, include_reliability)

    def get_matching_dataframe(self):
        """
        Returns the matching between ground truth and the proposals as a pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if not self.matching_dict or "all" not in self.matching_dict.keys():
            proposals = self.dataset.get_proposals(self._model_name)
            anns = self.dataset.get_annotations()
            self.matching_dict["all"] = self._match_detection_with_ground_truth(
                anns, proposals, self._iou_thresh_strong).copy()
        return self.matching_dict["all"].copy()

    def set_iou_threshold(self, iou=None, iou_weak=None):
        """
        Sets the intersection over union (IoU) threshold. The iou_weak is used as lower bound for the localization errors.
        Parameters
        ----------
        iou: float, optional
            Intersection Over Union threshold. All the predictions with a iou value less than the threshold are considered False Positives.
        iou_weak: float, optional
            Intersection Over Union weak threshold. Used for the identification of the localization errors.
        """

        if iou is None:
            iou = self._iou_thresh_strong
        elif not isinstance(iou, Number):
            logger.error(err_type.format("iou"))
            return -1
        elif iou < 0 or iou > 1:
            logger.error(err_value.format("iou", "0 <= iou <= 1"))
            return -1

        if iou_weak is None:
            iou_weak = self._iou_thresh_weak
        elif not isinstance(iou_weak, Number):
            logger.error(err_type.format("iou_weak"))
            return -1
        elif iou_weak < 0 or iou_weak > 1:
            logger.error(err_value.format("iou_weak", "0 <= iou_weak <= 1"))
            return -1

        if iou <= iou_weak:
            logger.error(err_value.format("iou/iou_weak", "iou > iou_weak"))
            return -1

        if iou != self._iou_thresh_strong or iou_weak != self._iou_thresh_weak:
            self.clear_saved_results()
            self.clear_saved_analyses()
            if "all" in self.matching_dict.keys():
                del self.matching_dict["all"]

        self._iou_thresh_strong = iou
        self._iou_thresh_weak = iou_weak

    # -- EVALUATION METRICS -- #

    def _compute_metric(self, gt, detections, matching, metric, is_micro_required=False):
        """
        Method used to call the metric that is used to calculate the performances

        Parameters
        ----------
        gt: pandas.DataFrame
            Ground Truth
        detections: pandas.DataFrame
            Detections
        metric: Metrics
            Metric selected for the computation
        matching: pandas.DataFrame
            Ground Truth and Proposals matching
        is_micro_required: bool, optional
            If True it is not a single class analysis
        Returns
        -------
        dict containing the metric value and the standard error
        """
        if metric == Metrics.PRECISION_SCORE:
            return self._compute_metric_precision_score(gt, matching)
        elif metric == Metrics.RECALL_SCORE:
            return self._compute_metric_recall_score(gt, matching)
        elif metric == Metrics.F1_SCORE:
            return self._compute_metric_f1_score(gt, matching)
        elif metric == Metrics.AVERAGE_PRECISION_SCORE:
            return self._compute_average_precision_score(gt, matching)
        elif metric == Metrics.PRECISION_RECALL_AUC:
            return self._compute_precision_recall_auc_curve(gt, matching)
        elif metric == Metrics.F1_AUC:
            return self._compute_f1_auc_curve(gt, matching)
        elif metric == Metrics.AVERAGE_PRECISION_INTERPOLATED:
            return self._compute_average_precision_interpolated(gt, matching)
        else:
            custom_metrics = self.get_custom_metrics()
            if metric not in custom_metrics:
                raise NotImplementedError(f"Not supported metric : {metric}")
            return custom_metrics[metric].evaluate_metric(gt, detections, matching, is_micro_required)

    def _compute_metric_precision_score(self, gt, matching):
        """
        Calculates the precision score

        Parameters
        ----------
        gt: pandas.DataFrame
            Ground Truth
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        precision_score, None (the standard error)
        """

        n_anns, n_normalized, numpy_confidence, _, tp, fp = self._support_metric(gt, matching)
        tp, tp_norm, fp = self._support_metric_threshold(n_anns, n_normalized, numpy_confidence, tp, fp,
                                                         self._conf_thresh)
        precision, precision_norm = self._support_precision_score(tp, tp_norm, fp)

        if self._use_normalization:
            return precision_norm, None
        else:
            return precision, None

    def _compute_metric_recall_score(self, gt, matching):
        """
        Calculates the recall score

        Parameters
        ----------
        gt: pandas.DataFrame
            Ground Truth
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        recall_score, None (the standard error)
        """

        n_anns, n_normalized, numpy_confidence, _, tp, fp = self._support_metric(gt, matching)
        tp, tp_norm, fp = self._support_metric_threshold(n_anns, n_normalized, numpy_confidence, tp, fp,
                                                         self._conf_thresh)
        fn = n_anns - tp

        recall, recall_norm = self._support_recall_score(tp, tp_norm, fn)
        if self._use_normalization:
            return recall_norm, None
        else:
            return recall, None

    def _compute_metric_f1_score(self, gt, matching):
        """
        Calculates the F1 score

        Parameters
        ----------
        gt: pandas.DataFrame
            Ground Truth
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        f1_score, None (the standard error)

        """

        n_anns, n_normalized, numpy_confidence, _, tp, fp = self._support_metric(gt, matching)
        tp, tp_norm, fp = self._support_metric_threshold(n_anns, n_normalized, numpy_confidence, tp, fp,
                                                         self._conf_thresh)

        fn = n_anns - tp
        f1, f1_norm = self._support_f1_score(tp, tp_norm, fp, fn)
        if self._use_normalization:
            return f1_norm, None
        else:
            return f1, None

    def _compute_average_precision_score(self, gt, matching):
        """
        Calculates the average precision score

        Parameters
        ----------
        gt: pandas.DataFrame
            Ground Truth
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        metric_value, std_err
        """

        n_anns, n_normalized, numpy_confidence, label, tp, fp = self._support_metric(gt, matching)

        metric_value, std_err = self._support_average_precision(len(gt.index), n_anns, n_normalized,
                                                                numpy_confidence, tp, fp, False)
        if np.isnan(metric_value):
            metric_value = 0

        return metric_value, std_err

    def _compute_average_precision_interpolated(self, gt, matching):
        """
        Calculates the average precision interpolated score

        Parameters
        ----------
        gt: pandas.DataFrame
            Ground Truth
        matching: pandas.DataFrame
            Matching between the ground truth and the predictions

        Returns
        -------
        metric_value, standard_error
        """
        n_anns = len(gt.index)
        n_normalized = self._get_normalized_number_of_images()

        confidence = matching["confidence"].tolist()
        label = matching["label"].tolist()

        return self._average_precision_normalized(confidence, label, n_anns, n_normalized)

    def _average_precision_normalized(self, confidence, label, n_anns, n_normalized):
        """
        Calculates the average precision interpolated.

        N.B In this implementation, if use_normalization=True, only the precision is normalized

        Parameters
        ----------
        confidence: array-like
            Predictions confidences
        label: array-like
            Predictions labels (1, 0, -1)
        n_anns: int
            Number of annotations
        n_normalized: float
            Normalized number of images

        Returns
        -------
        metric_value, standard_error
        """
        label = copy.deepcopy(label)

        si = sorted(range(len(confidence)), key=lambda k: confidence[k], reverse=True)
        label = [label[i] for i in si]

        tp = np.cumsum((np.array(label) == 1).astype(int))  # True positive cumsum
        fp = np.cumsum((np.array(label) == -1).astype(int))  # False Positive cumsum

        recall = np.true_divide(tp, n_anns)
        precision = np.true_divide(tp, np.add(tp, fp))

        rec_norm = np.true_divide(np.multiply(tp, n_normalized), n_anns)  # recall normalized
        prec_norm = np.true_divide(rec_norm, np.add(rec_norm, fp))  # precision normalized

        # compute interpolated precision and normalized precision
        label_numpy = np.array(label)
        is_tp = (label_numpy == 1)  # is true positive
        num_rec = recall.size
        # for the sake of memory, used the same variable to compute the interpolated precision
        for i in range(num_rec - 2, -1,
                       -1):  # go through the array in reverse to get the interpolated & interpol normalized
            precision[i] = max(precision[i], precision[i + 1])
            prec_norm[i] = max(prec_norm[i], prec_norm[i + 1])

        m_size_array = n_anns - sum((label_numpy == 1))
        m_size_array = m_size_array if m_size_array > 0 else 0

        missed = np.zeros(m_size_array,
                          dtype=np.float64)  # create an array of zeros for missed for the sake of calculation
        np.warnings.filterwarnings('ignore')  # just to not print warnings when there are NaN

        ap = np.mean(precision[is_tp]) * recall[-1]  # avg precision
        ap_std = np.std(np.concatenate((precision[is_tp], missed))) / np.sqrt(
            n_anns)  # avg precision standard error

        ap_normalized = np.mean(prec_norm[is_tp]) * recall[-1]
        ap_norm_std = np.std(np.concatenate((prec_norm[is_tp], missed))) / np.sqrt(n_anns)

        if np.isnan(ap):
            ap = ap_normalized = ap_std = ap_norm_std = 0
        if self._use_normalization:
            return ap_normalized, ap_norm_std
        else:
            return ap, ap_std

    def _compute_precision_recall_auc_curve(self, gt, matching):
        """
        Calculates the area under the Precision-Recall curve

        Parameters
        ----------
        gt:  pandas.DataFrame
            Ground Truth
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        auc, standard_error

        """

        recall, precision = self._compute_precision_recall_curve(gt, matching)
        std_err = np.std(precision) / np.sqrt(precision.size)
        return auc(recall, precision), std_err

    def _compute_f1_auc_curve(self, gt, matching):
        """
        Calculates the area under the F1 curve

        Parameters
        ----------
        gt: pandas.DataFrame
            Ground Truth
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        auc, standard_error

        """

        thresholds, f1 = self._compute_f1_curve(gt, matching)
        std_err = np.std(f1) / np.sqrt(f1.size)
        return auc(thresholds, f1), std_err

    # -- Evaluation metrics support functions -- #

    def _support_metric(self, gt, matching):
        """
        Calculates the True Positive and False Positive cumsum, the number of annotations, the normalized number of
        images and creates two array-like with the confidences and the corresponding matching labels

        Parameters
        ----------
        gt: pandas.DataFrame
            Ground Truth
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        n_anns, n_normalized, numpy_confidence, numpy_label, tp, fp
        """
        n_anns = len(gt.index)
        n_normalized = self._get_normalized_number_of_images()

        numpy_confidence = matching["confidence"].values
        numpy_label = matching["label"].values

        tp = np.cumsum((numpy_label == 1).astype(int))  # True positive cumsum
        fp = np.cumsum((numpy_label == -1).astype(int))  # False Positive cumsum
        return n_anns, n_normalized, numpy_confidence, numpy_label, tp, fp

    def _support_metric_threshold(self, n_true_gt, n_normalized, det_ord, tp, fp, threshold):
        """
        Calculates the True Positive, True Positive Normalized and False Positive based on the threshold

        Parameters
        ----------
        n_true_gt: int
            Number of annotations considered
        n_normalized: float
            Number of images normalized
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
        tp, tp_norm, fp
        """

        tp[det_ord < threshold] = 0
        fp[det_ord < threshold] = 0

        tp = tp[tp > 0]
        tp = tp[-1] if len(tp) > 0 else 0

        fp = fp[fp > 0]
        fp = fp[-1] if len(fp) > 0 else 0

        tp_norm = tp * n_normalized / n_true_gt if n_true_gt > 0 else 0
        return tp, tp_norm, fp

    def _match_detection_with_ground_truth(self, gt, proposals, iou_thres, disable_progress_status=False):
        """
        Matches the Ground Truth annotations with the predictions and assigns '1' as label value to all those
        predictions that match the ground truth category and have an intersection over union value greater than the
        threshold; to all others assigns '-1' as label value.

        Parameters
        ----------
        gt: pandas.DataFrame
            Ground Truth
        proposals: pandas.DataFrame
            Predictions
        iou_thres: float
            Intersection over Union threhsold

        Returns
        -------
        pandas.DataFrame
        """
        if not disable_progress_status:
            print("Matching proposals of {} model with ground truths...".format(self._model_name))

        annotations_matched_id = []
        matching = []

        imgs = self.dataset.images
        gt_ord = gt.sort_values(by="id")
        anns = pd.merge(gt_ord, imgs, how="left", left_on="image_id", right_on="id")
        anns = anns.groupby("image_id")

        props = proposals.sort_values(["confidence", "id"], ascending=[False, True])

        for det in self.__tqdm(props.to_dict("records"), total=len(props.index), disable=disable_progress_status):
            total_ious = []
            match_info = {"confidence": det["confidence"], "difficult": 0, "label": -1, "duplicated": 0, "iou": -1,
                          "det_id": det["id"], "ann_id": -1, "category_det": det["category_id"],
                          "category_ann": -1}

            img_id = det[self.dataset.match_param_props]
            if img_id in anns.groups:
                for ann in anns.get_group(img_id).to_dict("records"):
                    iou = self.__intersection_over_union(ann, det)
                    total_ious.append([iou, ann])

            # Sort iou by higher score to evaluate prediction
            total_ious = sorted(total_ious, key=lambda k: k[0], reverse=True)
            ious = [i for i in total_ious if (i[1]["category_id"] == det["category_id"]) and i[0] >= self._iou_thresh_weak]
            if len(ious) > 0:
                iou, ann = ious[0]
                match_info = {"confidence": det["confidence"], "difficult": 0, "label": -1,
                              "iou": iou, "det_id": det["id"], "ann_id": ann["id_x"],
                              "category_det": det["category_id"], "category_ann": ann["category_id"]}
                not_used = False
                for iou, ann in ious:
                    if not ann["id_x"] in annotations_matched_id:
                        not_used = True
                        break
                if not_used:
                    # Corroborate is correct detection
                    if iou >= iou_thres:
                        # Difficult annotations are ignored
                        if "difficult" in ann.keys() and ann["difficult"] == 1:
                            match_info = {"confidence": det["confidence"], "difficult": 1, "label": 0, "iou": iou,
                                          "det_id": det["id"], "ann_id": ann["id_x"],
                                          "category_det": det["category_id"],
                                          "category_ann": ann["category_id"]}
                        else:

                            match_info = {"confidence": det["confidence"], "difficult": 0, "label": 1, "iou": iou,
                                          "det_id": det["id"], "ann_id": ann["id_x"],
                                          "category_det": det["category_id"],
                                          "category_ann": ann["category_id"]}
                            annotations_matched_id.append(ann["id_x"])
            elif len(total_ious) > 0:
                iou, ann = total_ious[0]
                # Save the max iou for the detection for later analysis
                match_info = {"confidence": det["confidence"], "difficult": 0, "label": -1, "iou": iou,
                              "det_id": det["id"], "ann_id": ann["id_x"], "category_det": det["category_id"],
                              "category_ann": ann["category_id"]}
            matching.append(match_info)

        if not disable_progress_status:
            print("Done!")
        return pd.DataFrame(matching)

    def __intersection_over_union(self, ann, det):
        """
        Calculates the intersection over union between the annotation and the prediction

        Parameters
        ----------
        ann: pandas.Series
            Annotation
        det: pandas.Series
            Detection

        Returns
        -------
        iou: float
        """
        if self.dataset.task_type == TaskType.INSTANCE_SEGMENTATION:
            if "height" in ann.keys() and "width" in ann.keys():
                iou = sg_intersection_over_union(ann['segmentation'][0], det['segmentation'], ann["height"],
                                                 ann["width"])
            else:
                h, w = self.dataset.get_height_width_from_image(ann["image_id"])
                iou = sg_intersection_over_union(ann['segmentation'][0], det['segmentation'], h, w)
        else:
            bbox = ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + \
                   ann['bbox'][3]
            bbox_det = det['bbox'][0], det['bbox'][1], det['bbox'][0] + det['bbox'][2], det['bbox'][1] + \
                       det['bbox'][3]
            iou = bb_intersection_over_union(bbox, bbox_det)
        return iou

    # --- CURVES --- #

    def _compute_curve(self, anns, matching, curve):
        """
        Computes the selected curve values
        Parameters
        ----------
        anns: pandas.DataFrame
            Ground truth annotations
        matching: pandas.DataFrame
            Matching dataframe
        curve: Curves
            Curve used for the computation

        Returns
        -------
            x_values, y_values
        """
        if curve == Curves.PRECISION_RECALL_CURVE:
            return self._compute_precision_recall_curve(anns, matching)
        elif curve == Curves.F1_CURVE:
            return self._compute_f1_curve(anns, matching)
        else:
            raise ValueError("Invalid curve name.")

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
            annotations = self.dataset.get_annotations()

            if not self.matching_dict or "all" not in self.matching_dict.keys():
                proposals = self.dataset.get_proposals(self._model_name)
                self.matching_dict["all"] = self._match_detection_with_ground_truth(annotations, proposals,
                                                                                    self._iou_thresh_strong)
            matching = self.matching_dict["all"].copy()

            user_normalization = self._use_normalization
            self._use_normalization = False
            x_values, y_values = self._compute_curve(annotations, matching, curve)
            self._use_normalization = user_normalization
            auc_value = auc(x_values, y_values)
            self.saved_analyses[curve.value]["overall"] = {'overall': {'auc': auc_value,
                                                                       'x': x_values,
                                                                       'y': y_values}}
        results = self.saved_analyses[curve.value]["overall"]
        return results

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

        results = {}

        self._set_normalized_number_of_images_for_categories()
        for category in categories:
            if category not in self.saved_analyses[curve.value]:
                if not self.matching_dict or "all" not in self.matching_dict.keys():
                    self.matching_dict["all"] = self._match_detection_with_ground_truth(self.dataset.get_annotations(),
                                                                                        self.dataset.get_proposals(self._model_name),
                                                                                        self._iou_thresh_strong)
                matching = self.matching_dict["all"].copy()

                cat_id = self.dataset.get_category_id_from_name(category)
                matching = matching.loc[matching["category_det"] == cat_id]

                anns = self.dataset.get_anns_for_category(self.dataset.get_category_id_from_name(category))
                x_values, y_values = self._compute_curve(anns, matching, curve)

                auc_value = auc(x_values, y_values)
                self.saved_analyses[curve.value][category] = {'auc': auc_value,
                                                              'x': x_values,
                                                              'y': y_values}
            results[category] = self.saved_analyses[curve.value][category]
        return results

    def _compute_precision_recall_curve(self, gt, matching):
        """
        Calculates the Recall (x-values) and Precision (y-values) of the Precision-Recall curve

        Parameters
        ----------
        gt:  pandas.DataFrame
            Ground Truth
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        recall, precision

        """

        n_anns, n_normalized, numpy_confidence, numpy_label, tp, fp = self._support_metric(gt, matching)
        precision, precision_norm, recall, recall_norm = self._support_precision_recall_auc(len(gt.index), n_anns,
                                                                                            n_normalized,
                                                                                            numpy_confidence, tp, fp,
                                                                                            False)
        if self._use_normalization:
            return recall_norm, precision_norm
        else:
            return recall, precision

    def _compute_f1_curve(self, gt, matching):
        """
        Calculates the F1 scores for different thresholds

        gt: pandas.DataFrame
            Ground Truth
        detections: pandas.DataFrame
            Detections confidence scores
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        thresholds, f1_scores

        """

        n_anns, n_normalized, numpy_confidence, numpy_label, tp, fp = self._support_metric(gt, matching)
        precision, precision_norm, recall, recall_norm, rel_indexes = self._support_precision_recall(n_anns,
                                                                                                     n_normalized,
                                                                                                     numpy_confidence,
                                                                                                     tp, fp)
        return self._support_f1_curve(numpy_confidence, precision, precision_norm, recall, recall_norm, rel_indexes)

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

        proposals = self.dataset.get_proposals(self._model_name)
        if not self.matching_dict or "all" not in self.matching_dict.keys():
            self.matching_dict["all"] = self._match_detection_with_ground_truth(self.dataset.get_annotations(),
                                                                                proposals,
                                                                                self._iou_thresh_strong)
        matching = self.matching_dict["all"].copy()

        category_id = self.dataset.get_category_id_from_name(category)
        anns = self.dataset.get_anns_for_category(category_id)
        matching = matching.loc[matching["category_det"] == category_id]

        self._set_normalized_number_of_images_for_categories()
        result, std_err = self._compute_metric(anns, proposals, matching, metric)
        value = {"value": result, "std": std_err, "matching": matching}
        self.saved_results[metric][category]["all"] = value

        return value

    def _calculate_metric_for_properties_of_category(self, category_name, category_id, property_name, possible_values,
                                                     matching, metric):
        """
        Calculates the metric for a specific category and for each property value

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
        matching: pandas.DataFrame
            Matching between ground truth and predictions
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
            self._set_normalized_number_of_images_for_property_for_categories(property_name)
        else:
            self._set_normalized_number_of_images_for_categories()

        for value in possible_values:
            if value in self.saved_results[metric][category_name][property_name].keys():
                result = self.saved_results[metric][category_name][property_name][value]
            else:
                anns = self.dataset.get_annotations_of_class_with_property(category_id, property_name, value)
                ann_ids = anns["id"].tolist()

                property_match = matching[((matching["label"] != 1) & (matching["category_det"] == category_id)) |
                                          (matching["ann_id"].isin(ann_ids))]
                list_ids = property_match["det_id"].tolist()

                det = self.dataset.get_proposals_by_category_and_ids(category_name, list_ids, self._model_name)

                metricvalue, std_err = self._compute_metric(anns, det, property_match, metric)
                if math.isnan(metricvalue):
                    metricvalue = 0
                result = {"value": metricvalue, "std": std_err}

                self.saved_results[metric][category_name][property_name][value] = result

            properties_results[value] = result
        return properties_results

    # -- Distribution support functions -- #

    def _analyze_true_positive_for_categories(self, categories):
        """
        Computes the True Positive for each category

        Parameters
        ----------
        categories: array-like
            Categories considered in the analysis

        Returns
        -------
        dict containing the number of True Positive for each class,
        dict containing the ids of the True Positive for each class

        """
        if "tp" not in self.saved_analyses:
            self.saved_analyses["tp"] = {"classes": {},
                                         "ids": {}}
        classes, ids = {}, {}
        for category in categories:

            if category not in self.saved_analyses["tp"]["classes"] or category not in self.saved_analyses["tp"]["ids"]:

                if not self.matching_dict or "all" not in self.matching_dict.keys():
                    self.matching_dict["all"] = self._match_detection_with_ground_truth(self.dataset.get_annotations(),
                                                                                        self.dataset.get_proposals(self._model_name),
                                                                                        self._iou_thresh_strong)
                matching = self.matching_dict["all"].copy()

                cat_id = self.dataset.get_category_id_from_name(category)
                matching = matching[(matching["label"] == 1) & (matching["confidence"] >= self._conf_thresh) &
                                    (matching["category_det"] == cat_id)]
                self.saved_analyses["tp"]["classes"][category] = len(matching.index)
                self.saved_analyses["tp"]["ids"][category] = {"gt": matching["ann_id"].tolist(),
                                                              "props": matching["det_id"].tolist()}

            classes[category] = self.saved_analyses["tp"]["classes"][category]
            ids[category] = self.saved_analyses["tp"]["ids"][category]
        return classes, ids

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

                if not self.matching_dict or "all" not in self.matching_dict.keys():
                    anns = self.dataset.get_annotations()
                    proposals = self.dataset.get_proposals(self._model_name)
                    matching = self._match_detection_with_ground_truth(anns, proposals, self._iou_thresh_strong)
                    self.matching_dict["all"] = matching.copy()
                else:
                    matching = self.matching_dict["all"].copy()

                anns = self.dataset.get_anns_for_category(cat_id)

                matching = matching[(matching["label"] == 1) & (matching["confidence"] >= self._conf_thresh) &
                                    (matching["category_det"] == cat_id)]
                self.saved_analyses["fn"]["classes"][category] = len(anns.index) - len(matching.index)
                self.saved_analyses["fn"]["ids"][category] = {"gt": anns.loc[~anns["id"].isin(matching["ann_id"])]["id"].tolist(),
                                    "props": []}
            classes[category] = self.saved_analyses["fn"]["classes"][category]
            ids[category] = self.saved_analyses["fn"]["ids"][category]
        return classes, ids

    def _analyze_false_positive_for_categories(self, categories):
        if "fp" not in self.saved_analyses:
            self.saved_analyses["fp"] = {"classes": {},
                                         "ids": {}}

        classes, ids = {}, {}
        for category in categories:
            if category not in self.saved_analyses["fp"]["classes"] or category not in self.saved_analyses["fp"]["ids"]:

                if not self.matching_dict or "all" not in self.matching_dict.keys():
                    self.matching_dict["all"] = self._match_detection_with_ground_truth(self.dataset.get_annotations(),
                                                                                        self.dataset.get_proposals(
                                                                                            self._model_name),
                                                                                        self._iou_thresh_strong).copy()
                matching = self.matching_dict["all"].copy()

                cat_id = self.dataset.get_category_id_from_name(category)

                matching = matching.loc[(matching["label"] == -1) & (matching["confidence"] >= self._conf_thresh) &
                                        (matching["category_det"] == cat_id)]

                self.saved_analyses["fp"]["classes"][category] = len(matching.index)
                self.saved_analyses["fp"]["ids"][category] = {"gt": matching["ann_id"].tolist() if not matching.empty else [],
                                                              "props": matching["det_id"].tolist() if not matching.empty else []}

            classes[category] = self.saved_analyses["fp"]["classes"][category]
            ids[category] = self.saved_analyses["fp"]["ids"][category]

        return classes, ids

    def _analyze_false_positive_errors(self, categories, threshold):
        """
        Analyzes the False Positive for each category by dividing the errors in four tags: background, similar (if the
        error is due to a similarity with another category), localization and other

        Parameters
        ----------
        categories
        threshold

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
                if not self.matching_dict or "all" not in self.matching_dict.keys():
                    self.matching_dict["all"] = self._match_detection_with_ground_truth(self.dataset.get_annotations(),
                                                                                        self.dataset.get_proposals(self._model_name),
                                                                                        self._iou_thresh_strong).copy()
                matching = self.matching_dict["all"].copy()

                cat_id = self.dataset.get_category_id_from_name(category)
                fp_ids_cat = {}

                matching = matching.loc[(matching["label"] == -1) & (matching["confidence"] >= threshold) &
                                        (matching["category_det"] == cat_id)]

                # background errors
                matching["bg_error"] = np.where(matching["iou"] < self._iou_thresh_weak, 1, 0)
                bg_indexes = matching[matching["bg_error"] == 1]["det_id"].tolist()
                fp_ids_cat["background"] = {"gt": self.dataset.get_annotations()["id"].tolist(),
                                                  "props": matching[matching["bg_error"] == 1]["det_id"].tolist()}

                # localization errors
                matching = matching[matching["iou"] >= self._iou_thresh_weak]
                matching["loc_error"] = np.where((matching["category_det"] == matching["category_ann"]), 1, 0)
                localization_indexes = matching[matching["loc_error"] == 1]["det_id"].tolist()
                fp_ids_cat["localization"] = {"gt": matching[matching["loc_error"] == 1]["ann_id"].tolist(),
                                                    "props": matching[matching["loc_error"] == 1]["det_id"].tolist()}

                # similar and other errors
                matching = matching[(matching["category_det"] != matching["category_ann"])]
                cat_id = self.dataset.get_category_id_from_name(category)
                matching["similar_others_error"] = np.where(matching["category_ann"].apply(lambda x: self.dataset.is_similar(cat_id, x)), 1, 2)
                similar_indexes = matching[matching["similar_others_error"] == 1]["det_id"].tolist()
                other_indexes = matching[matching["similar_others_error"] == 2]["det_id"].tolist()
                fp_ids_cat["similar"] = {"gt": matching[matching["similar_others_error"] == 1]["ann_id"].tolist(),
                                               "props": matching[matching["similar_others_error"] == 1]["det_id"].tolist()}
                fp_ids_cat["other"] = {"gt": matching[matching["similar_others_error"] == 2]["ann_id"].tolist(),
                                             "props": matching[matching["similar_others_error"] == 2]["det_id"].tolist()}

                self.saved_analyses["false_positive_errors"][str(threshold)]["errors"][category] = {"localization": localization_indexes,
                                                                                    "similar": similar_indexes,
                                                                                    "other": other_indexes,
                                                                                    "background": bg_indexes}
                self.saved_analyses["false_positive_errors"][str(threshold)]["ids"][category] = fp_ids_cat

            fp_errors[category] = self.saved_analyses["false_positive_errors"][str(threshold)]["errors"][category]
            fp_ids[category] = self.saved_analyses["false_positive_errors"][str(threshold)]["ids"][category]
        return fp_errors, fp_ids

    def _analyze_true_positive_for_category_for_property(self, category, property_name, property_values):
        tp_p_values = defaultdict(int)
        tp_ids = {}

        annotations = self.dataset.get_annotations()
        if not self.matching_dict or "all" not in self.matching_dict.keys():
            self.matching_dict["all"] = self._match_detection_with_ground_truth(annotations,
                                                                                self.dataset.get_proposals(self._model_name),
                                                                                self._iou_thresh_strong).copy()
        matching = self.matching_dict["all"].copy()

        cat_id = self.dataset.get_category_id_from_name(category)

        for p_value in property_values:
            anns_p = annotations.loc[annotations.index.get_level_values(property_name) == p_value]
            anns_p_ids = anns_p["id"].tolist()
            match = matching.loc[(matching["label"] == 1) & (matching["confidence"] >= self._conf_thresh) &
                                 (matching["ann_id"].isin(anns_p_ids)) & (matching["category_det"] == cat_id)]
            tp_p_values[p_value] = len(match.index)
            tp_ids[p_value] = {"gt": match["ann_id"].tolist() if not match.empty else [],
                               "props": match["det_id"].tolist() if not match.empty else []}

        return tp_p_values, tp_ids

    def _analyze_false_negative_for_category_for_property(self, category, property_name, property_values):
        fn_p_values = defaultdict(int)
        fn_ids = {}

        annotations = self.dataset.get_annotations()
        if not self.matching_dict or "all" not in self.matching_dict.keys():
            self.matching_dict["all"] = self._match_detection_with_ground_truth(annotations,
                                                                                self.dataset.get_proposals(self._model_name),
                                                                                self._iou_thresh_strong).copy()
        matching = self.matching_dict["all"].copy()

        cat_id = self.dataset.get_category_id_from_name(category)

        for p_value in property_values:
            anns_p = annotations.loc[annotations.index.get_level_values(property_name) == p_value]
            anns_p_ids = anns_p["id"].tolist()
            match = matching.loc[(matching["label"] == 1) & (matching["confidence"] >= self._conf_thresh) &
                                 (matching["ann_id"].isin(anns_p_ids)) & (matching["category_det"] == cat_id)]
            fn_p_values[p_value] = len(anns_p.index) - len(match.index)
            fn_ids[p_value] = {"gt": anns_p.loc[~anns_p["id"].isin(matching["ann_id"])]["id"].tolist() if not match.empty else [],
                               "props": []}

        return fn_p_values, fn_ids

    # -- Reliability support functions -- #

    def _calculate_reliability(self, label, y_score, num_bins, category=None):
        """
        Calculates the reliability

        Parameters
        ----------
        label:  array-like
            Predictions label
        y_score: array-like
            Predictions scores
        num_bins: int
            Number of bins used to split the confidence values

        Returns
        -------
        dict : {'values': bin_precisions, 'gaps': gaps, 'counts': bin_counts, 'bins': bins,
                  'avg_value': avg_prec, 'avg_conf': avg_conf, 'ece': ece, 'mce': mce}
        """
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        indices = np.digitize(y_score, bins, right=True)
        bin_precisions = np.zeros(num_bins, dtype=float)
        bin_confidences = np.zeros(num_bins, dtype=float)
        bin_counts = np.zeros(num_bins, dtype=int)

        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                v_selected = label[selected]
                tp = np.cumsum(v_selected == 1)
                fp = np.cumsum(v_selected == -1)
                bin_precisions[b] = np.mean(np.divide(tp, np.add(tp, fp)))
                bin_confidences[b] = np.mean(y_score[selected])
                bin_counts[b] = len(selected)

        if category is None:
            precisions = []
            for c in self.dataset.get_categories_names():
                precisions.append(self._calculate_metric_for_category(c, Metrics.PRECISION_SCORE)["value"])
            avg_prec = mean(precisions)
        else:
            avg_prec = self._calculate_metric_for_category(category, Metrics.PRECISION_SCORE)["value"]
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

        gaps = bin_confidences - bin_precisions
        ece = np.sum(np.abs(gaps) * bin_counts) / np.sum(bin_counts)
        mce = np.max(np.abs(gaps))

        result = {'values': bin_precisions, 'gaps': gaps, 'counts': bin_counts, 'bins': bins,
                  'avg_value': avg_prec, 'avg_conf': avg_conf, 'ece': ece, 'mce': mce}
        return result

    def __support_reliability(self, matching, category_id=None):
        """
        Returns the confidences and the corresponding labels values from the matching pandas.DataFrame

        Parameters
        ----------
        matching: pandas.DataFrame
            Matching between ground truth and the predictions

        Returns
        -------
        confidences, labels
        """
        if category_id is None:
            match = matching.loc[matching["iou"] >= self._iou_thresh_strong]
        else:
            match = matching.loc[(matching["iou"] >= self._iou_thresh_strong) &
                                 (matching["category_det"] == category_id)]

        numpy_confidence = match["confidence"].values
        numpy_label = match["label"].values
        return numpy_confidence, numpy_label

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
        input_report = {"total": {}}
        anns = self.dataset.get_annotations()
        proposals = self.dataset.get_proposals(self._model_name)
        if not self.matching_dict or "all" not in self.matching_dict.keys():
            self.matching_dict["all"] = self._match_detection_with_ground_truth(anns, proposals, self._iou_thresh_strong).copy()
        all_matching = self.matching_dict["all"].copy()

        input_report["total"]["all"] = {"anns": anns,
                                        "props": proposals,
                                        "matching": all_matching}
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
        results = {}
        cat_metric_values = {}

        self._set_normalized_number_of_images_for_categories()
        # total
        user_normalizations = self._use_normalization
        self._use_normalization = False
        micro_value, _ = self._compute_metric(input_report["total"]["all"]["anns"],
                                                 input_report["total"]["all"]["props"],
                                                 input_report["total"]["all"]["matching"],
                                                 metric, is_micro_required=True)
        self._use_normalization = user_normalizations
        macro_values = []
        for cat in self.dataset.get_categories_names():
            value = self._calculate_metric_for_category(cat, metric)['value']

            cat_metric_values[cat] = value
            macro_values.append(value)
        results["avg macro"] = mean(macro_values)
        results["avg micro"] = micro_value

        # categories
        if show_categories:
            for cat in categories:
                results[cat] = cat_metric_values[cat]

        # properties
        if show_properties:
            if not self.matching_dict or "all" not in self.matching_dict.keys():
                self.matching_dict["all"] = self._match_detection_with_ground_truth(self.dataset.get_annotations(),
                                                                                    self.dataset.get_proposals(),
                                                                                    self._iou_thresh_strong).copy()
            matching = self.matching_dict["all"].copy()

            for prop in properties:
                if self._use_new_normalization:
                    self._set_normalized_number_of_images_for_property_for_categories(prop)
                else:
                    self._set_normalized_number_of_images_for_categories()
                p_values = self.dataset.get_values_for_property(prop)
                for p_value in p_values:
                    macro_values = []
                    for cat in self.dataset.get_categories_names():
                        cat_id = self.dataset.get_category_id_from_name(cat)
                        value = self._calculate_metric_for_properties_of_category(cat, cat_id, prop, [p_value], matching, metric)[
                            p_value]['value']

                        macro_values.append(value)
                    p_value = prop + "_" + "{}".format(p_value)
                    results[p_value] = mean(macro_values)
        return results

    # -- Normalization support functions -- #

    def _get_normalized_number_of_images(self):
        """
        Returns
        -------
        normalized_number_of_images: float
            Number of images normalized
        """
        return self.__normalized_number_of_images

    def _set_normalized_number_of_images_for_categories(self):
        """
        Normalizes the number of images considering only the categories
        """
        self.__normalized_number_of_images = self._norm_factors["categories"] * \
                                             self.dataset.get_number_of_images()

    def _set_normalized_number_of_images_for_property_for_categories(self, property_name):
        """
        Normalizes the number of images considering the categories and the property

        Parameters
        ----------
        property_name:
            Property to consider in the normalization
        """
        if property_name not in self._norm_factors:
            self.update_property_normalization_factor(property_name)
        self.__normalized_number_of_images = self._norm_factors[property_name] * \
                                             self._norm_factors["categories"] * \
                                             self.dataset.get_number_of_images()
