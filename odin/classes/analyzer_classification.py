import copy
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from abc import ABC

from sklearn.metrics import auc, multilabel_confusion_matrix, confusion_matrix

from odin.classes import DatasetClassification, TaskType
from odin.classes.analyzer_interface import AnalyzerInterface
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_false_positive_errors, plot_reliability_diagram, \
    display_confusion_matrix, display_confusion_matrix_categories, pie_plot, plot_class_distribution

logger = get_root_logger()


class AnalyzerClassification(AnalyzerInterface, ABC):
    """Class to perform analysis for classification problems"""

    __SAVE_PNG_GRAPHS = True  # if set to false graphs will just be displayed

    __normalized_number_of_images = 1000

    __valid_metrics = ['accuracy', 'precision_score', 'recall_score', 'f1_score', 'average_precision_score', 'roc_auc',
                       'precision_recall_auc', 'f1_auc', 'custom']

    __valid_curves = ['precision_recall_curve', 'roc_curve', 'f1_curve']

    def __init__(self, detector_name, dataset, result_saving_path='./results/', use_normalization=False,
                 norm_factor_categories=None, norm_factors_properties=None, conf_thresh=None, metric='f1_score',
                 save_graphs_as_png=True):
        """
        Parameters
        ----------
        detector_name: string
            Name of the detector. Used as folder name for saving the results.
        dataset: DatasetClassification
            Dataset used to perform the analysis
        result_saving_path: string, optional
            The path to save results
        use_normalization: bool, optional
            Specifies whether or not to use the normalization (default is False)
        norm_factor_categories: float, optional
            Categories normalization factor (default is 1/number of classes)
        norm_factors_properties: list of pairs, optional
            Properties normalization factors.

            Each pair (property name, normalization factor value) specifies the normalization factor to apply to a
            specific property.
            The default value for each property is 1/number of property values
        conf_thresh: float, optional
            Confidence threshold. Predictions with a confidence value less than the threshold are ignored.

            For single-label problem the default value is 0.
            For binary and multi-label problems the default value is 0.5
        metric: string
            Default metric for analysis (default is f1_score)
        save_graphs_as_png: bool
            Specifies whether or not to save the graphs as .png images (default is True)
        """
        if type(dataset) is not DatasetClassification:
            raise TypeError("Invalid type for 'dataset'. Use DatasetClassification.")
        if not dataset.get_property_keys():
            raise Exception("No properties available. Please make sure to load the properties to the dataset.")

        if conf_thresh is None:
            if dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
                conf_thresh = 0
            else:
                conf_thresh = 0.5

        if dataset.classification_type == TaskType.CLASSIFICATION_BINARY:
            self.__is_binary = True
        else:
            self.__is_binary = False

        self.__SAVE_PNG_GRAPHS = save_graphs_as_png

        super().__init__(detector_name, dataset, result_saving_path, use_normalization, norm_factor_categories,
                         norm_factors_properties, conf_thresh, metric, self.__valid_metrics, self.__valid_curves,
                         self.__is_binary, self.__SAVE_PNG_GRAPHS)


    def analyze_false_positive_error_for_category(self, category, categories=None, metric=None):
        if category not in self.dataset.get_categories_names():
            logger.error("Invalid category name")
            return

        if self.__is_binary:
            logger.error("Not supported for binary classification")
            return

        if categories is None or category not in categories:
            categories = [category]
        elif not self._are_valid_categories(categories):
            return
        if metric is None:
            metric = self.metric
        elif not self._is_valid_metric(metric):
            return

        error_dict_total = self._analyze_false_positive_errors(categories)
        error_dict = error_dict_total[category]
        values = self._calculate_metric_for_category(category, metric)

        category_metric_value = values['value']
        errors = ["similar", "without_gt", "generic"]
        error_values = []

        for error in errors:
            if len(error_dict[error]) == 0:
                error_values.append([category_metric_value, 0])
                continue
            images = self.dataset.get_all_observations()
            proposals = self.dataset.get_proposals()
            props = proposals[~proposals["id"].isin(error_dict[error])]
            y_true, y_score = self.__convert_input_format_for_category(images, props, category)

            self._set_normalized_number_of_images_for_categories()
            metric_value, _ = self._compute_metric(y_true, y_score, metric, None)

            count_error = len(error_dict[error])
            error_values.append([metric_value, count_error])

        plot_false_positive_errors(error_values, errors, category_metric_value, category, metric,
                                   self.result_saving_path, self.__SAVE_PNG_GRAPHS)

    def analyze_reliability(self, categories=None, num_bins=10):
        if num_bins < 2:
            logger.error("Minimum number of bins is 2")
            return
        if num_bins > 50:
            logger.error("Maximum number of bins is 50")
            return
        images = self.dataset.get_all_observations()
        proposals = self.dataset.get_proposals()
        if self.__is_binary:
            categories = None
        if categories is None:
            y_true, y_pred, y_score = self.__convert_input_reliability(images, proposals)
            result = self._calculate_reliability(y_true, y_pred, y_score, num_bins)
            plot_reliability_diagram(result, self.__SAVE_PNG_GRAPHS, self.result_saving_path, is_classification=True)
        else:
            if not self._are_valid_categories(categories):
                return
            for category in categories:
                y_true, y_pred, y_score = self.__convert_input_reliability(images, proposals, category)
                result = self._calculate_reliability(y_true, y_pred, y_score, num_bins)
                plot_reliability_diagram(result, self.__SAVE_PNG_GRAPHS, self.result_saving_path,
                                         is_classification=True, category=category)

    def analyze_confusion_matrix(self, categories=None, properties_names=None, properties_values=None):
        if self.__is_binary:
            categories = [self.dataset.get_categories_names()[0]]
        else:
            if categories is None:
                categories = self.dataset.get_categories_names()
            elif not self._are_valid_categories(categories):
                return
        if properties_names is not None:
            if not self._are_valid_properties(properties_names):
                return
            elif properties_values is not None:
                if len(properties_names) != len(properties_values):
                    logger.error("Inconsistency between properties_names and properties_values")
                    return
                for i, p in enumerate(properties_names):
                    if not self._is_valid_property(p, properties_values[i]):
                        return
        properties_filter = self.__get_properties_filter_from_names_and_values(properties_names, properties_values)
        images = self.dataset.get_all_observations()
        proposals = self.dataset.get_proposals()
        y_true, y_pred, labels = self.__convert_input_confusion_matrix(images, proposals, categories, properties_filter)
        if len(y_true) == 0:
            logger.warn("No images found")
            return
        result = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        display_confusion_matrix(result, categories, properties_filter, self.__SAVE_PNG_GRAPHS, self.result_saving_path)

        if self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL and len(categories) > 1:
            if len(categories) < len(self.dataset.get_categories_names()):
                labels.append(-1)
                categories.append('other')
            result = confusion_matrix(y_true, y_pred, labels=labels)
            display_confusion_matrix_categories(result, categories, properties_filter, self.__SAVE_PNG_GRAPHS,
                                                self.result_saving_path)

    def base_report(self, metrics=None, categories=None, properties=None, show_categories=True,
                    show_properties=True):
        default_metrics = ['accuracy', 'precision_score', 'recall_score', 'f1_score', 'average_precision_score']
        return self._get_report_results(default_metrics, metrics, categories, properties, show_categories,
                                        show_properties)

    def show_distribution_of_property(self, property_name):
        if property_name not in self.dataset.get_property_keys():
            logger.error(f"Property '{property_name}' not valid.")
            return
        property_name_to_show = self.dataset.get_display_name_of_property(property_name)

        values = self.dataset.get_values_for_property(property_name)
        display_names = [self.dataset.get_display_name_of_property_value(property_name, v) for v in values]

        images = self.dataset.get_all_observations()
        count = images.groupby(property_name).size()
        p_values = count.index.tolist()
        sizes = []
        for pv in values:
            if pv not in p_values:
                sizes.append(0)
            else:
                sizes.append(count[pv])

        title = "Distribution of {}".format(property_name_to_show)
        output_path = os.path.join(self.result_saving_path, "distribution_total_{}.png".format(property_name))
        pie_plot(sizes, display_names, title, output_path, self.__SAVE_PNG_GRAPHS)

        if self.dataset.classification_type == TaskType.CLASSIFICATION_BINARY:
            return

        labels = self.dataset.get_categories_names()
        for pv in values:
            sizes = []
            for cat_name in self.dataset.get_categories_names():
                cat_id = self.dataset.get_category_id_from_name(cat_name)
                sizes.append(len(self.dataset.get_observations_from_property_category(cat_id, property_name, pv).index))

            title = "Distribution of {} among categories".format(pv)
            output_path = os.path.join(self.result_saving_path, "distribution_{}_in_categories.png".format(pv))
            pie_plot(sizes, labels, title, output_path, self.__SAVE_PNG_GRAPHS)

    def get_tn_distribution(self, categories=None):
        if self.__is_binary:
            logger.error("Not supported for binary classification")
            return
        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not self._are_valid_categories(categories):
            return
        if categories is not None:
            tp_classes = self._analyze_true_negative_for_categories(categories)
            plot_class_distribution(tp_classes, self.result_saving_path, self.__SAVE_PNG_GRAPHS, "True Negative")

    def _get_normalized_number_of_images(self):
        """
        Returns
        -------
        normalized_number_of_images: float
            Number of images normalized
        """
        return self.__normalized_number_of_images

    def _set_normalized_number_of_images_for_categories(self):
        self.__normalized_number_of_images = self._norm_factors["categories"] * \
                                             self.dataset.get_number_of_observations()

    def _set_normalized_number_of_images_for_property_for_categories(self, property_name):
        self.__normalized_number_of_images = self._norm_factors[property_name] * \
                                             self._norm_factors["categories"] * \
                                             self.dataset.get_number_of_observations()

    def _calculate_metric_for_category(self, category, metric):
        images = self.dataset.get_all_observations()
        proposals = self.dataset.get_proposals()
        y_true, y_score = self.__convert_input_format_for_category(images, proposals, category)
        self._set_normalized_number_of_images_for_categories()
        result, std_err = self._compute_metric(y_true, y_score, metric, None)
        return {"value": result, "std": std_err, "matching": None}

    def _compute_metric(self, gt, detections, metric, matching, is_micro_required=False):
        if metric == 'accuracy':
            return self.__compute_metric_accuracy(gt, detections)
        elif metric == 'precision_score':
            return self._compute_metric_precision_score(gt, detections, matching)
        elif metric == 'recall_score':
            return self._compute_metric_recall_score(gt, detections, matching)
        elif metric == 'f1_score':
            return self._compute_metric_f1_score(gt, detections, matching)
        elif metric == 'roc_auc':
            return self._compute_roc_auc_curve(gt, detections)
        elif metric == 'precision_recall_auc':
            return self._compute_precision_recall_auc_curve(gt, detections, matching)
        elif metric == 'f1_auc':
            return self._compute_f1_auc_curve(gt, detections, matching)
        elif metric == 'average_precision_score':
            return self._compute_average_precision_score(gt, detections, matching)
        elif metric == 'custom':
            return self._evaluation_metric(gt, detections, matching, is_micro_required)
        else:
            raise NotImplementedError("metric '{}' unknown".format(metric))

    def _calculate_metric_for_properties_of_category(self, category_name, category_id, property_name, possible_values,
                                                     matching, metric):
        skip = False
        if property_name in self.saved_results[metric][category_name].keys():
            skip = True
            for p_value in list(self.saved_results[metric][category_name][property_name].keys()):
                if p_value not in possible_values:
                    self.saved_results[metric][category_name][property_name].pop(p_value, None)
            for p_value in possible_values:
                if p_value not in list(self.saved_results[metric][category_name][property_name].keys()):
                    skip = False
                    break
                else:
                    tmp_result = self.saved_results[metric][category_name][property_name].pop(p_value, None)
                    self.saved_results[metric][category_name][property_name][p_value] = tmp_result
        if skip:
            return self.saved_results[metric][category_name][property_name]

        if self.use_new_normalization:
            self._set_normalized_number_of_images_for_property_for_categories(property_name)
        else:
            self._set_normalized_number_of_images_for_categories()
        properties_results = {}
        proposals = self.dataset.get_proposals()
        for value in possible_values:
            images = self.dataset.get_observations_from_property(property_name, value)
            y_true, y_score = self.__convert_input_format_for_category(images, proposals, category_name)
            metricvalue, std_err = self._compute_metric(y_true, y_score, metric, None)
            if math.isnan(metricvalue):
                metricvalue = 0
            properties_results[value] = {"value": metricvalue, "std": std_err}
        return properties_results

    def _analyze_false_negative_for_categories(self, categories):
        fn_classes = defaultdict(int)
        observations = self.dataset.get_all_observations()
        for category in categories:
            cat_id = self.dataset.get_category_id_from_name(category)
            preds = self.dataset.get_proposals().copy(deep=True)

            if self.dataset.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
                preds = preds[(preds["category_id"] == cat_id) & (preds["confidence"] >= self.conf_thresh)]
                preds_img_ids = preds[self.dataset.match_param_props]
                obs_no_img_ids = observations[~observations[self.dataset.match_param_gt].isin(preds_img_ids)].copy(deep=True)
                fn_obs = obs_no_img_ids[obs_no_img_ids["categories"].apply(lambda x: cat_id in x)]
                fn_classes[category] = len(fn_obs.index)
            elif self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
                preds = preds.sort_values(by="confidence", ascending=False).groupby(self.dataset.match_param_props).first()
                preds_img_ids = preds[(preds["category_id"] == cat_id) &
                                      (preds["confidence"] >= self.conf_thresh)].index
                obs_no_img_ids = observations[~observations[self.dataset.match_param_gt].isin(preds_img_ids)].copy(deep=True)
                fn_obs = obs_no_img_ids[obs_no_img_ids["category"] == cat_id]
                fn_classes[category] = len(fn_obs.index)
        return fn_classes

    def _analyze_true_positive_for_categories(self, categories):
        tp_classes = defaultdict(int)
        observations = self.dataset.get_all_observations()
        for category in categories:
            cat_id = self.dataset.get_category_id_from_name(category)
            preds = self.dataset.get_proposals().copy(deep=True)

            if self.dataset.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
                preds = preds[(preds["category_id"] == cat_id) & (preds["confidence"] >= self.conf_thresh)]
                match = pd.merge(observations, preds, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan,
                                                                                                                0)
                match = match[(match["category_id"] == cat_id) & (match["categories"].apply(lambda x: cat_id in x))]
                tp_classes[category] = len(match.index)
            elif self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
                preds = preds.sort_values(by="confidence", ascending=False).groupby(self.dataset.match_param_props).first()
                match = pd.merge(observations, preds, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan,
                                                                                                               0)
                match = match[(match["category_id"] == cat_id) & (match["confidence"] >= self.conf_thresh) &
                              (match["category"] == cat_id)]
                tp_classes[category] = len(match.index)
        return tp_classes

    def _analyze_true_negative_for_categories(self, categories):
        tn_classes = defaultdict(int)
        observations = self.dataset.get_all_observations()
        for category in categories:
            cat_id = self.dataset.get_category_id_from_name(category)
            preds = self.dataset.get_proposals().copy(deep=True)

            if self.dataset.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
                preds = preds[(preds["category_id"] == cat_id) & (preds["confidence"] >= self.conf_thresh)]
                preds_img_ids = preds[self.dataset.match_param_props]
                obs_no_img_ids = observations[~observations[self.dataset.match_param_gt].isin(preds_img_ids)].copy(deep=True)
                tn_obs = obs_no_img_ids[obs_no_img_ids["categories"].apply(lambda x: cat_id not in x)]
                tn_classes[category] = len(tn_obs.index)
            elif self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
                preds = preds.sort_values(by="confidence", ascending=False).groupby(self.dataset.match_param_props).first()
                preds_img_ids = preds[(preds["category_id"] == cat_id) &
                                      (preds["confidence"] >= self.conf_thresh)].index
                obs_no_img_ids = observations[~observations[self.dataset.match_param_gt].isin(preds_img_ids)].copy(deep=True)
                tn_obs = obs_no_img_ids[obs_no_img_ids["category"] != cat_id]
                tn_classes[category] = len(tn_obs.index)
        return tn_classes

    def _analyze_false_positive_errors(self, categories):
        skip = False
        if self.fp_errors is not None:
            skip = True
            if len(self.fp_errors.keys()) != len(categories):
                skip = False
            if skip:
                for c in categories:
                    if c not in self.fp_errors.keys():
                        skip = False
                        break
        if skip:
            return self.fp_errors

        self.fp_errors = {}
        fp_classes = defaultdict(int)
        images = self.dataset.get_all_observations().copy(deep=True)
        for category in categories:
            cat_id = self.dataset.get_category_id_from_name(category)
            preds = self.dataset.get_proposals().copy(deep=True)

            similar_indexes, without_gt_indexes, other_indexes = [], [], []

            if self.dataset.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
                preds = preds[(preds["category_id"] == cat_id) & (preds["confidence"] >= self.conf_thresh)]
                match = pd.merge(images, preds, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan,
                                                                                                               0)
                match = match[(match["category_id"] == cat_id) & (match["categories"].apply(lambda x: cat_id not in x))]

                fp_classes[category] = len(match.index)

                match_no_gt = match[match["categories"].str.len() == 0]
                if not match_no_gt.empty:
                    without_gt_indexes = match_no_gt["id_y"].values
                match = match[match["categories"].str.len() != 0]
                match["similar"] = np.where(match["categories"].apply(
                    lambda x: self.dataset.is_similar_classification(cat_id, x)), 1, 0)
                similar_indexes = match[match["similar"] == 1]["id_y"].values
                other_indexes = match[match["similar"] == 0]["id_y"].values
            elif self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
                preds = preds.sort_values(by="confidence", ascending=False).groupby(self.dataset.match_param_props).first()
                match = pd.merge(images, preds, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan,
                                                                                                               0)
                match = match[(match["category_id"] == cat_id) & (match["confidence"] >= self.conf_thresh) &
                              (match["category"] != cat_id)]

                fp_classes[category] = len(match.index)

                match["similar"] = np.where(match["category"].apply(
                    lambda x: self.dataset.is_similar_classification(cat_id, x)), 1, 0)
                similar_indexes = match[match["similar"] == 1]["id_y"].values
                other_indexes = match[match["similar"] == 0]["id_y"].values
            else:
                preds = preds[preds["category_id"] == cat_id]
                match = pd.merge(images, preds, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan,
                                                                                                               0)
                match = match[(match["category_id"] == cat_id) & (match["confidence"] >= self.conf_thresh) &
                              (match["category"] != cat_id)]
                other_indexes = match["id_y"].values

            self.fp_errors[category] = {"similar": similar_indexes,
                                        "without_gt": without_gt_indexes,
                                        "generic": other_indexes}
        self.fp_errors["distribution"] = fp_classes
        return self.fp_errors

    def _get_input_report(self, properties, show_properties_report):
        input_report = {}
        images = self.dataset.get_all_observations()
        proposals = self.dataset.get_proposals()

        input_report["total"] = self.__support_get_input_report(images, proposals)

        if show_properties_report:
            for property in properties:
                property_values = self.dataset.get_values_for_property(property)
                for p_value in property_values:
                    images = self.dataset.get_observations_from_property(property, p_value)
                    prop_value = property + "_" + "{}".format(p_value)
                    input_report[prop_value] = self.__support_get_input_report(images, proposals)

        return input_report

    def _calculate_report_for_metric(self, input_report, categories, properties, show_categories, show_properties,
                                     metric):
        warn_metrics = ['average_precision_score']
        results = {}
        cat_metric_values = {}

        # total
        self._set_normalized_number_of_images_for_categories()
        user_normalization = self._use_normalization
        if not self.__is_binary:
            self._use_normalization = False
        if self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL and metric in warn_metrics:
            micro_value = "not supported"
        else:
            micro_value, _ = self._compute_metric(input_report["total"]["all"]["y_true"],
                                                     input_report["total"]["all"]["y_score"],
                                                     metric, None, is_micro_required=True)
        self._use_normalization = user_normalization
        if self.__is_binary:
            results["total"] = micro_value
        else:
            tot_value = 0
            counter = 0
            for cat in self.dataset.get_categories_names():
                value, _ = self._compute_metric(input_report["total"][cat]["y_true"],
                                                   input_report["total"][cat]["y_score"],
                                                   metric, None)
                tot_value += value
                cat_metric_values[cat] = value
                counter += 1
            results["avg macro"] = tot_value / counter
            results["avg micro"] = micro_value

        # categories
        if show_categories:
            for cat in categories:
                results[cat] = cat_metric_values[cat]

        # properties
        if show_properties:
            for prop in properties:
                if self.use_new_normalization:
                    self._set_normalized_number_of_images_for_property_for_categories(prop)
                else:
                    self._set_normalized_number_of_images_for_categories()
                p_values = self.dataset.get_values_for_property(prop)
                for p_value in p_values:
                    p_value = prop + "_" + "{}".format(p_value)
                    if self.__is_binary:
                        results[p_value], _ = self._compute_metric(input_report[p_value]["all"]["y_true"],
                                                                      input_report[p_value]["all"]["y_score"],
                                                                      metric, None)
                    else:
                        tot_value = 0
                        counter = 0
                        for cat in self.dataset.get_categories_names():
                            value, _ = self._compute_metric(input_report[p_value][cat]["y_true"],
                                                               input_report[p_value][cat]["y_score"],
                                                               metric, None)
                            tot_value += value
                            counter += 1
                        results[p_value] = tot_value / counter
        return results

    # For single-label and multi-label, the metric is always calculated with average='micro'
    def _compute_metric_precision_score(self, gt, detections, matching):
        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)
        if is_single_label:
            return self.__compute_metric_accuracy(gt, detections)
        tp, tp_norm, fp, _ = self._support_metric_threshold(np.sum(np.array(gt_ord) == 1),
                                                            self._get_normalized_number_of_images(),
                                                            gt_ord, det_ord, tp, fp, self.conf_thresh)

        precision, precision_norm = self._support_precision_score(tp, tp_norm, fp)

        if self._use_normalization:
            return precision_norm, None
        else:
            return precision, None

    # For single-label and multi-label, the metric is always calculated with average='micro'
    def _compute_metric_recall_score(self, gt, detections, matching):
        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)
        if is_single_label:
            return self.__compute_metric_accuracy(gt, detections)

        tp, tp_norm, fp, tn = self._support_metric_threshold(np.sum(np.array(gt_ord) == 1),
                                                             self._get_normalized_number_of_images(),
                                                             gt_ord, det_ord, tp, fp, self.conf_thresh)
        fn = np.sum(np.array(gt_ord) == 1) - tp

        recall, recall_norm = self._support_recall_score(tp, tp_norm, fn)
        if self._use_normalization:
            return recall_norm, None
        else:
            return recall, None

    # For single-label and multi-label, the metric is always calculated with average='micro'
    def _compute_metric_f1_score(self, gt, detections, matching):
        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)
        if is_single_label:
            return self.__compute_metric_accuracy(gt, detections)

        tp, tp_norm, fp, tn = self._support_metric_threshold(np.sum(np.array(gt_ord) == 1),
                                                             self._get_normalized_number_of_images(),
                                                             gt_ord, det_ord, tp, fp, self.conf_thresh)
        fn = np.sum(np.array(gt_ord) == 1) - tp

        f1, f1_norm = self._support_f1_score(tp, tp_norm, fp, fn)

        if self._use_normalization:
            return f1_norm, None
        else:
            return f1, None

    def _compute_curve_for_categories(self, categories, curve):
        images = self.dataset.get_all_observations()
        proposals = self.dataset.get_proposals()
        results = {}
        self._set_normalized_number_of_images_for_categories()
        for category in categories:
            y_true, y_score = self.__convert_input_format_for_category(images, proposals, category)
            if curve == "precision_recall_curve":
                x_values, y_values = self._compute_precision_recall_curve(y_true, y_score)
            elif curve == "roc_curve":
                x_values, y_values = self._compute_roc_curve(y_true, y_score)
            elif curve == "f1_curve":
                x_values, y_values = self._compute_f1_curve(y_true, y_score)
            else:
                raise ValueError("Invalid curve name.")
            auc_value = auc(x_values, y_values)
            results[category] = {'auc': auc_value,
                                 'x': x_values,
                                 'y': y_values}
        return results

    def _compute_roc_curve(self, gt, detections):
        if not (np.array(gt).ndim == 1 and np.array_equal(np.array(gt), np.array(gt).astype(bool)) and
                np.array(detections).ndim == 1):
            raise ValueError("Only binary input is supported.")
        gt_ord, det_ord, tp, tn, fp = self._support_metric(gt, detections, None)

        fpr, tpr, tpr_norm = self.__support_roc_auc(np.sum(np.array(gt) == 1),
                                                    self._get_normalized_number_of_images(),
                                                    det_ord, tp, tn, fp)
        if self._use_normalization:
            return fpr, tpr_norm
        else:
            return fpr, tpr

    def _compute_roc_auc_curve(self, gt, detections):
        fpr, tpr = self._compute_roc_curve(gt, detections)
        std_err = np.std(tpr) / np.sqrt(tpr.size)
        return auc(fpr, tpr), std_err

    def _compute_precision_recall_curve(self, gt, detections):
        if not (np.array(gt).ndim == 1 and np.array_equal(np.array(gt), np.array(gt).astype(bool)) and
                np.array(detections).ndim == 1):
            raise ValueError("Only binary input is supported.")
        gt_ord, det_ord, tp, _, fp = self._support_metric(gt, detections, None)
        precision, precision_norm, recall, recall_norm = self._support_precision_recall_auc(
            len(gt), np.sum(np.array(gt) == 1), self._get_normalized_number_of_images(),
            np.array(det_ord), tp, fp, True)
        if self._use_normalization:
            return recall_norm, precision_norm
        else:
            return recall, precision

    def _compute_precision_recall_auc_curve(self, gt, detections, matching):
        recall, precision = self._compute_precision_recall_curve(gt, detections)
        std_err = np.std(precision) / np.sqrt(precision.size)
        return auc(recall, precision), std_err

    def _compute_f1_curve(self, gt, detections):
        if not (np.array(gt).ndim == 1 and np.array_equal(np.array(gt), np.array(gt).astype(bool)) and
                np.array(detections).ndim == 1):
            raise ValueError("Only binary input is supported.")
        gt_ord, det_ord, tp, _, fp = self._support_metric(gt, detections, None)
        precision, precision_norm, recall, recall_norm, rel_indexes = self._support_precision_recall(
            np.sum(np.array(gt) == 1), self._get_normalized_number_of_images(), np.array(det_ord), tp, fp)

        return self._support_f1_curve(np.array(det_ord), precision, precision_norm, recall, recall_norm, rel_indexes)

    def _compute_f1_auc_curve(self, gt, detections, matching):
        thresholds, f1 = self._compute_f1_curve(gt, detections)
        std_err = np.std(f1) / np.sqrt(f1.size)
        return auc(thresholds, f1), std_err

    def __convert_input_format_for_category(self, images, proposals, category):
        cat_id = self.dataset.get_category_id_from_name(category)
        gt = images.copy(deep=True)
        preds = proposals.copy(deep=True)
        if self.dataset.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            gt[cat_id] = np.where(gt['categories'].apply(lambda x: cat_id in x), 1, 0)
            y_true = gt[cat_id].values
            preds["cat_conf"] = np.where(preds['category_id'] == cat_id, preds['confidence'], 0)
            preds = preds.sort_values(by="cat_conf", ascending=False).groupby(self.dataset.match_param_props).first()
            match = pd.merge(gt, preds, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan, 0)
            y_scores = match["cat_conf"].values
        elif self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            gt[cat_id] = np.where(gt['category'] == cat_id, 1, 0)
            y_true = gt[cat_id].values
            preds["cat_conf"] = np.where(preds['category_id'] == cat_id, preds['confidence'], 0)
            preds = preds.sort_values(by="confidence", ascending=False).groupby(self.dataset.match_param_props).first()
            match = pd.merge(gt, preds, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan, 0)
            y_scores = match["cat_conf"].values
        else:
            gt[cat_id] = np.where(gt['category'] == cat_id, 1, 0)
            y_true = gt[cat_id].values
            preds = preds[preds["category_id"] == cat_id]
            match = pd.merge(gt, preds, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan, 0)
            y_scores = match["confidence"].values
        return y_true, y_scores

    def _support_metric(self, gt, detections, matching):
        si = sorted(range(len(detections)), key=lambda k: detections[k], reverse=True)
        gt_ord = []
        det_ord = []
        for i in si:
            gt_ord.append(gt[i])
            det_ord.append(detections[i])
        tp = []
        tn = []
        fp = []
        for i in range(len(gt_ord)):
            # tp
            if gt_ord[i] == 1 and det_ord[i] > 0:
                tp.append(1)
            else:
                tp.append(0)
            # fp
            if gt_ord[i] == 0 and det_ord[i] > 0:
                fp.append(1)
            else:
                fp.append(0)
            # tn
            if gt_ord[i] == 0 and det_ord[i] == 0:
                tn.append(1)
            else:
                tn.append(0)
        tp = np.cumsum(tp)
        tn = np.sum(tn)
        fp = np.cumsum(fp)
        return gt_ord, det_ord, tp, tn, fp

    def __support_roc_auc(self, n_true_imgs, n_normalized, det_ord, tp, tn, fp):
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
        for i in range(0, len(fpr)):
            if fpr[i] == v_fpr:
                if tpr[i] == v_tpr:
                    continue
            v_fpr = fpr[i]
            v_tpr = tpr[i]
            indexes.append(i)

        fpr = fpr[indexes]
        tpr = tpr[indexes]
        tpr_norm = tpr_norm[indexes]

        return fpr, tpr, tpr_norm

    def _support_metric_threshold(self, n_true_gt, n_normalized, gt_ord, det_ord, tp, fp, threshold):
        det_ord = np.array(det_ord)
        tp[det_ord < threshold] = 0
        fp[det_ord < threshold] = 0
        try:
            tp = tp[tp > 0][-1]
        except IndexError:
            tp = 0
        try:
            fp = fp[fp > 0][-1]
        except IndexError:
            fp = 0
        det_ord[det_ord < self.conf_thresh] = 0
        det_ord[det_ord > 0] = 1
        tn = np.sum(np.array(gt_ord) == det_ord) - tp
        np.warnings.filterwarnings('ignore')
        try:
            tp_norm = tp * n_normalized / n_true_gt
        except ZeroDivisionError:
            tp_norm = 0

        return tp, tp_norm, fp, tn

    # Single-label:
    #   gt: list of categories indexes ([1, 2, 2, 5, 4, 3, ...])
    #   detections: 2d array with prediction scores for each sample ([0, 0, 0.45, 0], [0, 0, 0.45, 0], ...])
    # Multi-label:
    #   gt: 2d array with label indicator for each sample ([1, 1, 0, 0], [0, 1, 0, 1], ...)
    #   detections: 2d array with prediction scores for each sample ([0, 0.9, 0.45, 0], [0, 0, 0.45, 0], ...])
    # Return:
    #   y_true: 2d array with label indicator for each sample ([1, 1, 0, 0], [0, 1, 0, 1], ...)
    #   y_pred: 2d array with label indicator for each sample based on the threshold ([1, 1, 0, 0], [0, 1, 0, 1], ...)
    #   y_true_all: list of all the categories samples (binary)
    #   y_score_all: list of all samples scores
    # N.B. y_true_all and y_score_all used only for multi-label
    def __convert_input_ml_sl(self, gt, detections):
        det_copy = copy.deepcopy(detections)
        categories_index = np.array(self.dataset.get_categories_id_from_names(self.dataset.get_categories_names()))
        n_categories = categories_index.size
        y_true = []
        y_true_all = []
        y_pred = []
        y_score_all = []
        for i, v in enumerate(gt):
            if type(v) == np.int64:
                im_true = np.zeros(n_categories)
                im_true[np.where(categories_index == v)] = 1
                y_true.append(im_true)
            else:
                y_true.append(v)
                y_true_all.extend(v)
                y_score_all.extend(det_copy[i])
            im_score = det_copy[i]
            im_score[im_score < self.conf_thresh] = 0
            im_score[im_score > 0] = 1
            y_pred.append(im_score)
        return y_true, y_pred, y_true_all, y_score_all

    def __check_and_parse_input(self, gt, detections):
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

    def __compute_metric_accuracy(self, gt, detections):
        np.warnings.filterwarnings('ignore')
        # binary
        if np.array(gt).ndim == 1 and np.array_equal(np.array(gt), np.array(gt).astype(bool)) and \
                np.array(detections).ndim == 1:
            gt_ord, det_ord, tp, _, fp = self._support_metric(gt, detections, None)
            tp, tp_norm, fp, tn = self._support_metric_threshold(np.sum(np.array(gt_ord) == 1),
                                                                 self._get_normalized_number_of_images(),
                                                                 gt_ord, det_ord, tp, fp, self.conf_thresh)
            tn_fp_fn = len(gt) - tp

            try:
                accuracy = (tp + tn) / len(gt)
                accuracy_norm = (tp_norm + tn) / (tp_norm + tn_fp_fn)
            except ZeroDivisionError:
                accuracy = 0
                accuracy_norm = 0

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

    # same as sklearn (no interpolation)
    # For multi-label, the metric is always calculated with average='micro'
    def _compute_average_precision_score(self, gt, detections, matching):
        gt_ord, det_ord, tp, tn, fp, is_single_label = self.__check_and_parse_input(gt, detections)
        if is_single_label:
            logger.warn("Single-label not supported for average_precision_score metric")
            return 0, 0

        metric_value, std_err = self._support_average_precision(len(np.array(gt_ord)),
                                                                np.sum(np.array(gt_ord) == 1),
                                                                self._get_normalized_number_of_images(),
                                                                np.array(det_ord), tp, fp, True)
        return metric_value, std_err

    def _calculate_reliability(self, y_true, y_pred, y_score, num_bins):
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        indices = np.digitize(y_score, bins, right=True)

        bin_accuracies = np.zeros(num_bins, dtype=np.float)
        bin_confidences = np.zeros(num_bins, dtype=np.float)
        bin_counts = np.zeros(num_bins, dtype=np.int)

        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(y_true[selected] == y_pred[selected])
                bin_confidences[b] = np.mean(y_score[selected])
                bin_counts[b] = len(selected)

        avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

        gaps = bin_confidences - bin_accuracies
        ece = np.sum(np.abs(gaps) * bin_counts) / np.sum(bin_counts)
        mce = np.max(np.abs(gaps))

        result = {'values': bin_accuracies, 'gaps': gaps, 'counts': bin_counts, 'bins': bins,
                  'avg_value': avg_acc, 'avg_conf': avg_conf, 'ece': ece, 'mce': mce}
        return result

    def __convert_input_reliability(self, gt, proposals, category=None):
        images = gt.copy(deep=True)
        props = proposals.copy(deep=True)
        if self.dataset.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            match = pd.merge(images, props, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan, 0)
            if category is None:
                match = match[match["confidence"] > self.conf_thresh]
                match = match.assign(is_correct=[1 if a in b else 0 for a, b in zip(match["category_id"],
                                                                                    match["categories"])])
            else:
                cat_id = self.dataset.get_category_id_from_name(category)
                match = match[(match["confidence"] > self.conf_thresh) & (match["category_id"] == cat_id)]
                match["is_correct"] = np.where(match["categories"].apply(lambda x: cat_id in x), 1, 0)
            y_score = match["confidence"].values
            y_pred = match["is_correct"].values
            y_true = np.ones(y_pred.size)

        elif self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            props = props.sort_values(by="confidence", ascending=False).groupby(self.dataset.match_param_props).first()
            match = pd.merge(images, props, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan, 0)
            if category is None:
                match = match[match["confidence"] > 0]
            else:
                cat_id = self.dataset.get_category_id_from_name(category)
                match = match[(match["confidence"] > 0) & (match["category_id"] == cat_id)]
            y_score = match["confidence"].values
            y_pred = match["category_id"].values
            y_true = match["category"].values
        else:
            cat_id = self.dataset.get_category_id_from_name(self.dataset.get_categories_names()[0])
            props = props[props["category_id"] == cat_id]
            match = pd.merge(images, props, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan, 0)
            y_score = match["confidence"].values
            y_pred = match["category_id"].values
            y_true = match["category"].values

        return y_true, y_pred, y_score

    def __convert_input_confusion_matrix(self, gt, proposals, categories, properties_filter):
        images = gt.copy(deep=True)
        props = proposals.copy(deep=True)

        cat_ids = self.dataset.get_categories_id_from_names(categories)

        if properties_filter is not None:
            for p_name in properties_filter.keys():
                images = images[images[p_name].isin(properties_filter[p_name])]
            if images.empty:
                return [], [], cat_ids
        if self.dataset.classification_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            all_cat_ids = self.dataset.get_categories_id_from_names(self.dataset.get_categories_names())

            y_true = images["categories"].apply(lambda x: [1 if i in x else 0 for i in all_cat_ids]).values
            props = props[props["confidence"] >= self.conf_thresh]
            props = props.groupby(self.dataset.match_param_props)["category_id"].apply(list).reset_index(name="categories_prop")
            match = pd.merge(images, props, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props)

            no_proposals = np.zeros(len(all_cat_ids)).tolist()
            y_pred = match["categories_prop"].apply(lambda x: no_proposals if type(x) == float else
                                                    [1 if i in x else 0 for i in all_cat_ids]).values
            tmp_cat_ids = []
            for id in cat_ids:
                tmp_cat_ids.append(np.where(all_cat_ids == id)[0][0])
            cat_ids = tmp_cat_ids
        elif self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            props = props.sort_values(by="confidence", ascending=False).groupby(self.dataset.match_param_props).first()
            props["confusion_id"] = np.where((props["confidence"] >= self.conf_thresh) &
                                             (props["category_id"].isin(cat_ids)), props["category_id"], -1)
            match = pd.merge(images, props, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan, -1)
            y_true = match["category"].values
            y_pred = match["confusion_id"].values
        else:
            cat_id = cat_ids[0]
            images[cat_id] = np.where(images['category'] == cat_id, 1, 0)
            y_true = images[cat_id].values
            props = props[props["category_id"] == cat_id]
            props["confusion_id"] = np.where(props["confidence"] >= self.conf_thresh, 1, 0)
            match = pd.merge(images, props, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan, 0)
            y_pred = match["confusion_id"].values

        return y_true.tolist(), y_pred.tolist(), cat_ids

    def __get_properties_filter_from_names_and_values(self, properties_names, properties_values):
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

    def __match_id_confidence_sl(self, cat_id, confidence, size, cats_ids):
        try:
            cat_id = int(cat_id)
            index = np.zeros(size)
            index[np.where(cats_ids == cat_id)] = confidence
        except:
            index = np.zeros(size)
        return index

    def __match_id_confidence_ml(self, cat_ids, confidences, size, categories_ids):
        try:
            index = np.zeros(size)
            for i, cat_id in enumerate(cat_ids):
                cat_id = int(cat_id)
                index[np.where(categories_ids == cat_id)] = confidences[i]
        except:
            index = np.zeros(size)
        return index

    def __support_get_input_report(self, gt, proposals):
        images = gt.copy(deep=True)
        props = proposals.copy(deep=True)

        data = {}

        if self.dataset.classification_type == TaskType.CLASSIFICATION_BINARY:
            cat_id = self.dataset.get_categories_id_from_names(self.dataset.get_categories_names())[0]
            images[cat_id] = np.where(images['category'] == cat_id, 1, 0)
            y_true_micro = images[cat_id].values
            preds = props[props["category_id"] == cat_id]
            match = pd.merge(images, preds, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props).replace(np.nan, 0)
            y_score_micro = match["confidence"].values
            data["all"] = {"y_true": y_true_micro,
                           "y_score": y_score_micro}
            return data

        elif self.dataset.classification_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            y_true_micro = images["category"].values
            props = props.sort_values(by="confidence", ascending=False).groupby(self.dataset.match_param_props).first()
            match = pd.merge(images, props, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props)
            cats_ids = self.dataset.get_categories_id_from_names(self.dataset.get_categories_names())
            match["micro"] = match.apply(lambda x: self.__match_id_confidence_sl(x["category_id"], x["confidence"],
                                                                                 cats_ids.size, cats_ids), axis=1)
            y_score_micro = np.array(match["micro"].values.tolist())
            data["all"] = {"y_true": y_true_micro,
                           "y_score": y_score_micro}
            for index, cat in enumerate(self.dataset.get_categories_names()):
                cat_id = self.dataset.get_category_id_from_name(cat)
                tmp_macro = y_true_micro.copy()
                tmp_macro[tmp_macro != cat_id] = 0
                tmp_macro[tmp_macro == cat_id] = 1
                data[cat] = {"y_true": tmp_macro,
                             "y_score": y_score_micro[:, index]}
            return data

        else:
            cat_ids = self.dataset.get_categories_id_from_names(self.dataset.get_categories_names())
            y_true_micro = np.array(images["categories"].apply(lambda x: [1 if i in x else 0 for i in cat_ids]).
                                    values.tolist())
            props_categories = props.groupby(self.dataset.match_param_props)["category_id"].apply(list).reset_index(
                name="categories_prop")
            props_confidences = props.groupby(self.dataset.match_param_props)["confidence"].apply(list).reset_index(
                name="confidences_prop")
            props = pd.merge(props_categories, props_confidences, on=self.dataset.match_param_props)
            match = pd.merge(images, props, how="left", left_on=self.dataset.match_param_gt, right_on=self.dataset.match_param_props)
            match["micro"] = match.apply(
                lambda x: self.__match_id_confidence_ml(x["categories_prop"], x["confidences_prop"], cat_ids.size,
                                                        cat_ids), axis=1)
            y_score_micro = np.array(match["micro"].values.tolist())
            data["all"] = {"y_true": y_true_micro,
                           "y_score": y_score_micro}
            for index, cat in enumerate(self.dataset.get_categories_names()):
                data[cat] = {"y_true": y_true_micro[:, index],
                             "y_score": y_score_micro[:, index]}
            return data
