import abc
import os

import numpy as np
import pandas as pd

from odin.utils import get_root_logger
from odin.utils.draw_utils import make_multi_category_plot, display_sensitivity_impact_plot, \
    plot_categories_curve, plot_class_distribution

logger = get_root_logger()


class AnalyzerInterface(metaclass=abc.ABCMeta):

    __detector_name = "detector"
    result_saving_path = "./results/"

    dataset = None

    __valid_metrics = None
    __valid_curves = None
    metric = None

    # ONLY FOR TESTING, TO REMOVE
    use_new_normalization = True  # if True use the new implementation of normalization (categories + properties),
                                  # otherwise use the old one (only categories)

    _use_normalization = False
    _norm_factors = None
    normalizer_factor = 1

    conf_thresh = 0.5

    saved_results = {}
    fp_errors = None

    __SAVE_PNG_GRAPHS = True

    __is_binary = False

    def __init__(self, detector_name, dataset, result_saving_path, use_normalization, norm_factor_categories,
                 norm_factors_properties, conf_thresh, metric, valid_metrics, valid_curves, is_binary,
                 save_graphs_as_png):

        self.__detector_name = detector_name
        self.dataset = dataset

        if not os.path.exists(result_saving_path):
            os.mkdir(result_saving_path)
        self.result_saving_path = os.path.join(result_saving_path, detector_name)
        if not os.path.exists(self.result_saving_path):
            os.mkdir(self.result_saving_path)

        self._use_normalization = use_normalization
        self._norm_factors = self.__create_norm_factors_dict(norm_factor_categories, norm_factors_properties)
        self.conf_thresh = conf_thresh
        self.metric = metric
        self.__valid_metrics = valid_metrics
        self.__valid_curves = valid_curves
        self.__is_binary = is_binary
        self.__SAVE_PNG_GRAPHS = save_graphs_as_png

    def analyze_property(self, property_name, possible_values=None, labels=None, show=True, metric=None):
        """Analyzes the performances of the model for each category considering only the ground truth having a certain
        property value.

        Parameters
        ----------
        property_name: str
            Name of the property to analyze
        possible_values: list, optional
            Property values to be analyzed. If None consider all the possible values of the property. (default is None)
        labels: list, optional
            Property values names to show in the graph. If None use the display name in the properties file.
            (default is None)
        show: bool, optional
            If True results are shown in a graph. (default is True)
        metric: str, optional
            Metric used for the analysis. If None use the default metrics. (default is None)

        """
        if property_name not in self.dataset.get_property_keys():
            logger.error(f"Property '{property_name}' not valid")
            return
        if possible_values is None or not possible_values:
            possible_values = self.dataset.get_values_for_property(property_name)
        else:
            if not self._is_valid_property(property_name, possible_values):
                return
        if labels is None:
            labels = []
            for p in possible_values:
                display_name = self.dataset.get_display_name_of_property_value(property_name, p)
                if display_name is None:
                    labels.append(p)
                else:
                    labels.append(display_name)
        elif len(possible_values) != len(labels):
            logger.error("Inconsistency between number of possible values and labels.")
            return
        if metric is None:
            metric = self.metric
        elif not self._is_valid_metric(metric):
            return
        if metric not in self.saved_results.keys():
            self.saved_results[metric] = {}
        if self.__is_binary:
            categories = [self.dataset.get_categories_names()[0]]
        else:
            categories = self.dataset.get_categories_names()
        for category in categories:
            category_id = self.dataset.get_category_id_from_name(category)
            if category not in self.saved_results[metric].keys():
                self.saved_results[metric][category] = {}
                self.saved_results[metric][category]['all'] = self._calculate_metric_for_category(category,
                                                                                                  metric=metric)
            matching = self.saved_results[metric][category]['all']["matching"]
            self.saved_results[metric][category][property_name] = self._calculate_metric_for_properties_of_category(
                category, category_id, property_name, possible_values, matching, metric=metric)

            title = "Analysis of {} property".format(property_name)

        if show:
            make_multi_category_plot(self.saved_results[metric], property_name, labels, title, metric,
                                     self.__SAVE_PNG_GRAPHS, self.result_saving_path)

    def analyze_properties(self, properties=None, metric=None):
        """Analyzes the performances of the model for each category considering only the ground truth having a certain
        property value. The analysis is performed for all the properties specified in the parameters.

        Parameters
        ----------
        properties: list of str, optional
            Names of the properties to analyze. If None perform the analysis for all the properties. (default is None)
        metric: str
            Metric used for the analysis. If None use the default metrics. (default is None)
        """
        if properties is None:
            properties = self.dataset.get_property_keys()
        else:
            if not self._are_valid_properties(properties):
                return
        if metric is None:
            metric = self.metric
        elif not self._is_valid_metric(metric):
            return
        for pkey in properties:
            values = self.dataset.get_values_for_property(pkey)
            self.analyze_property(pkey, values, metric=metric)

    def show_distribution_of_properties(self, properties=None):
        """Shows the distribution of the property among its different values and for each property value shows the
        distribution among the categories.

        Parameters
        ----------
        properties: list of str, optional
            Names of the properties to analyze the distribution. If None perform the analysis for all the properties.
            (default is None)
        """
        if properties is None:
            properties = self.dataset.get_property_keys()
        elif not self._are_valid_properties(properties):
            return
        for property in properties:
            self.show_distribution_of_property(property)


    def analyze_sensitivity_impact_of_properties(self, properties=None, metric=None):
        """Analyzes the sensitivity and the impact of the properties specified in the parameters.

        Parameters
        ----------
        properties: list of str, optional
            Names of the properties to consider in the analysis. If None consider all the properties. (default is None)
        metric: str
            Metric used for the analysis. If None use the default metrics. (default is None)
        """
        if properties is None:
            properties = self.dataset.get_property_keys()
        else:
            if not self._are_valid_properties(properties):
                return
        display_names = [self.dataset.get_display_name_of_property(pkey) for pkey in properties]

        if metric is None:
            metric = self.metric
        elif not self._is_valid_metric(metric):
            return

        for pkey in properties:
            values = self.dataset.get_values_for_property(pkey)
            self.analyze_property(pkey, values, show=False, metric=metric)

        display_sensitivity_impact_plot(self.saved_results[metric], self.result_saving_path, properties,
                                        display_names, metric, self.__SAVE_PNG_GRAPHS)

    def get_tp_distribution(self, categories=None):
        if self.__is_binary:
            logger.error("Not supported for binary classification")
            return
        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not self._are_valid_categories(categories):
            return

        if categories is not None:
            tp_classes = self._analyze_true_positive_for_categories(categories)
            plot_class_distribution(tp_classes, self.result_saving_path, self.__SAVE_PNG_GRAPHS, "True Positive")

    def get_fn_distribution(self, categories=None):
        if self.__is_binary:
            logger.error("Not supported for binary classification")
            return
        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not self._are_valid_categories(categories):
            return

        if categories is not None:
            tp_classes = self._analyze_false_negative_for_categories(categories)
            plot_class_distribution(tp_classes, self.result_saving_path, self.__SAVE_PNG_GRAPHS, "False Negative")

    @abc.abstractmethod
    def _analyze_true_positive_for_categories(self, categories):
        pass

    @abc.abstractmethod
    def _analyze_false_negative_for_categories(self, categories):
        pass

    def get_fp_error_distribution(self, categories=None):
        if self.__is_binary:
            logger.error("Not supported for binary classification")
            return
        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not self._are_valid_categories(categories):
            return
        self.fp_errors = None
        error_dict_total = self._analyze_false_positive_errors(categories)
        plot_class_distribution(error_dict_total["distribution"], self.result_saving_path, self.__SAVE_PNG_GRAPHS,
                                "False Positive")

    def analyze_false_positive_errors(self, categories=None, metric=None):
        if self.__is_binary:
            logger.error("Not supported for binary classification")
            return
        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not self._are_valid_categories(categories):
            return
        if metric is None:
            metric = self.metric
        elif not self._is_valid_metric(metric):
            return
        if not self.__is_binary:
            self.get_fp_error_distribution(categories)
        for category in categories:
            self.analyze_false_positive_error_for_category(category, categories=categories, metric=metric)

    def analyze_curve_for_categories(self, categories=None, curve='precision_recall_curve'):
        if self.__is_binary:
            categories = [self.dataset.get_categories_names()[0]]
        else:
            if categories is None:
                categories = self.dataset.get_categories_names()
            elif not self._are_valid_categories(categories):
                return
        if not self.__is_valid_curve(curve):
            return
        results = self._compute_curve_for_categories(categories, curve)
        plot_categories_curve(results, curve, self.__SAVE_PNG_GRAPHS, self.result_saving_path)

    def set_normalization(self, use_normalization, with_properties=True, norm_factor_categories=None,
                          norm_factors_properties=None):
        """Sets the normalization for the metrics calculation

        Parameters
        ----------
        use_normalization: bool
            Specifies whether or not to use normalization
        with_properties: bool
            Specifies whether or not to normalize also on properties values
        norm_factor_categories: float, optional
            Categories normalization factor (default is 1/number of categories)
        norm_factors_properties: list of pairs, optional
            Properties normalization factors.

            Each pair specifies the normalization factor to apply to a specific property.
            (Example: [(name1, value1), (name2, value2), ...]
        """
        self._use_normalization = use_normalization
        if with_properties:
            self.use_new_normalization = True
        else:
            self.use_new_normalization = False
        if norm_factor_categories is not None:
            self._norm_factors["categories"] = norm_factor_categories
        if norm_factors_properties is not None:
            dataset_p_names = self.dataset.get_property_keys()
            for p_name, p_value in norm_factors_properties:
                if p_name in dataset_p_names:
                    self._norm_factors[p_name] = p_value
                else:
                    logger.warn("Invalid property name in 'norm_factors_properties'.")
        self.clear_saved_results()
        self.fp_errors = None

    def set_confidence_threshold(self, threshold):
        """Sets the threshold value. Predictions with a confidence lower than the threshold are ignored.

        Parameters
        ----------
        threshold: float
            Threshold value. Must be between 0 and 1
        """
        if threshold < 0 or threshold > 1:
            logger.error("Invalid threshold value.")
            return
        self.conf_thresh = threshold
        self.clear_saved_results()
        self.fp_errors = None

    def clear_saved_results(self, metrics=None):
        if metrics is None:
            self.saved_results = {}
        else:
            for m in metrics:
                if m in self.saved_results.keys():
                    self.saved_results[m] = {}
                else:
                    if self._is_valid_metric(m):
                        logger.warn(f"No data for metric {m}")

    def _get_report_results(self, default_metrics, metrics, categories, properties, show_categories, show_properties):
        if metrics is None:
            metrics = default_metrics
        else:
            for m in metrics:
                if m not in default_metrics and m != 'custom':
                    logger.error(
                        "Metric {} not supported for report. Available metrics: {}.".format(m, default_metrics))
                    return

        if self.__is_binary:
            show_categories = False
        else:
            if categories is None:
                categories = self.dataset.get_categories_names()
            elif not categories:
                logger.warn("Empty categories list")
                show_categories = False
            else:
                if not self._are_valid_categories(categories):
                    return

        if properties is None:
            properties = self.dataset.get_property_keys()
        elif not properties:
            logger.warn("Empty properties list")
            show_properties = False
        else:
            if not self._are_valid_properties(properties):
                return

        input_report = self._get_input_report(properties, show_properties)
        results = {}
        types = {}
        if self.__is_binary:
            types = {"total": "Total"}
        else:
            types["avg macro"] = "Total"
            types["avg micro"] = "Total"
        if show_categories:
            for cat in categories:
                types[cat] = "Category"
        if show_properties:
            for prop in properties:
                p_values = self.dataset.get_values_for_property(prop)
                for p_value in p_values:
                    p_value = prop + "_" + "{}".format(p_value)
                    types[p_value] = "Property"

        type_dict = {"type": types}

        for metric in metrics:
            results[metric] = self._calculate_report_for_metric(input_report, categories, properties, show_categories,
                                                                show_properties, metric)

        type_dataframe = pd.DataFrame(type_dict)

        data = pd.DataFrame(results)
        data = pd.merge(data, type_dataframe, left_index=True, right_index=True).reset_index()
        data = data.rename(columns={"index": "label"})
        data = data.set_index(["type", "label"])
        return data

    def _is_valid_metric(self, metric):
        if metric in self.__valid_metrics:
            return True
        logger.error(f"Metric '{metric}' not valid. Valid metrics: {self.__valid_metrics}")
        return False

    def _support_precision_score(self, tp, tp_norm, fp):
        np.warnings.filterwarnings('ignore')
        try:
            precision = tp / (tp + fp)
            precision_norm = tp_norm / (tp_norm + fp)
        except ZeroDivisionError:
            precision = 0
            precision_norm = 0

        if np.isnan(precision):
            precision = 0
            precision_norm = 0
        return precision, precision_norm

    def _support_recall_score(self, tp, tp_norm, fn):
        np.warnings.filterwarnings('ignore')
        try:
            recall = tp / (tp + fn)
            recall_norm = tp_norm / (tp_norm + fn)
        except ZeroDivisionError:
            recall = 0
            recall_norm = 0

        if np.isnan(recall):
            recall = 0
            recall_norm = 0
        return recall, recall_norm

    def _support_f1_score(self, tp, tp_norm, fp, fn):
        np.warnings.filterwarnings('ignore')
        precision, precision_norm = self._support_precision_score(tp, tp_norm, fp)
        recall, recall_norm = self._support_recall_score(tp, tp_norm, fn)
        try:
            f1 = 2 * precision * recall / (precision + recall)
            f1_norm = 2 * precision_norm * recall_norm / (precision_norm + recall_norm)
        except ZeroDivisionError:
            f1 = 0
            f1_norm = 0
        if np.isnan(f1):
            f1 = 0
            f1_norm = 0
        return f1, f1_norm

    def _support_precision_recall(self, n_anns, n_normalized, confidence, tp, fp):
        tp_norm = np.multiply(tp, n_normalized) / n_anns
        precision = np.true_divide(tp, np.add(tp, fp))
        recall = np.true_divide(tp, n_anns)
        fn = n_anns - tp[-1]

        np.warnings.filterwarnings('ignore')
        recall_norm = np.true_divide(tp_norm, tp_norm + fn)
        precision_norm = np.true_divide(tp_norm, np.add(tp_norm, fp))

        precision = np.nan_to_num(precision)
        precision_norm = np.nan_to_num(precision_norm)
        recall = np.nan_to_num(recall)
        recall_norm = np.nan_to_num(recall_norm)

        # same threshold, same value
        thresholds = np.unique(confidence)
        rel_indexes = []
        for t in thresholds:
            indexes = np.where(confidence == t)[0]
            for i in indexes:
                precision[i] = precision[indexes[-1]]
                precision_norm[i] = precision_norm[indexes[-1]]
                recall[i] = recall[indexes[-1]]
                recall_norm[i] = recall_norm[indexes[-1]]
            rel_indexes.append(indexes[0])

        return precision, precision_norm, recall, recall_norm, rel_indexes

    def _support_precision_recall_auc(self, n_gt, n_tot, n_normalized, confidence, tp, fp, is_classification):
        precision, precision_norm, recall, recall_norm, rel_indexes = self._support_precision_recall(n_tot,
                                                                                                     n_normalized,
                                                                                                     confidence, tp, fp)
        one = np.ones(1)
        zero = np.zeros(1)
        precision = np.concatenate([one, precision[np.sort(rel_indexes)]])
        precision_norm = np.concatenate([one, precision_norm[np.sort(rel_indexes)]])
        recall = np.concatenate([zero, recall[np.sort(rel_indexes)]])
        recall_norm = np.concatenate([zero, recall_norm[np.sort(rel_indexes)]])

        recall = np.flip(recall)
        recall_norm = np.flip(recall_norm)
        precision = np.flip(precision)
        precision_norm = np.flip(precision_norm)

        if recall[0] != 1:
            p_value = np.zeros(1)
            if is_classification:
                p_value[0] = n_tot / n_gt
            else:
                p_value[0] = 0  # set 0 in od
            recall = np.concatenate([one, recall])
            recall_norm = np.concatenate([one, recall_norm])
            precision = np.concatenate([p_value, precision])
            if is_classification:
                p_value[0] = n_normalized / (n_normalized + (n_gt - n_tot))
            else:
                p_value[0] = 0  # set 0 in od
            precision_norm = np.concatenate([p_value, precision_norm])

        indexes = []
        v_r = -1
        v_p = -1
        index_one_recall = -1
        for i in range(0, len(recall)):
            if recall[i] == 1:
                if precision[i] > v_p:
                    v_p = precision[i]
                    index_one_recall = i
                try:
                    if recall[i+1] != 1:
                        indexes.append(index_one_recall)
                except IndexError:
                    indexes.append(index_one_recall)
            else:
                if recall[i] == v_r:
                    if precision[i] == v_p:
                        continue
                v_r = recall[i]
                v_p = precision[i]
                indexes.append(i)

        recall = recall[indexes]
        precision = precision[indexes]
        recall_norm = recall_norm[indexes]
        precision_norm = precision_norm[indexes]
        return precision, precision_norm, recall, recall_norm

    def _support_f1_curve(self, det_ord, precision, precision_norm, recall, recall_norm, rel_indexes):
        f1 = 2 * np.divide(np.multiply(precision, recall), np.add(precision, recall))
        f1_norm = 2 * np.divide(np.multiply(precision_norm, recall_norm), np.add(precision_norm, recall_norm))
        thresholds = det_ord[np.sort(rel_indexes)]
        f1 = f1[np.sort(rel_indexes)]
        f1_norm = f1_norm[np.sort(rel_indexes)]

        one = np.ones(1)
        zero = np.zeros(1)
        if thresholds[0] != 1:
            thresholds = np.concatenate([one, thresholds])
            f1 = np.concatenate([zero, f1])
            f1_norm = np.concatenate([zero, f1_norm])
        if thresholds[-1] != 0:
            thresholds = np.concatenate([thresholds, zero])
            tmp_value = np.zeros(1)
            tmp_value[0] = f1[-1]
            f1 = np.concatenate([f1, tmp_value])
            tmp_value[0] = f1_norm[-1]
            f1_norm = np.concatenate([f1_norm, tmp_value])

        f1 = np.flip(f1)
        f1_norm = np.flip(f1_norm)
        thresholds = np.flip(thresholds)

        f1 = np.nan_to_num(f1)
        f1_norm = np.nan_to_num(f1_norm)

        if self._use_normalization:
            return thresholds, f1_norm
        else:
            return thresholds, f1

    def _support_average_precision(self, n_gt, n_tot, n_normalized, confidence, tp, fp, is_classification):
        precision, precision_norm, recall, recall_norm = self._support_precision_recall_auc(n_gt, n_tot, n_normalized,
                                                                                            confidence, tp, fp,
                                                                                            is_classification)
        std_err = np.std(precision) / np.sqrt(len(precision))
        std_err_norm = np.std(precision_norm) / np.sqrt(len(precision_norm))

        ap = -np.sum(np.multiply(np.diff(recall), precision[:-1]))
        ap_norm = -np.sum(np.multiply(np.diff(recall_norm), precision_norm[:-1]))

        if self._use_normalization:
            return ap_norm, std_err_norm
        else:
            return ap, std_err

    def _are_valid_categories(self, categories):
        if len(categories) == 0:
            logger.error(f"Empty categories list.")
            return False
        for c in categories:
            if c not in self.dataset.get_categories_names():
                logger.error(f"Category '{c}' not valid")
                return False
        return True

    def _are_valid_properties(self, properties):
        if len(properties) == 0:
            logger.error(f"Empty properties list.")
            return False
        for p in properties:
            if p not in self.dataset.get_property_keys():
                logger.error(f"Property '{p}' not valid.")
                return False
        return True

    def _is_valid_property(self, property_name, possible_values):
        if property_name in self.dataset.get_property_keys():
            if len(possible_values) == 0:
                logger.error(f"Empty possible values list")
                return False
            for value in possible_values:
                if value not in self.dataset.get_values_for_property(property_name):
                    logger.error(f"Property value '{value}' not valid for property '{property_name}'")
                    return False
        else:
            logger.error(f"Property '{property_name}' not valid")
            return False
        return True

    def __is_valid_curve(self, curve):
        if curve not in self.__valid_curves:
            logger.error(f"Curve '{curve}' not valid")
            return False
        return True

    def __create_norm_factors_dict(self, norm_factor_classes, norm_factors_properties):
        norm_factors = {}
        if norm_factor_classes is None:
            norm_factors["categories"] = 1 / len(self.dataset.get_categories_names())
        else:
            norm_factors["categories"] = norm_factor_classes
        if norm_factors_properties is None:
            for p_name in self.dataset.get_property_keys():
                norm_factors[p_name] = 1 / len(self.dataset.get_values_for_property(p_name))
        else:
            p_names = np.array(norm_factors_properties)[:, 0]
            dataset_p_names = self.dataset.get_property_keys()
            remaining_p_names = dataset_p_names - p_names
            for p_name, p_value in norm_factors_properties:
                if p_name in dataset_p_names:
                    norm_factors[p_name] = p_value
                else:
                    logger.warn("Invalid property name in 'norm_factors_properties'.")
            for p_name in remaining_p_names:
                norm_factors[p_name] = 1 / len(self.dataset.get_values_for_property(p_name))
        return norm_factors

    @abc.abstractmethod
    def analyze_false_positive_error_for_category(self, category, categories, metric=None):
        pass

    @abc.abstractmethod
    def show_distribution_of_property(self, property_name):
        pass

    @abc.abstractmethod
    def analyze_reliability(self, categories=None, num_bins=10):
        pass

    @abc.abstractmethod
    def _compute_curve_for_categories(self, categories, curve):
        pass

    @abc.abstractmethod
    def _get_normalized_number_of_images(self):
        pass

    @abc.abstractmethod
    def _set_normalized_number_of_images_for_categories(self):
        """Normalizes the number of images based on the normalization factor of the categories"""
        pass

    @abc.abstractmethod
    def _set_normalized_number_of_images_for_property_for_categories(self, property_name):
        """Normalizes the number of images based on the normalization factor of the specified property and the
        normalization factor of the categories

        Parameters
        ----------
        property_name: string
            Property name to which the images should be normalized
        """
        pass

    @abc.abstractmethod
    def _compute_metric(self, gt, detections, matching, metric, is_micro_required=False):
        pass

    def _evaluation_metric(self, gt, detections, matching, is_micro_required=False):
        raise NotImplementedError

    @abc.abstractmethod
    def _calculate_metric_for_category(self, category, metric):
        pass

    @abc.abstractmethod
    def _calculate_metric_for_properties_of_category(self, category_name, category_id, property_name, possible_values,
                                                     matching, metric):
        pass

    @abc.abstractmethod
    def _analyze_false_positive_errors(self, categories):
        pass

    @abc.abstractmethod
    def _get_input_report(self, properties, show_properties_report):
        pass

    @abc.abstractmethod
    def _calculate_report_for_metric(self, input_report, categories, properties, show_categories, show_properties,
                                     metric):
        pass

    @abc.abstractmethod
    def _compute_metric_precision_score(self, gt, detections, matching):
        pass

    @abc.abstractmethod
    def _compute_metric_recall_score(self, gt, detections, matching):
        pass

    @abc.abstractmethod
    def _compute_metric_f1_score(self, gt, detections, matching):
        pass

    @abc.abstractmethod
    def _compute_precision_recall_auc_curve(self, gt, detections, matching):
        pass

    @abc.abstractmethod
    def _compute_f1_auc_curve(self, gt, detections, matching):
        pass

    @abc.abstractmethod
    def _calculate_reliability(self, y_true, y_pred, y_score, num_bins):
        pass

    @abc.abstractmethod
    def _support_metric_threshold(self, n_true_gt, n_normalized, gt_ord, det_ord, tp, fp, threshold):
        pass

    @abc.abstractmethod
    def _compute_average_precision_score(self, gt, detections, matching):
        pass

    @abc.abstractmethod
    def _support_metric(self, gt, detections, matching):
        pass
