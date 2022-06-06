import abc
import os
from numbers import Number
from statistics import mean

import numpy as np
import pandas as pd

from aenum import extend_enum

from odin.classes import Metrics, Curves, TaskType, CustomMetric
from odin.classes.strings import *
from odin.utils import get_root_logger
from odin.utils.draw_utils import make_multi_category_plot, display_sensitivity_impact_plot, plot_multiple_curves, \
    plot_class_distribution

logger = get_root_logger()
np.warnings.filterwarnings('ignore')


class AnalyzerInterface(metaclass=abc.ABCMeta):

    _model_name = ""
    result_saving_path = "./results/"

    dataset = None

    _valid_metrics = None
    _valid_curves = None
    _valid_cams_metrics = None
    _metrics_with_threshold = [Metrics.ACCURACY, Metrics.PRECISION_SCORE, Metrics.RECALL_SCORE, Metrics.F1_SCORE]
    _custom_metrics = {}
    metric = None

    _use_new_normalization = True  # if True normalize also for properties, otherwise only for categories
    _use_normalization = False
    _norm_factors = None
    normalizer_factor = 1

    _conf_thresh = 0.5

    saved_results = {}
    saved_analyses = {}

    _SAVE_PNG_GRAPHS = True

    __is_binary = False

    def __init__(self, model_name, dataset, result_saving_path, use_normalization, norm_factor_categories,
                 norm_factors_properties, conf_thresh, metric):

        self._model_name = model_name
        self.dataset = dataset

        if self._SAVE_PNG_GRAPHS:
            if not os.path.exists(result_saving_path):
                os.mkdir(result_saving_path)
            self.result_saving_path = os.path.join(result_saving_path, model_name)
            if not os.path.exists(self.result_saving_path):
                os.mkdir(self.result_saving_path)

        self._use_normalization = use_normalization
        self._norm_factors = self.__create_norm_factors_dict(norm_factor_categories, norm_factors_properties)
        self._conf_thresh = conf_thresh
        self.metric = metric
        self.__is_binary = self.dataset.task_type == TaskType.CLASSIFICATION_BINARY
        self._custom_metrics = {}
        self.saved_results = {}
        self.saved_analyses = {}

    def add_custom_metric(self, custom_metric):
        """
        Add user custom metric
        Parameters
        ----------
        custom_metric: CustomMetric
            User custom metric
        """
        if not isinstance(custom_metric, CustomMetric):
            logger.error(err_type.format("custom_metric"))
            return -1

        if custom_metric.get_name() not in [item.value for item in Metrics]:
            extend_enum(Metrics, custom_metric.get_name().upper().replace(" ", "_"), custom_metric.get_name())

        if custom_metric.is_single_threshold() and Metrics(custom_metric.get_name()) not in self._metrics_with_threshold:
            self._metrics_with_threshold.append(Metrics(custom_metric.get_name()))

        self._custom_metrics[Metrics(custom_metric.get_name())] = custom_metric

        if Metrics(custom_metric.get_name()) not in self._valid_metrics:
            self._valid_metrics.append(Metrics(custom_metric.get_name()))

    def get_custom_metrics(self):
        """
        Returns all the custom metrics added by the user
        Returns
        -------
            dict
        """
        return self._custom_metrics

    def analyze_property(self, property_name, possible_values=None, categories=None, show=True, metric=None, split_by="meta-annotations", sort=True):
        """
        The model performances are analyzed for each category considering subsets of the data set which have a specific property value.

        Parameters
        ----------
        property_name: str
            Name of the property to be analyzed.
        possible_values: list, optional
            Property values to be analyzed. If not specified, all the property values are considered. (default is None)
        categories: list, optional
            List of the categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        metric: Metrics, optional
            Evaluation metric used for the analysis. If not specified, the default one is used. (default is None)
        split_by: str, optional
            If there are too many values, the plot is divided into subplots. The split can be performed by 'categories' or by 'meta-annotations'. default is 'meta-annotations')
        sort: bool, optional
            Indicates whether the property values should be sorted by the score achieved. (default is True)
        """
        if not self.dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset.")
            return -1

        if not isinstance(property_name, str):
            logger.error(err_type.format("property_name"))
            return -1
        if not self.dataset.are_valid_properties([property_name]):
            return -1

        if possible_values is None or not possible_values:
            possible_values = self.dataset.get_values_for_property(property_name)
        elif not isinstance(possible_values, list):
            logger.error(err_type.format("possible_values"))
            return -1
        elif not self.dataset.is_valid_property(property_name, possible_values):
            return -1

        if self.__is_binary:
            categories = [self.dataset.get_categories_names()[0]]
        elif categories is None or not categories:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        if metric is None:
            metric = self.metric
        elif not isinstance(metric, Metrics):
            logger.error(err_type.format("metric"))
            return -1
        elif not self._is_valid_metric(metric):
            return -1

        if not isinstance(split_by, str):
            logger.error(err_type.format("split_by"))
            return -1
        elif split_by not in ["meta-annotations", "categories"]:
            logger.error(err_value.format("split_by", "['meta-annotations', 'categories']"))
            return -1

        labels = []
        for p in possible_values:
            display_name = self.dataset.get_display_name_of_property_value(property_name, p)
            labels.append(p) if display_name is None else labels.append(display_name)

        results = {}
        for category in categories:
            # skip categories with no values for the property
            check_values = self.dataset.get_property_values_for_category(property_name, category)
            if not check_values or (len(check_values) == 1 and 'no value' in check_values):
                continue

            results[category] = {}
            category_id = self.dataset.get_category_id_from_name(category)

            results[category]['all'] = self._calculate_metric_for_category(category, metric=metric)
            matching = self.saved_results[metric][category]['all']["matching"]

            results[category][property_name] = self._calculate_metric_for_properties_of_category(category, category_id,
                                                                                                 property_name,
                                                                                                 possible_values,
                                                                                                 matching,
                                                                                                 metric=metric)

            title = "Analysis of {} property".format(property_name)

        if not show:
            return results
        if not results:
            logger.error("No available results")
            return -1
        make_multi_category_plot(results, property_name, labels,
                                 self.dataset.get_display_name_of_categories(), title, metric,
                                 self._SAVE_PNG_GRAPHS, self.result_saving_path, split_by=split_by, sort=sort)

    def analyze_properties(self, properties=None, categories=None, metric=None, split_by="meta-annotations", show=True, sort=True):
        """
        For each property, the model performances are analyzed for each category considering subsets of the data set which have a specific property value.

        Parameters
        ----------
        properties: list, optional
            List of properties to be included in the analysis. If not specified, all the properties are included. (default is None)
        categories: list, optional
            List of the categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        metric: Metrics, optional
            Evaluation metric used for the analysis. If not specified, the default one is used. (default is None)
        split_by: str, optional
            If there are too many values, the plot is divided into subplots. The split can be performed by 'categories' or by 'meta-annotations'. (default is 'meta-annotations')
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        sort: bool, optional
            Indicates whether the property values should be sorted by the score achieved. (default is True)
        """
        if not self.dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset.")
            return -1

        if properties is None:
            properties = self.dataset.get_property_keys()
        elif not isinstance(properties, list):
            logger.error(err_type.format("properties"))
            return -1
        elif not self.dataset.are_valid_properties(properties):
            return -1

        if self.__is_binary:
            categories = [self.dataset.get_categories_names()[0]]
        elif categories is None or not categories:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        if metric is None:
            metric = self.metric
        elif not isinstance(metric, Metrics):
            logger.error(err_type.format("metric"))
            return -1
        elif not self._is_valid_metric(metric):
            return -1

        if not isinstance(split_by, str):
            logger.error(err_type.format("split_by"))
            return -1
        elif split_by not in ["meta-annotations", "categories"]:
            logger.error(err_value.format("split_by", "['meta-annotations', 'categories']"))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        results = {}
        for pkey in properties:
            values = self.dataset.get_values_for_property(pkey)
            results[pkey] = self.analyze_property(pkey, values, categories, metric=metric, split_by=split_by, show=show,
                                                  sort=sort)

        if not show:
            return results

    def analyze_sensitivity_impact_of_properties(self, properties=None, metric=None, show=True, sort=True):
        """
        It provides the sensitivity of the model for each property and the impact that the latter could have on the overall performance of the model. The sensitivity to a property is the difference between the maximum and minimum score obtained for that meta-annotation. The impact of a property, instead, is the difference between the maximum score achieved for it and the overall score obtained by the model.

        Parameters
        ----------
        properties: list, optional
            List of properties to be included in the analysis. If not specified, all the properties are included. (default is None)
        metric: Metrics, optional
            Evaluation metric used for the analysis. If not specified, the default one is used. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        sort: bool, optional
            Indicates whether the properties should be sorted by the model sensitivity. (default is True)
        """
        if not self.dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset.")
            return -1

        if properties is None:
            properties = self.dataset.get_property_keys()
        elif not isinstance(properties, list):
            logger.error(err_type.format("properties"))
            return -1
        elif not self.dataset.are_valid_properties(properties):
            return -1

        if metric is None:
            metric = self.metric
        elif not isinstance(metric, Metrics):
            logger.error(err_type.format("metric"))
            return -1
        elif not self._is_valid_metric(metric):
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        property_result = {}
        for pkey in properties:
            values = self.dataset.get_values_for_property(pkey)
            property_result[pkey] = self.analyze_property(pkey, values, show=False, metric=metric)

        results = {}

        categories = self.dataset.get_categories_names()
        if self.__is_binary:
            categories = [categories[0]]
        for cat in categories:
            if cat not in self.saved_results[metric]:
                continue
            if cat not in results:
                results[cat] = {'all': self.saved_results[metric][cat]['all']}

            for pkey in properties:
                results[cat][pkey] = property_result[pkey][cat][pkey]

        if not show:
            return results

        display_names = [self.dataset.get_display_name_of_property(pkey) for pkey in properties]

        display_sensitivity_impact_plot(results, self.result_saving_path, properties,
                                        display_names, metric, self._SAVE_PNG_GRAPHS, sort)

    def analyze_false_negative_errors(self, categories=None, show=True):
        """
        It analyzes the false negative errors for each category, by identifying the type of the errors.
        Parameters
        ----------
        categories: list, optional
            List of the categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            logger.error("Analysis not supported for binary classification")
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

        results = {}
        for category in categories:
            self.analyze_false_negative_errors_for_category(category, show)
            results[category] = self.saved_analyses["false_negative_errors"][category]

        if not show:
            return results

    def analyze_false_positive_errors(self, categories=None, metric=None, show=True):
        """
        For each class, it analyzes the false positives by identifying the type of the errors and shows the gain that the model could achieve by removing all the false positives of each type.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        metric: Metrics, optional
            Evaluation metric used for the analysis. If not specified, the default one is used. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.__is_binary:
            logger.error("Not supported for binary classification")
            return -1

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        if metric is None:
            metric = self.metric
        elif not isinstance(metric, Metrics):
            logger.error(err_type.format("metric"))
            return -1
        elif not self._is_valid_metric(metric):
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        results = {}
        for category in categories:
            results[category] = self.analyze_false_positive_errors_for_category(category, metric=metric, show=show)

        if not show:
            return results

    def analyze_false_positive_trend(self, categories=None, include_correct_predictions=True, show=True):
        """
        For each class, it analyzes the trend of the false positives by indicating the percentage of each error type.
        Parameters
        ----------
        categories: list, optional
        List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        include_correct_predictions: bool, optional
            Indicates whether the correct detections should be included in the trend analysis or not. (default is True)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.__is_binary:
            logger.error("Not supported for binary classification")
            return -1

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        if not isinstance(include_correct_predictions, bool):
            logger.error(err_type.format("include_correct_detections"))
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        results = {}
        for category in categories:
            results[category] = self.analyze_false_positive_trend_for_category(category, include_correct_predictions, show)

        if not show:
            return results

    def analyze_curve(self, curve=Curves.PRECISION_RECALL_CURVE, average="macro", show=True):
        """
        It provides an overall analysis of the model performances by plotting the desired curve.

        Parameters
        ----------
        curve: Curves, optional
            Evaluation curve used for the analysis. (default is Curves.PRECISION_RECALL_CURVE)
        average: str, optional
            Indicates the averaging method. It can be 'macro' or 'micro'. (default is 'macro')
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not isinstance(curve, Curves):
            logger.error(err_type.format("curve"))
            return -1
        elif not self.__is_valid_curve(curve):
            return -1

        if not isinstance(average, str):
            logger.error(err_type.format("average"))
            return -1
        elif average not in ["micro", "macro"]:
            logger.error(err_value.format("average", "['micro', 'macro']"))
            return -1
        elif average == "micro" and self.dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            logger.error("average=micro not supported for single-label task")
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("show"))
            return -1

        if self.__is_binary:
            average = "micro"

        if average == "macro":
            results = self._compute_curve_for_categories(self.dataset.get_categories_names(), curve)
            auc_value = mean([results[c]["auc"] for c in results])
            x_values = np.arange(0, 1.001, 0.05).round(5)

            y_values_all = []
            for c in results:
                x = np.array(results[c]["x"])
                y = results[c]["y"]
                if curve == Curves.PRECISION_RECALL_CURVE:
                    x = np.flip(x)
                    y = np.flip(y)
                y_v = []
                for x_v in x_values:
                    y_index = np.where(x >= x_v)[0][0]
                    y_v.append(y[y_index])
                y_values_all.append(y_v)

            y_values = np.mean(np.array(y_values_all), axis=0)
            results = {'overall': {'auc': auc_value,
                                   'x': x_values,
                                   'y': y_values}}
        else:  # micro
            results = self._compute_curve_overall(curve)

        if not show:
            return results

        display_name = {'overall': {'display_name': 'overall'}}
        plot_multiple_curves(results, curve, display_name, self._SAVE_PNG_GRAPHS,
                             self.result_saving_path)

    def analyze_curve_for_categories(self, categories=None, curve=Curves.PRECISION_RECALL_CURVE, show=True):
        """
        For each category, it provides an analysis of the model performances by plotting the desired curve.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        curve: Curves, optional
            Evaluation curve used for the analysis. (default is Curves.PRECISION_RECALL_CURVE)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.__is_binary:
            categories = [self.dataset.get_categories_names()[0]]
        else:
            if categories is None:
                categories = self.dataset.get_categories_names()
            elif not isinstance(categories, list):
                logger.error(err_type.format("categories"))
                return -1
            elif not self.dataset.are_valid_categories(categories):
                return -1

        if not isinstance(curve, Curves):
            logger.error(err_type.format("curve"))
            return -1
        elif not self.__is_valid_curve(curve):
            return -1

        if not isinstance(show, bool):
            logger.error(err_type.format("bool"))
            return -1

        results = self._compute_curve_for_categories(categories, curve)

        if not show:
            return results

        plot_multiple_curves(results, curve, self.dataset.get_display_name_of_categories(), self._SAVE_PNG_GRAPHS,
                             self.result_saving_path)

    def show_true_positive_distribution(self, categories=None, show=True):
        """
        It provides the true positive distribution among the categories.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.__is_binary:
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

        tp_classes, _ = self._analyze_true_positive_for_categories(categories)

        if not show:
            return tp_classes

        labels = [self.dataset.get_display_name_of_category(cat) for cat in tp_classes.keys()]
        plot_class_distribution(tp_classes, labels, self.result_saving_path, self._SAVE_PNG_GRAPHS, "True Positive distribution")

    def show_false_negative_distribution(self, categories=None, show=True):
        """
        It provides the false negative distribution among the categories.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.__is_binary:
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

        tp_classes, _ = self._analyze_false_negative_for_categories(categories)

        if not show:
            return tp_classes

        labels = [self.dataset.get_display_name_of_category(cat) for cat in tp_classes.keys()]
        plot_class_distribution(tp_classes, labels, self.result_saving_path, self._SAVE_PNG_GRAPHS, "False Negative distribution")

    def show_true_positive_distribution_for_categories_for_property(self, property_name, property_values=None, categories=None, show=True):
        """
        It provides the true positive distribution of the property values for each category.

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
            categories = [self.dataset.get_category_name_from_id(1)] if self.__is_binary else self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type, "categories")
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        results = {}
        p_label = self.dataset.get_display_name_of_property(property_name)
        labels = [self.dataset.get_display_name_of_property_value(property_name, p_v) for p_v in property_values]
        for c in categories:
            tp_p, _ = self._analyze_true_positive_for_category_for_property(c, property_name, property_values)
            results[c] = tp_p

            if show:
                c_label = self.dataset.get_display_name_of_category(c)
                plot_class_distribution(tp_p, labels, self.result_saving_path, self._SAVE_PNG_GRAPHS, f"True Positive distribution of {p_label} for {c_label}")

        if not show:
            return results

    def show_false_negative_distribution_for_categories_for_property(self, property_name, property_values=None, categories=None, show=True):
        """
        It provides the false negative distribution of the property values for each category.

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
            categories = [self.dataset.get_category_name_from_id(1)] if self.__is_binary else self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type, "categories")
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        results = {}
        p_label = self.dataset.get_display_name_of_property(property_name)
        labels = [self.dataset.get_display_name_of_property_value(property_name, p_v) for p_v in property_values]
        for c in categories:
            fn_p, _ = self._analyze_false_negative_for_category_for_property(c, property_name, property_values)
            results[c] = fn_p

            if show:
                c_label = self.dataset.get_display_name_of_category(c)
                plot_class_distribution(fn_p, labels, self.result_saving_path, self._SAVE_PNG_GRAPHS, f"False Negative distribution of {p_label} for {c_label}")

        if not show:
            return results

    def show_false_positive_distribution(self, categories=None, show=True):
        """
        It provides the false positive distribution among the categories.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if self.__is_binary:
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

        error_dict_total, _ = self._analyze_false_positive_for_categories(categories)

        if not show:
            return error_dict_total

        labels = [self.dataset.get_display_name_of_category(cat) for cat in error_dict_total.keys()]
        plot_class_distribution(error_dict_total, labels, self.result_saving_path,
                                self._SAVE_PNG_GRAPHS, "False Positive distribution")

    def set_normalization(self, use_normalization, with_properties=True, norm_factor_categories=None,
                          norm_factors_properties=None):
        """Sets the normalization for the metrics calculation

        Parameters
        ----------
        use_normalization: bool
            Specifies whether or not to use normalization
        with_properties: bool, optional
            Specifies whether or not to normalize also on properties values (default is True)
        norm_factor_categories: float, optional
            Categories normalization factor (default is 1/number of categories)
        norm_factors_properties: list of tuple, optional
            Properties normalization factors.

            Each pair specifies the normalization factor to apply to a specific property.
            (Example: [(name1, value1), (name2, value2), ...]
        """
        if not isinstance(use_normalization, bool):
            logger.error(err_type.format("use_normalization"))
            return -1

        if not isinstance(with_properties, bool):
            logger.error(err_type.format("with_properties"))
            return -1

        if norm_factor_categories is not None and not isinstance(norm_factor_categories, Number):
            logger.error(err_type.format("norm_factor_categories"))
            return -1

        if norm_factors_properties is not None and (not isinstance(norm_factors_properties, list) or
                                                    not (all(isinstance(item, tuple) and len(item) == 2
                                                             for item in norm_factors_properties))):
            logger.error(err_type.format("norm_factors_properties"))
            return -1

        self._use_normalization = use_normalization
        self._use_new_normalization = with_properties
        if norm_factor_categories is not None:
            self._norm_factors["categories"] = norm_factor_categories
        if norm_factors_properties is not None:
            dataset_p_names = self.dataset.get_property_keys()
            for p_name, p_value in norm_factors_properties:
                if p_name in dataset_p_names:
                    self._norm_factors[p_name] = p_value
                else:
                    logger.warning("Invalid property name in 'norm_factors_properties'.")
        self.clear_saved_results()
        self.clear_saved_analyses()

    def set_confidence_threshold(self, threshold):
        """Sets the threshold value. Predictions with a confidence lower than the threshold are ignored.

        Parameters
        ----------
        threshold: float
            Threshold value. Must be between 0 and 1
        """
        if not isinstance(threshold, Number):
            logger.error("Invalid threshold type.")
            return -1
        if threshold < 0 or threshold > 1:
            logger.error("Invalid threshold value.")
            return -1
        self._conf_thresh = threshold
        self.clear_saved_results()
        self.clear_saved_analyses()

    def get_true_positive_ids(self, categories):
        """
        Returns the gt and proposals ids of the true positive predictions
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

        _, ids = self._analyze_true_positive_for_categories(categories)
        return ids

    def get_false_positive_ids(self, categories):
        """
        Returns the gt and proposals ids of the false positive predictions
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

        _, ids = self._analyze_false_positive_for_categories(categories)
        return ids

    def get_false_positive_errors_ids(self, categories):
        if not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1

        _, ids = self._analyze_false_positive_errors(categories, self._conf_thresh)
        return ids

    def get_false_negative_ids(self, categories):
        """
        Returns the gt and proposals ids of the false negative predictions
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

        _, ids = self._analyze_false_negative_for_categories(categories)
        return ids

    def clear_saved_results(self, metrics=None):
        """
        Removes already calculated metric values from the memory
        Parameters
        ----------
        metrics: list
            Metrics to remove already calculated values
        """
        if metrics is None:
            self.saved_results = {}
            return

        if not isinstance(metrics, list) or not (all(isinstance(m, Metrics) and isinstance(m, str) for m in metrics)):
            logger.error(err_type.format("metrics"))
            return -1

        for m in metrics:
            if m in self.saved_results.keys():
                self.saved_results[m] = {}

    def clear_saved_analyses(self):
        self.saved_analyses = {}

    def _get_metrics_with_threshold(self):
        """
        Returns the list of the metrics that use thresholding
        Returns
        -------
        list
        """
        return self._metrics_with_threshold

    def _get_report_results(self, default_metrics, metrics, categories, properties, show_categories, show_properties, include_reliability):
        """
        Calculates the metrics values for the base report.
        The specified metrics are calculated for the entire dataset (with micro and macro averaging) and can be
        calculated also per-class and per-property value

        Parameters
        ----------
        default_metrics: list
            Allowed metrics
        metrics: list
            Selected metrics for the base report
        categories: list
            Selected categories for the base report
        properties: list
            Selected properties for the base report
        show_categories: bool
            If False the categories metrics values are not shown
        show_properties: bool
            If False the properties metrics values are not shown

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(show_categories, bool):
            logger.error(err_type.format("show_categories"))
            return -1
        if not isinstance(show_properties, bool):
            logger.error(err_type.format("show_categories"))
            return -1

        default_metrics.extend(self.get_custom_metrics())
        if metrics is None:
            metrics = default_metrics
        else:
            if not isinstance(metrics, list):
                logger.error(err_type.format("metrics"))
                return -1
            if len(metrics) == 0:
                logger.error(err_value.format("metrics", self._valid_metrics))
                return -1
            for m in metrics:
                if m not in default_metrics:
                    logger.error(
                        "Metric {} not supported for report. Available metrics: {}.".format(m, default_metrics))
                    return -1

        if self.__is_binary:
            show_categories = False
        else:
            if categories is None:
                categories = self.dataset.get_categories_names()
            else:
                if not isinstance(categories, list):
                    logger.error(err_type.format("categories"))
                    return -1
                if not categories:
                    logger.warning("Empty categories list")
                    show_categories = False
                elif not self.dataset.are_valid_categories(categories):
                    return -1

        if properties is None:
            properties = list(self.dataset.get_property_keys())
        else:
            if not isinstance(properties, list):
                logger.error(err_type.format("properties"))
                return -1
            if not properties:
                logger.warning("Empty properties list")
                show_properties = False
            elif not self.dataset.are_valid_properties(properties):
                return -1

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
            m_name = metric.value if isinstance(metric, Metrics) else metric
            results[m_name] = self._calculate_report_for_metric(input_report, categories, properties,
                                                                      show_categories, show_properties, metric)

        if Metrics.ACCURACY in metrics and self.dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            top_res = self.analyze_top1_top5_error(metric=Metrics.ACCURACY, show=False)[1]
            results["Top5 Accuracy"] = {"avg micro": top_res}

            if show_properties:
                top_res = self.analyze_top1_top5_error(properties=properties, metric=Metrics.ACCURACY, show=False)
                for p in properties:
                    for p_value in top_res[p]:
                        name = p + "_" + "{}".format(p_value)
                        results["Top5 Accuracy"][name] = top_res[p][p_value][1]
        if Metrics.ERROR_RATE in metrics and self.dataset.task_type == TaskType.CLASSIFICATION_SINGLE_LABEL:
            top_res = self.analyze_top1_top5_error(metric=Metrics.ERROR_RATE, show=False)[1]
            results["Top5 Error Rate"] = {"avg micro": top_res}

            if show_properties:
                top_res = self.analyze_top1_top5_error(properties=properties, metric=Metrics.ERROR_RATE, show=False)
                for p in properties:
                    for p_value in top_res[p]:
                        name = p + "_" + "{}".format(p_value)
                        results["Top5 Error Rate"][name] = top_res[p][p_value][1]

        if include_reliability:
            rel_res = self.analyze_reliability(show=False)
            t_name = "total" if self.__is_binary else "avg micro"
            results["ece"] = {t_name: rel_res["ece"]}
            results["mce"] = {t_name: rel_res["mce"]}
            if show_categories:
                rel_res = self.analyze_reliability_for_categories(categories=categories, show=False)
                for c in categories:
                    results["ece"][c] = rel_res[c]["ece"]
                    results["mce"][c] = rel_res[c]["mce"]

        type_dataframe = pd.DataFrame(type_dict)

        data = pd.DataFrame(results)
        data = pd.merge(data, type_dataframe, left_index=True, right_index=True).reset_index()
        data = data.rename(columns={"index": "label"})
        data = data.set_index(["type", "label"])
        data = data.fillna("not supported")
        return data

    def update_property_normalization_factor(self, property_name, value=None):
        properties = self.dataset.get_property_keys()
        if property_name not in properties:
            logger.error(err_value.format("property_name", properties))
            return -1

        self._norm_factors[property_name] = 1/len(self.dataset.get_values_for_property(property_name)) if value is None else value

    def __create_norm_factors_dict(self, norm_factor_classes, norm_factors_properties):
        """
        Creates a dict containing the normalization factors of the categories and of all the properties

        Parameters
        ----------
        norm_factor_classes: float
            Normalization factor of the categories
        norm_factors_properties: list of (str, float)
            Pairs of property name and property normalization factor

        Returns
        -------
        dict
        """
        norm_factors = {}
        norm_factors["categories"] = 1 / len(self.dataset.get_categories_names()) if norm_factor_classes is None else norm_factor_classes
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
                    logger.warning("Invalid property name in 'norm_factors_properties'.")
            for p_name in remaining_p_names:
                norm_factors[p_name] = 1 / len(self.dataset.get_values_for_property(p_name))
        return norm_factors

    # -- Input validation support functions -- #

    def _is_valid_metric(self, metric):
        """
        Checks if the metric is valid for predictions evaluation
        Parameters
        ----------
        metric: Metrics

        Returns
        -------
        bool
        """
        if metric in self._valid_metrics:
            return True
        logger.error(f"Metric '{metric}' not valid. Valid metrics: {self._valid_metrics}")
        return False

    def __is_valid_curve(self, curve):
        """
        Checks if the curve is valid

        Parameters
        ----------
        curve: Curves
            Curve to check the validity

        Returns
        -------
        bool
            True if the curve is valid
        """
        if curve not in self._valid_curves:
            logger.error(f"Invalid curve: {curve}")
            return False
        return True

    # -- Evaluation metrics support functions -- #

    def _support_precision_score(self, tp, tp_norm, fp):
        """
        Calculates the precision score  and the precision normalized score

        Parameters
        ----------
        tp: int
            Number of True Positive
        tp_norm: int
            Number of True Positive Normalized
        fp: int
            Number of False Positive

        Returns
        -------
        precision, precision_norm
        """
        precision = tp / (tp + fp) if tp > 0 else 0
        precision_norm = tp_norm / (tp_norm + fp) if tp_norm > 0 else 0
        return precision, precision_norm

    def _support_recall_score(self, tp, tp_norm, fn):
        """
        Calculates the recall score and the recall normalized score

        Parameters
        ----------
        tp: int
            Number of True Positive
        tp_norm: int
            Number of True Positive Normalized
        fn: int
            Number of False Negative

        Returns
        -------
        recall, recall_norm
        """
        recall = tp / (tp + fn) if tp > 0 else 0
        recall_norm = tp_norm / (tp_norm + fn) if tp_norm > 0 else 0
        return recall, recall_norm

    def _support_f1_score(self, tp, tp_norm, fp, fn):
        """
        Calculates the F1 score and the F1 normalized score

        Parameters
        ----------
        tp: int
            Number of True Positive
        tp_norm: int
            Number of True Positive Normalized
        fp: int
            Number of False Positive
        fn: int
            Number of False Negative

        Returns
        -------
        f1, f1_norm
        """
        precision, precision_norm = self._support_precision_score(tp, tp_norm, fp)
        recall, recall_norm = self._support_recall_score(tp, tp_norm, fn)
        f1_den = precision + recall
        f1_norm_den = precision_norm + recall_norm
        f1 = 2 * precision * recall / f1_den if f1_den > 0 else 0
        f1_norm = 2 * precision_norm * recall_norm / f1_norm_den if f1_norm_den > 0 else 0
        return f1, f1_norm

    def _support_precision_recall(self, n_anns, n_normalized, confidence, tp, fp):
        """
        Calculates the precision, the precision normalized, the recall and the recall normalized at different
        thresholds values

        Parameters
        ----------
        n_anns: int
            Number of positive annotations/observations considered
        n_normalized: float
            Normalized number of images/observations
        confidence: array-like
            Sorted confidence values
        tp: array-like
            Cumsum of True Positive at different confidence values
        fp: array-like
            Cumsum of False Positive at different confidence values

        Returns
        -------
        precision, precision_norm, recall, recall_norm, rel_indexes: rel_indexes is a list indicating the indexes of
        the metric values at different thresholds
        """
        tp_norm = np.multiply(tp, n_normalized) / n_anns
        precision = np.true_divide(tp, np.add(tp, fp))
        recall = np.true_divide(tp, n_anns)
        fn = n_anns - tp[-1] if len(tp) > 0 else n_anns

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
        """
        Calculates the precision, the precision normalized, the recall and the recall normalized in order to plot
        the Precision-Recall curve.

        It is different from '_support_precision_recall' because this method calculates the precision and recall at
        different threshold values by considering only the unique values of the recall and the unique precision values
        that corresponds to the recall value in order to be able to plot the curve

        Parameters
        ----------
        n_gt: int
            Number of annotations/observations considered
        n_tot: int
            Number of positive annotations/observations considered
        n_normalized: float
            Normalized number of images/observations
        confidence: array-like
            Sorted confidence values
        tp: array-like
            Cumsum of True Positive at different confidence values
        fp: array-like
            Cumsum of False Positive at different confidence values
        is_classification: bool
            True if it is a classification problem

        Returns
        -------
        precision, precision_norm, recall, recall_norm
        """
        precision, precision_norm, recall, recall_norm, rel_indexes = self._support_precision_recall(n_tot,
                                                                                                     n_normalized,
                                                                                                     confidence, tp, fp)
        if rel_indexes:
            precision = precision[np.sort(rel_indexes)]
            precision_norm = precision_norm[np.sort(rel_indexes)]
            recall = recall[np.sort(rel_indexes)]
            recall_norm = recall_norm[np.sort(rel_indexes)]

        one = np.ones(1)
        zero = np.zeros(1)

        recall = np.concatenate([zero, recall])
        recall_norm = np.concatenate([zero, recall_norm])
        precision = np.concatenate([one, precision])
        precision_norm = np.concatenate([one, precision_norm])

        if recall[-1] != 1:
            p_value = np.zeros(1)
            if is_classification:
                p_value[0] = n_tot / n_gt
            else:
                p_value[0] = 0  # set 0 in od
            recall = np.concatenate([recall, one])
            recall_norm = np.concatenate([recall_norm, one])
            precision = np.concatenate([precision, p_value])
            if is_classification:
                p_value[0] = n_normalized / (n_normalized + (n_gt - n_tot))
            else:
                p_value[0] = 0  # set 0 in od
            precision_norm = np.concatenate([precision_norm, p_value])

        recall = np.flip(recall)
        recall_norm = np.flip(recall_norm)
        precision = np.flip(precision)
        precision_norm = np.flip(precision_norm)

        return precision, precision_norm, recall, recall_norm

    def _support_f1_curve(self, det_ord, precision, precision_norm, recall, recall_norm, rel_indexes):
        """
        Calculates the F1 and the F1 normalized at different thresholds
        Parameters
        ----------
        det_ord: array-like
            Sorted detections
        precision: array-like
            Precision at different threshold
        precision_norm: array-like
            Precision normalized at different threshold
        recall: array-like
            Recall at different threshold
        recall_norm: array-like
            Recall normalized at different threshold
        rel_indexes: array-like
            Indexes of different thresholds

        Returns
        -------
        thresholds, f1
        """
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
        """
        Calculates the average precision score and the average precision normalized score

        Parameters
        ----------
        n_gt: int
            Number of annotations/observations considered
        n_tot: int
            Number of positive annotations/observations considered
        n_normalized: float
            Normalized number of images/observations
        confidence: array-like
            Sorted confidence values
        tp: array-like
            Cumsum of True Positive at different confidence values
        fp: array-like
            Cumsum of False Positive at different confidence values
        is_classification: bool
            True if it is a classification problem

        Returns
        -------
        average_precision_score, standard_error
        """
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

    # -- ABSTRACT METHODS -- #

    @abc.abstractmethod
    def base_report(self, metrics=None, categories=None, properties=None, show_categories=True, show_properties=True):
        pass

    @abc.abstractmethod
    def analyze_false_positive_errors_for_category(self, category, metric=None, show=True):
        pass

    @abc.abstractmethod
    def analyze_false_positive_trend_for_category(self, category, include_correct_predictions=True, show=True):
        pass

    @abc.abstractmethod
    def analyze_false_negative_errors_for_category(self, category, show=True):
        pass

    @abc.abstractmethod
    def analyze_reliability(self, num_bins=10, show=True):
        pass

    @abc.abstractmethod
    def analyze_reliability_for_categories(self, categories=None, num_bins=10, show=True):
        pass

    @abc.abstractmethod
    def _analyze_true_positive_for_categories(self, categories):
        pass

    @abc.abstractmethod
    def _analyze_false_negative_for_categories(self, categories):
        pass

    @abc.abstractmethod
    def _analyze_false_positive_for_categories(self, categories):
        pass

    @abc.abstractmethod
    def _analyze_true_positive_for_category_for_property(self, category, property_name, property_values):
        pass

    @abc.abstractmethod
    def _analyze_false_negative_for_category_for_property(self, category, property_name, property_values):
        pass

    @abc.abstractmethod
    def _compute_curve_overall(self, curve):
        pass

    @abc.abstractmethod
    def _compute_curve_for_categories(self, categories, curve):
        pass

    @abc.abstractmethod
    def _calculate_metric_for_category(self, category, metric):
        pass

    @abc.abstractmethod
    def _calculate_metric_for_properties_of_category(self, category_name, category_id, property_name, possible_values,
                                                     matching, metric):
        pass

    @abc.abstractmethod
    def _analyze_false_positive_errors(self, categories, threshold):
        pass

    @abc.abstractmethod
    def _get_input_report(self, properties, show_properties_report):
        pass

    @abc.abstractmethod
    def _calculate_report_for_metric(self, input_report, categories, properties, show_categories, show_properties,
                                     metric):
        pass
