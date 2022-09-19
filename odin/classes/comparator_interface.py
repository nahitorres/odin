import abc
import os

import numpy as np
import pandas as pd

from odin.classes import CustomMetric, TaskType, Curves, Metrics
from odin.classes.strings import err_type, err_value, err_property_not_loaded
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_models_comparison_on_error_impact, plot_models_comparison_on_property, \
    plot_models_comparison_on_tp_fp_fn_tn, plot_multiple_curves, plot_models_comparison_on_sensitivity_impact

logger = get_root_logger()


class ComparatorInterface(metaclass=abc.ABCMeta):

    models = {}
    _default_dataset = None
    _valid_metrics = []
    _valid_curves = {}

    def __init__(self,
                 task_type,
                 multiple_proposals_path,
                 result_saving_path,
                 use_normalization,
                 norm_factor_categories,
                 norm_factors_properties,
                 conf_thresh,
                 metric,
                 similar_classes,
                 match_on_filename,
                 save_graph_as_png
                 ):

        self.task_type = task_type
        self.proposals_path = multiple_proposals_path
        self.result_saving_path = result_saving_path
        if save_graph_as_png and not os.path.exists(result_saving_path):
            os.mkdir(result_saving_path)
        self.use_normalization = use_normalization
        self.norm_factor_categories = norm_factor_categories
        self.norm_factors_properties = norm_factors_properties
        self.conf_thresh = conf_thresh
        self.metric = metric
        self.similar_classes = similar_classes
        self.match_on_filename = match_on_filename
        self.save_graph_as_png = save_graph_as_png
        self._is_binary = task_type == TaskType.CLASSIFICATION_BINARY
        self._allow_analyses = False
        self.models = {}

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

        for model in self.models:
            self.models[model]["analyzer"].add_custom_metric(custom_metric)

        self._valid_metrics.append(Metrics(custom_metric.get_name()))

    def load_categories_display_names(self):
        """Loads into memory the names of the categories to display in the plots.
        The names are retrieved from the properties file (default is 'properties.json')
        """
        self._default_dataset.load_categories_display_names()

    def load_properties_display_names(self):
        """Loads into memory the properties and their values.
        The properties are retrieved from the properties file (default is 'properties.json')
        """
        self._default_dataset.load_properties_display_names()

    def analyze_property(self, property_name, possible_values=None, categories=None, metric=None, models=None, show=True):
        """
        It compares the performances of the models for each category considering subsets pf the data set which have a specific property value.

        Parameters
        ----------
        property_name: str
            Name of the property to be analyzed.
        possible_values: list, optional
            Property values to be analyzed. If not specified, all the property values are considered. (default is None)
        categories: list, optional
            List of the categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        metric: Metrics, optional
            Evaluation metric used for the analysis. If not specified, the default one is used. (default is None)
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
        elif property_name not in self._default_dataset.get_property_keys():
            if self._default_dataset.is_possible_property(property_name):
                logger.error(err_property_not_loaded.format(property_name))
                return -1
            logger.error(err_value.format("property_name", list(self._default_dataset.get_property_keys())))
            return -1

        if possible_values is None:
            possible_values = self._default_dataset.get_values_for_property(property_name)
        elif not isinstance(possible_values, list):
            logger.error(err_type.format("possible_values"))
            return -1
        elif not possible_values:
            possible_values = self._default_dataset.get_values_for_property(property_name)
        elif not self._default_dataset.is_valid_property(property_name, possible_values):
            return -1

        if self._is_binary:
            categories = [self._default_dataset.get_categories_names()[0]]
        elif categories is None or not categories:
            categories = self._default_dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self._default_dataset.are_valid_categories(categories):
            return -1

        if metric is None:
            metric = self.metric
        elif not isinstance(metric, Metrics):
            logger.error(err_type.format("metric"))
            return -1
        elif metric not in self._valid_metrics:
            logger.error(err_value.format("metric", self._valid_metrics))
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

        all_performance = {}

        if len(models) == 1:
            all_performance[models[0]] = self.models[models[0]]["analyzer"].analyze_property(property_name, possible_values,
                                                                                         categories, show, metric)
            if show:
                return
        else:
            for model in models:
                all_performance[model] = self.models[model]["analyzer"].analyze_property(property_name, possible_values,
                                                                                             categories, False, metric)
        if not show:
            return all_performance

        for cat in categories:
            results = {}
            for p_value in possible_values:
                results[p_value] = {}
                for model in models:
                    results[p_value][model] = all_performance[model][cat][property_name][p_value]
            label_cat = self._default_dataset.get_display_name_of_category(cat)
            label_p_name = self._default_dataset.get_display_name_of_property(property_name)
            labels_p_values = [self._default_dataset.get_display_name_of_property_value(property_name, p_value) for p_value in results]
            plot_models_comparison_on_property(results, models, label_cat, label_p_name, labels_p_values, metric, self.save_graph_as_png,
                                               self.result_saving_path)

    def analyze_sensitivity_impact_of_properties(self, properties=None, metric=None, models=None, show=True):
        """
        It compares the sensitivity and impact of the models on the different properties.

        Parameters
        ----------
        properties: list, optional
            List of properties to be included in the analysis. If not specified, all the properties are included. (default is None)
        metric: Metrics, optional
            Evaluation metric used for the analysis. If not specified, the default one is used. (default is None)
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

        if properties is None:
            properties = list(self._default_dataset.get_property_keys())
        elif not isinstance(properties, list):
            logger.error(err_type.format("properties"))
            return -1
        elif not self._default_dataset.are_valid_properties(properties):
            return -1

        if metric is None:
            metric = self.metric
        elif not isinstance(metric, Metrics):
            logger.error(err_type.format("metric"))
            return -1
        elif metric not in self._valid_metrics:
            logger.error(err_value.format("metric", self._valid_metrics))
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

        all_performance = {}

        if len(models) == 1:
            all_performance[models[0]] = self.models[models[0]]["analyzer"].analyze_sensitivity_impact_of_properties(properties, metric, show)
            if show:
                return
        else:
            for model in models:
                all_performance[model] = self.models[model]["analyzer"].analyze_sensitivity_impact_of_properties(properties, metric, show=False)

        if not show:
            return all_performance

        display_names = [self._default_dataset.get_display_name_of_property(pkey) for pkey in properties]

        plot_models_comparison_on_sensitivity_impact(all_performance, models, properties, display_names, metric,
                                                     self.result_saving_path, self.save_graph_as_png)

    def analyze_curve(self, curve=Curves.PRECISION_RECALL_CURVE, average="macro", models=None, show=True):
        """
        It compares the overall models performances by plotting the desired curve.

        Parameters
        ----------
        curve: Curves, optional
            Evaluation curve used for the analysis. (default is Curves.PRECISION_RECALL_CURVE)
        average: str, optional
            Indicates the averaging method. It can be 'macro' or 'micro'. (default is 'macro')
        models: list, optional
            List of models on which to perform the analysis. If not specified, all models are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not self._allow_analyses:
            logger.error("Please select the properties first")
            return -1

        if not isinstance(curve, Curves):
            logger.error(err_type.format("curve"))
            return -1
        elif curve not in self._valid_curves:
            logger.error(err_value.format("curve", self._valid_curves))
            return -1

        if not isinstance(average, str):
            logger.error(err_type.format("average"))
            return -1
        elif average not in ["macro", "micro"]:
            logger.error(err_value.format("average", "[macro, micro]"))
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

        all_performance = {}
        display_names = {}

        if len(models) == 1:
            all_performance[models[0]] = self.models[models[0]]["analyzer"].analyze_curve(curve=curve, average=average, show=show)["overall"]
            if show:
                return
        else:
            for model in models:
                all_performance[model] = self.models[model]["analyzer"].analyze_curve(curve=curve, average=average, show=False)["overall"]
                display_names[model] = {"display_name": model}

        if not show:
            return all_performance

        plot_multiple_curves(all_performance, curve, display_names, self.save_graph_as_png, self.result_saving_path, legend_title="Models")

    def analyze_curve_for_categories(self, categories=None, curve=Curves.PRECISION_RECALL_CURVE, models=None, show=True):
        """
        It compares the models performances for each category by plotting the desired curve.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        curve: Curves, optional
            Evaluation curve used for the analysis. (default is Curves.PRECISION_RECALL_CURVE)
        models: list, optional
            List of models on which to perform the analysis. If not specified, all models are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not self._allow_analyses:
            logger.error("Please select the properties first")
            return -1

        all_performance = {}
        display_names = {}

        if categories is None:
            categories = self._default_dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self._default_dataset.are_valid_categories(categories):
            return -1

        if not isinstance(curve, Curves):
            logger.error(err_type.format("curve"))
            return -1
        elif curve not in self._valid_curves:
            logger.error(err_value.format("curve", self._valid_curves))
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

        for model in models:
            results = self.models[model]["analyzer"].analyze_curve_for_categories(categories=categories, curve=curve,
                                                                                  show=(show and len(models) == 1))
            for c in categories:
                c_display_name = self._default_dataset.get_display_name_of_category(c)
                name = model + "-" + c_display_name
                all_performance[name] = results[c]
                display_names[name] = {"display_name": name}

        if show and len(models) == 1:
            return

        if not show:
            return all_performance

        plot_multiple_curves(all_performance, curve, display_names, self.save_graph_as_png, self.result_saving_path,
                             legend_title="Models")

    def analyze_false_positive_errors(self, categories=None, metric=None, models=None, show=True):
        """
        It compares the false positives by identifying the type of the errors of the models and shows the gain that each model could achieve by removing all the false positives of each type.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        metric: Metrics, optional
            Evaluation metric used for the analysis. If not specified, the default one is used. (default is None)
        models: list, optional
            List of models on which to perform the analysis. If not specified, all models are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """
        if not self._allow_analyses:
            logger.error("Please select the properties first")
            return -1

        if self._is_binary:
            categories = [self._default_dataset.get_categories_names()[0]]

        if categories is None:
            categories = self._default_dataset.get_categories_names()
        elif not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1
        elif not self._default_dataset.are_valid_categories(categories):
            return -1

        if metric is None:
            metric = self.metric
        elif not isinstance(metric, Metrics):
            logger.error(err_type.format("metric"))
            return -1
        elif metric not in self._valid_metrics:
            logger.error(err_value.format("metric", self._valid_metrics))
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

        all_performance = {}

        for cat in categories:
            results = {}
            errors = None
            for model in models:
                impact_counts = []
                error_values, err, category_metric_value = self.models[model]["analyzer"].analyze_false_positive_errors_for_category(cat, metric=metric, show=(show and len(models)==1))
                for value, count in error_values:
                    impact = value - category_metric_value
                    impact_counts.append([impact, count])
                results[model] = impact_counts
                if errors is None:
                    errors = err
            all_performance[cat] = results
            if show and len(models) > 1:
                cat_label = self._default_dataset.get_display_name_of_category(cat)
                plot_models_comparison_on_error_impact(results, errors, cat_label, metric, self.save_graph_as_png,
                                                       self.result_saving_path)
        if not show:
            return all_performance

    def analyze_false_negative_errors(self, categories=None, models=None, show=True):
        """
        It compares the false negatives by identifying the type of the errors of the models.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        models: list, optional
            List of models on which to perform the analysis. If not specified, all models are included. (default is None)
        show: bool, optional
            Indicates whether the plot should be shown or not. If False, returns the results as dict. (default is True)
        """

        if self._is_binary:
            logger.error("Analysis not supported for binary classification")
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
        tmp_results = {}
        if len(models) == 1:
            tmp_results[models[0]] = self.models[models[0]]["analyzer"].analyze_false_negative_errors(categories=categories,
                                                                                              show=show)
            if show:
                return
        else:
            for model in models:
                tmp_results[model] = self.models[model]["analyzer"].analyze_false_negative_errors(categories=categories, show=False)

        for c in categories:
            results[c] = {}
            for model in models:
                results[c][model] = tmp_results[model][c]

        if not show:
            return results

        c = list(results.keys())[0]
        m = list(results[c].keys())[0]
        labels = list(results[c][m].keys())
        for c in categories:
            c_label = self._default_dataset.get_display_name_of_category(c)
            plot_models_comparison_on_tp_fp_fn_tn(results[c], labels, f"False Negative categorization - {c_label}", "Errors", self.save_graph_as_png,
                                                  self.result_saving_path)

    def show_true_positive_distribution(self, categories=None, models=None, show=True):
        """
        It compares the true positives of the models.

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
            results[models[0]] = self.models[models[0]]["analyzer"].show_true_positive_distribution(categories, show=show)
            if show:
                return
        else:
            for model in models:
                results[model] = self.models[model]["analyzer"].show_true_positive_distribution(categories, show=False)

        if not show:
            return results

        labels = [self._default_dataset.get_display_name_of_category(cat) for cat in categories]
        plot_models_comparison_on_tp_fp_fn_tn(results, labels, "True Positive comparison", "Categories", self.save_graph_as_png,
                                              self.result_saving_path)

    def show_false_positive_distribution(self, categories=None, models=None, show=True):
        """
        It compares the false positives of the models.

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
            results[models[0]] = self.models[models[0]]["analyzer"].show_false_positive_distribution(categories, show=show)
            if show:
                return
        else:
            for model in models:
                results[model] = self.models[model]["analyzer"].show_false_positive_distribution(categories, show=False)

        if not show:
            return results

        labels = [self._default_dataset.get_display_name_of_category(cat) for cat in categories]
        plot_models_comparison_on_tp_fp_fn_tn(results, labels, "False Positive comparison", "Categories", self.save_graph_as_png,
                                              self.result_saving_path)

    def show_false_negative_distribution(self, categories=None, models=None, show=True):
        """
        It compares the false negatives of the models.

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
            results[models[0]] = self.models[models[0]]["analyzer"].show_false_negative_distribution(categories, show=show)
            if show:
                return
        else:
            for model in models:
                results[model] = self.models[model]["analyzer"].show_false_negative_distribution(categories, show=False)
        labels = [self._default_dataset.get_display_name_of_category(cat) for cat in categories]

        if not show:
            return results

        plot_models_comparison_on_tp_fp_fn_tn(results, labels, "False Negative comparison", "Categories", self.save_graph_as_png,
                                              self.result_saving_path)

    def base_report(self, metrics=None, categories=None, properties=None, show_categories=True,
                    show_properties=True, models=None, include_reliability=True):
        """
        It summarizes the models performances at all levels of granularity.

        Parameters
        ----------
        metrics: list, optional
            List of evaluation metrics to be included in the analysis. If not specified, all the evaluation metrics are included. (default is None)
        categories: list, optional
            List of categories to be included in the analysis. If not specified, all the categories are included. (default is None)
        properties: list, optional
            List of properties to be included in the analysis. If not specified, all the properties are included. (default is None)
        show_categories: bool, optional
            List of properties to be included in the analysis. If not specified, all the properties are included. (default is None)
        show_properties: bool, optional
            Indicates whether the properties should be included in the report. (default is True)
        models: list, optional
            List of models on which to perform the analysis. If not specified, all models are included. (default is None)
        include_reliability: bool, optional
            Indicates whether the 'ece' and 'mce' should be included in the report. (default is True)
        Returns
        -------
            pandas.DataFrame
        """
        if not self._allow_analyses:
            logger.error("Please select the properties first")
            return -1

        if show_properties and not self._default_dataset.are_analyses_with_properties_available():
            logger.error("No properties available. Please make sure to load the properties to the dataset "
                         "or set 'show_properties=False")
            return -1

        if metrics is not None and not isinstance(metrics, list):
            logger.error(err_type.format("metrics"))
            return -1

        if categories is not None and not isinstance(categories, list):
            logger.error(err_type.format("categories"))
            return -1

        if properties is not None and not isinstance(properties, list):
            logger.error(err_type.format("properties"))
            return -1

        if not isinstance(show_categories, bool):
            logger.error(err_type.format("show_categories"))
            return -1

        if not isinstance(show_properties, bool):
            logger.error(err_type.format("show_properties"))
            return -1

        if models is None:
            models = self.models.keys()
        elif not isinstance(models, list):
            logger.error(err_type.format("models"))
            return -1
        elif any(m not in self.models for m in models):
            logger.error(err_value.format("models", list(self.models.keys())))
            return -1

        if not isinstance(include_reliability, bool):
            logger.error(err_type.format("include_reliability"))
            return -1

        results = []
        df = None
        if len(models) == 1:
            df = self.models[models[0]]["analyzer"].base_report(metrics, categories, properties, show_categories,
                                                                          show_properties, include_reliability)
        else:
            for model in models:
                results.append(self.models[model]["analyzer"].base_report(metrics, categories, properties, show_categories,
                                                                          show_properties, include_reliability))

            if isinstance(results[0], int):
                return -1

            metrics_names = list(results[0].columns)

            for index, model in enumerate(models):
                new_names = []
                for name in results[index].columns:
                    name += "_" + str(model)
                    new_names.append(name)
                results[index].columns = new_names

            for index in range(0, len(results) - 1):
                if df is None:
                    df = pd.merge(results[index], results[index + 1], on=["type", "label"])
                else:
                    df = pd.merge(df, results[index + 1], on=["type", "label"])

            tuples, order = [], []
            second_level_names = np.tile(list(models), len(metrics_names))
            for m_name in metrics_names:
                for model in models:
                    name = str(m_name) + "_" + str(model)
                    tuples.append((m_name, name))
                    order.append(name)
            df = df[order]
            df.columns = pd.MultiIndex.from_tuples(tuples)
            df.columns.set_levels(second_level_names, level=1, inplace=True, verify_integrity=False)
        return df

    def show_true_positive_distribution_for_categories_for_property(self, property_name, property_values=None, categories=None, models=None, show=True):
        """
        It compares the true positive distribution of the property values for each category of the models.

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
                "analyzer"].show_true_positive_distribution_for_categories_for_property(property_name, property_values,
                                                                                        categories=categories,
                                                                                        show=show)
            if show:
                return
        else:
            for model in models:
                tmp_results[model] = self.models[model]["analyzer"].show_true_positive_distribution_for_categories_for_property(property_name, property_values, categories=categories, show=False)

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
            plot_models_comparison_on_tp_fp_fn_tn(results[c], labels, f"True Positive comparison of {p_label} for {c_label}", "Property values", self.save_graph_as_png,
                                                  self.result_saving_path)

    def show_false_negative_distribution_for_categories_for_property(self, property_name, property_values=None, categories=None, models=None, show=True):
        """
        It compares the false negative distribution of the property values for each category of the models.

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
                "analyzer"].show_false_negative_distribution_for_categories_for_property(property_name, property_values,
                                                                                         categories=categories,
                                                                                         show=show)
            if show:
                return
        else:
            for model in models:
                tmp_results[model] = self.models[model]["analyzer"].show_false_negative_distribution_for_categories_for_property(property_name, property_values, categories=categories, show=False)

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
            plot_models_comparison_on_tp_fp_fn_tn(results[c], labels, f"False Negative comparison of {p_label} for {c_label}", "Property values", self.save_graph_as_png,
                                                  self.result_saving_path)

    @abc.abstractmethod
    def _load_all_models_proposals(self, load_properties=True):
        pass
