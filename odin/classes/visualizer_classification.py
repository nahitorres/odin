import os

import cv2
import imageio
import numpy as np
from IPython.core.display import display
from ipywidgets import VBox, Checkbox, HBox, FloatSlider, Label, HTML, Box
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt

from .dataset_classification import DatasetClassification
from odin.utils import Iterator, get_root_logger
from .error_type import ErrorType
from .strings import err_type, err_value
from .visualizer_interface import VisualizerInterface
from odin.classes import strings as labels_str, TaskType, AnalyzerClassification, DatasetCAMs, AnnotationType, \
    AnalyzerCAMs


class VisualizerClassification(VisualizerInterface):
    __cam_visualization = False
    __heatmap = True
    __cam_threshold = 0.5

    def __init__(self, dataset, analyzers=None, is_image=True, custom_display_function=None):
        """
        The VisualizerClassification class can be used to visualize the ground truth, the predictions  and the CAMs of classification tasks.

        Parameters
        ----------
        dataset: DatasetClassification or DatasetCAMs
            Data set used for the visualization
        analyzers: list, optional
            List of the analyzers of different models. (default is None)
        is_image: bool, optional
            Indicates whether the data set represent images or not. (default is True)
        custom_display_function: function, optional
            User custom display function. If not specified, it is used the default one. (default is None)
        """

        if not isinstance(dataset, DatasetClassification) and not isinstance(dataset, DatasetCAMs):
            raise TypeError(err_type.format("dataset"))

        analyzers_dict = {}
        if analyzers is not None:
            if isinstance(analyzers, AnalyzerClassification) or isinstance(analyzers, AnalyzerCAMs):
                analyzers = [analyzers]
            elif not isinstance(analyzers, list):
                raise TypeError(err_type.format("analyzers"))
            for analyzer in analyzers:
                if isinstance(dataset, DatasetClassification) and not isinstance(analyzer, AnalyzerClassification):
                    raise TypeError("Invalid analyzers instances. Please, provide 'AnalyzerClassification' type")
                if not isinstance(analyzer, AnalyzerClassification) and not isinstance(analyzer, AnalyzerCAMs):
                    raise TypeError(
                        "Invalid analyzers instances. Please, provide 'AnalyzerClassification' or 'AnalyzerCAMs' type")

                analyzers_dict[analyzer._model_name] = analyzer

        super().__init__(dataset, analyzers_dict)
        self.__show_predictions = False
        self.__models_predictions = []
        self.__colors = {}
        self.__ids_to_show = None
        self.__is_image = is_image
        self.__display_function = self.__show_image
        self.__show_gt_cams = False
        self.__cams_categories = None
        self.__gt_preds_labels = HBox()

        self.__iterator = None

        if custom_display_function is not None:
            self.__display_function = custom_display_function
        elif not is_image:
            raise Exception(labels_str.warn_display_function_needed)

    def __create_cams_widgets(self):
        heatmap_chkbox = Checkbox(value=self.__heatmap, description="HeatMap")
        threshold_slider = FloatSlider(value=self.__cam_threshold, min=0, max=1.0, step=0.05,
                                       description='CAM threshold', disabled=False,
                                       continuous_update=False, orientation='horizontal', readout=True,
                                       readout_format='.2f')
        predictions_chkbox = Checkbox(value=self.__show_predictions, description="Show predictions")

        heatmap_chkbox.observe(self.cams_widget_changed, names=['value'])
        threshold_slider.observe(self.cams_widget_changed, names=['value'])
        predictions_chkbox.observe(self.cams_widget_changed, names=['value'])

        if not self.__cams_categories:
            all_cams_categories = Checkbox(value=self.__show_gt_cams, description="Show only GT CAMs")
            all_cams_categories.observe(self.cams_widget_changed, names=['value'])
            widgets = VBox([HBox([threshold_slider]), HBox([heatmap_chkbox, predictions_chkbox, all_cams_categories])])
        else:
            widgets = VBox([HBox([threshold_slider]), HBox([heatmap_chkbox, predictions_chkbox])])
        return widgets

    def cams_widget_changed(self, widget):
        if widget["owner"].description == "HeatMap":
            self.__heatmap = widget["new"]
        if widget["owner"].description == "CAM threshold":
            self.__cam_threshold = widget["new"]
        if widget["owner"].description == "Show predictions":
            self.__show_predictions = widget["new"]
        if widget["owner"].description == "Show only GT CAMs":
            self.__show_gt_cams = widget["new"]

        self.__iterator.refresh_output()

    def __show_image_cam(self, image, path):
        """
        Shows the image with the Class Activation Maps.
        Parameters
        ----------
        image: dict
            Image to show
        path: str
            Image path
        """

        categories = image["categories"] if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL else [
            image["category"]]

        orig_img = np.asarray(imageio.imread(path))
        annotations = self.dataset.masks_annotations.copy()

        gt_cat_names = self.dataset.get_categories_names_from_ids(categories)
        all_predictions = {}
        if self.__show_predictions:
            predictions = self.dataset.get_proposals_from_observation_id(image['id'], self.__models_predictions[0])
            if not isinstance(predictions, int):
                all_predictions[self.__models_predictions[0]] = predictions

        cams = self.dataset.get_cams_for_images_ids([image["id"]], self.__models_predictions[0])

        y_size = 2
        x_size = int(len(cams.index) / 2) if len(cams.index) % 2 == 0 else int(len(cams.index) / 2) + 1
        fig = plt.figure(figsize=(20, (x_size + 1) * 6))

        fig.add_subplot(x_size + 1, y_size, 1)
        plt.imshow(orig_img)
        plt.title(f"{image['file_name']}")
        plt.axis('off')

        self.__display_gt_predictions_labels(gt_cat_names, all_predictions)

        counter = 2
        for index, cam in cams.iterrows():
            if not self.__cams_categories:
                if self.__show_gt_cams and cam["category_id"] not in categories:
                    continue
            elif cam["category_id"] not in self.__cams_categories:
                continue

            fig.add_subplot(x_size + 1, y_size, counter)
            plt.imshow(orig_img)
            if self.__heatmap:
                if self.__cam_threshold > 0:
                    plt.imshow(cam["cam"] * (cam["cam"] >= self.__cam_threshold), cmap='jet', alpha=0.5, vmin=0, vmax=1)
                else:
                    plt.imshow(cam["cam"], cmap='jet', alpha=0.5, vmin=0, vmax=1)
            else:
                if self.__cam_threshold > 0:
                    plt.imshow(cam["cam"] >= self.__cam_threshold, cmap='gray', alpha=0.5, vmin=0, vmax=1)
                else:
                    plt.imshow(cam["cam"] > 0, cmap='gray', alpha=0.5, vmin=0, vmax=1)

            if self.dataset.ann_type == AnnotationType.BBOX:
                bboxes = annotations.loc[(annotations["image_id"] == image["id"])]["bbox"].tolist()
                for bbox in bboxes:
                    bbox_x, bbox_y, bbox_w, bbox_h = bbox
                    np_poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                               [bbox_x + bbox_w, bbox_y]]
                    plt.gca().add_patch(Polygon(np_poly, linestyle='-', facecolor='none', edgecolor='red', linewidth=2))
            else:
                seg_points = annotations.loc[annotations["image_id"] == image["id"]]["segmentation"].tolist()
                for pol in seg_points:
                    poly = [[float(pol[i]), float(pol[i + 1])] for i in range(0, len(pol), 2)]
                    np_poly = np.array(poly)
                    plt.gca().add_patch(Polygon(np_poly, linestyle='-', facecolor='none', edgecolor='red', linewidth=2))
            plt.axis('off')
            cat_name = self.dataset.get_category_name_from_id(cam["category_id"])
            plt.title(f"{cat_name}")
            counter += 1
        plt.show()

    def __get_gt_predictions_labels(self, gt_categories, predictions):
        gt_label = r"$\bf{Labels}$" + "\n\n"
        models_label = []

        for name in gt_categories:
            gt_label = gt_label + str(name) + "\n"

        for i, model in enumerate(predictions):
            if predictions[model].empty:
                continue

            preds_label = r"$\bf{" + str(model).replace('_', '-') + "}$" + "\n\n"

            for _, pred in predictions[model].iterrows():
                cat_name = self.dataset.get_category_name_from_id(int(pred['category_id']))
                preds_label = preds_label + f"{cat_name} | Confidence:{pred['confidence']:.2f}" + "\n"
            models_label.append(preds_label)

        return gt_label, models_label

    def __display_gt_predictions_labels(self, gt_categories, predictions):
        preds_labels = []
        for i, model in enumerate(predictions):
            if predictions[model].empty:
                continue
            vbox = []
            vbox.append(HTML(value=f"<b>{model}</b>"))
            for _, pred in predictions[model].iterrows():
                cat_name = self.dataset.get_category_name_from_id(int(pred['category_id']))
                vbox.append(Label(value=f"{cat_name} | Confidence:{pred['confidence']:.2f}"))
            box = VBox(vbox)
            box.layout.border = '1px solid red'
            box.layout.margin = '0px 5px 0px 0px'
            preds_labels.append(box)

        vbox = []
        vbox.append(HTML(value="<b>Labels</b>"))
        for name in gt_categories:
            vbox.append(Label(value=f"{name}"))
        gt_box = VBox(vbox)
        gt_box.layout.border = '1px solid green'
        gt_box.layout.margin = '0px 15px 0px 0px'

        widgets = [gt_box]
        widgets.extend(preds_labels)
        display(HBox(widgets))

    def __show_image(self, observation, index):
        """
        Shows the image with the meta-annotations specified previously by the user

        Parameters
        ----------
        observation: dict
            Observation to show
        index:
        """

        if "file_name" not in observation:
            print("File name not found")
            return

        print("Image with id:{}".format(observation["id"]))
        path = os.path.join(self.dataset.images_abs_path, observation["file_name"])
        if not os.path.exists(path):
            print("Image path does not exist: " + path)
            return

        if self.__cam_visualization:
            self.__show_image_cam(observation, path)
            return

        plt.figure(figsize=(10, 10))
        all_predictions = {}

        obs_cat_ids = observation['categories'] if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL else [observation['category']]

        if self.__current_category is None:
            gt_cat_names = self.dataset.get_categories_names_from_ids(obs_cat_ids)
            if self.__show_predictions:
                for model in self.__models_predictions:
                    predictions = self.dataset.get_proposals_from_observation_id(observation['id'], model)
                    predictions = predictions.sort_values("confidence", ascending=False).head(5)
                    if self.__ids_to_show is not None:
                        predictions = predictions.loc[predictions["id"].isin(self.__ids_to_show["props"])]
                    all_predictions[model] = predictions

        elif type(self.__current_category) == list:
            gt_cat_names = self.dataset.get_categories_names_from_ids(obs_cat_ids)
            if self.__show_predictions:
                for model in self.__models_predictions:
                    predictions = self.dataset.get_proposals_from_observation_id_and_categories(
                        observation['id'], self.__current_category, model)
                    if self.__ids_to_show is not None:
                        predictions = predictions.loc[predictions["id"].isin(self.__ids_to_show["props"])]
                    all_predictions[model] = predictions
        else:
            gt_cat_names = self.dataset.get_categories_names_from_ids(obs_cat_ids)
            if self.__show_predictions:
                for model in self.__models_predictions:
                    predictions = self.dataset.get_proposals_from_observation_id_and_categories(observation['id'],
                                                                                                [self.__current_category],
                                                                                                model)
                    if self.__ids_to_show is not None:
                        predictions = predictions.loc[predictions["id"].isin(self.__ids_to_show["props"])]
                    all_predictions[model] = predictions

        gt_label, preds_label = self.__get_gt_predictions_labels(gt_cat_names, all_predictions)

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        last_ann = plt.imshow(img)
        plt.axis('off')
        last_ann = plt.annotate(gt_label, (1, 1), xycoords=last_ann, xytext=(30, 0), textcoords="offset points", va="top", fontsize=12,
                                bbox=dict(boxstyle="round", fc="w", ec="g"))
        for p_l in preds_label:
            last_ann = plt.annotate(p_l, (1, 1), xycoords=last_ann, xytext=(30, 0), textcoords="offset points", va="top", fontsize=12,
                         bbox=dict(boxstyle="round", fc="w", ec="r"))
        plt.show()

    def visualize_annotations_for_ids(self, gt_ids, show_predictions=False):
        self.__ids_to_show = None
        self.__cam_visualization = False

        self.__models_predictions = list(self.dataset.proposals.keys())

        if isinstance(gt_ids, int):
            gt_ids = [gt_ids]
        elif not isinstance(gt_ids, list):
            get_root_logger().error(err_type.format("id"))
            return -1
        if not isinstance(show_predictions, bool):
            get_root_logger().error(err_type.format("show_predictions"))
            return -1

        observations = self.dataset.get_observations_from_ids(gt_ids)
        self.__start_iterator(observations, show_predictions=show_predictions)

    def visualize_annotations(self, categories=None, show_predictions=False):
        """
        It shows all the ground truth and, at user request, the related predictions.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the visualization. If not specified, all the categories are included. (default is None)
        show_predictions: bool, optional
            Indicates whether to visualize also the predictions. (default is False)
        """
        if not isinstance(show_predictions, bool):
            get_root_logger().error(err_type.format("show_predictions"))
            return -1

        self.__models_predictions = list(self.dataset.proposals.keys())
        self.__ids_to_show = None
        self.__cam_visualization = False

        if categories is None:
            observations = self.dataset.get_all_observations()
            categories_ids = self.dataset.get_categories_id_from_names(self.dataset.get_categories_names())
        elif not isinstance(categories, list):
            get_root_logger().error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1
        else:
            categories_ids = self.dataset.get_categories_id_from_names(categories)
            observations = self.dataset.get_observations_from_categories(categories)

        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            self.__start_iterator(observations, category=categories_ids, show_predictions=show_predictions)
        else:
            self.__start_iterator(observations, show_predictions=show_predictions)

    def visualize_annotations_for_property(self, meta_annotation, meta_annotation_value, show_predictions=False):
        """
        It shows all the ground truth with a specific property value and, at user request, the related predictions.

        Parameters
        ----------
        meta_annotation: str
            Name of the property to be visualized.
        meta_annotation_value: str or float
            Value of the property to be visualized.
        show_predictions: bool, optional
            Indicates whether to visualize also the predictions. (default is False)
        """
        self.__ids_to_show = None
        self.__cam_visualization = False

        self.__models_predictions = list(self.dataset.proposals.keys())

        if not self.dataset.are_analyses_with_properties_available():
            get_root_logger().error("No properties available")
            return -1

        if not isinstance(meta_annotation, str):
            get_root_logger().error(err_type.format("meta_annotation"))
            return -1
        elif not self.dataset.are_valid_properties([meta_annotation]):
            return -1

        if not isinstance(meta_annotation_value, str) and not isinstance(meta_annotation_value, float):
            get_root_logger().error(err_type.format("meta_annotation_value"))
            return -1
        elif not self.dataset.is_valid_property(meta_annotation, [meta_annotation_value]):
            return -1

        if not isinstance(show_predictions, bool):
            get_root_logger().error(err_type.format("show_predictions"))
            return -1

        observations = self.dataset.get_observations_from_property(meta_annotation, meta_annotation_value)

        self.__start_iterator(observations, show_predictions=show_predictions)

    def visualize_annotations_for_class_for_property(self, category, meta_annotation, meta_annotation_value,
                                                     show_predictions=False):
        """
        It shows all the ground truth of a category with a specific property value and, at user request, the related predictions.

        Parameters
        ----------
        category: str
            Name of the category to be visualized.
        meta_annotation:
            Name of the property to be visualized.
        meta_annotation_value:
            Value of the property to be visualized.
        show_predictions: bool, optional
            Indicates whether to visualize also the predictions. (default is False)
        """
        self.__ids_to_show = None
        self.__cam_visualization = False

        self.__models_predictions = list(self.dataset.proposals.keys())

        if not self.dataset.are_analyses_with_properties_available():
            get_root_logger().error("No properties available")
            return -1

        if not isinstance(category, str):
            get_root_logger().error(err_type.format("category"))
            return -1
        elif not self.dataset.is_valid_category(category):
            get_root_logger().error(err_value.format("category", self.dataset.get_categories_names()))
            return -1

        if not isinstance(meta_annotation, str):
            get_root_logger().error(err_type.format("meta_annotation"))
            return -1
        elif not self.dataset.are_valid_properties([meta_annotation]):
            return -1

        if not isinstance(meta_annotation_value, str) and not isinstance(meta_annotation_value, float):
            get_root_logger().error(err_type.format("meta_annotation_value"))
            return -1
        elif not self.dataset.is_valid_property(meta_annotation, [meta_annotation_value]):
            return -1

        if not isinstance(show_predictions, bool):
            get_root_logger().error(err_type.format("show_predictions"))
            return -1

        category_id = self.dataset.get_category_id_from_name(category)
        observations = self.dataset.get_observations_from_property_category(category_id, meta_annotation,
                                                                                       meta_annotation_value)
        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            self.__start_iterator(observations, category=category_id, show_predictions=show_predictions)
        else:
            self.__start_iterator(observations, show_predictions=show_predictions)

    def visualize_annotations_for_true_positive(self, categories=None, model=None):
        """
        It shows the ground truth and the predictions based on the true positive analysis.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the visualization. If not specified, all the categories are included. (default is None)
        model: str, optional
            Name of the model used for the analysis. If not specified, it is considered the first provided. (default is None)
        """

        if not self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer")
            return -1

        if model is None:
            model = list(self.analyzers.keys())[0]
        elif not isinstance(model, str):
            get_root_logger().error(err_type.format("model"))
            return -1
        elif model not in self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer for {} model".format(model))
            return -1

        self.__models_predictions = [model]

        if categories is not None and not isinstance(categories, list):
            get_root_logger().error(err_type.format("categories"))
            return -1

        self.__support_visualize_annotations_tp_fp_fn_tn(self.analyzers[model].get_true_positive_ids, categories)

    def visualize_annotations_for_false_positive(self, categories=None, model=None):
        """
        It shows the ground truth and the predictions based on the false positive analysis.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the visualization. If not specified, all the categories are included. (default is None)
        model: str, optional
            Name of the model used for the analysis. If not specified, it is considered the first provided. (default is None)
        """

        if not self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer")
            return -1

        if model is None:
            model = list(self.analyzers.keys())[0]
        elif not isinstance(model, str):
            get_root_logger().error(err_type.format("model"))
            return -1
        elif model not in self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer for {} model".format(model))
            return -1

        self.__models_predictions = [model]

        if categories is not None and not isinstance(categories, list):
            get_root_logger().error(err_type.format("categories"))
            return -1

        self.__support_visualize_annotations_tp_fp_fn_tn(self.analyzers[model].get_false_positive_ids, categories)

    def visualize_annotations_for_false_negative(self, categories=None, model=None):
        """
        It shows the ground truth and the predictions based on the false negative analysis.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the visualization. If not specified, all the categories are included. (default is None)
        model: str, optional
            Name of the model used for the analysis. If not specified, it is considered the first provided. (default is None)
        """

        if not self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer")
            return -1

        if model is None:
            model = list(self.analyzers.keys())[0]
        elif not isinstance(model, str):
            get_root_logger().error(err_type.format("model"))
            return -1
        elif model not in self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer for {} model".format(model))
            return -1

        self.__models_predictions = [model]

        if categories is not None and not isinstance(categories, list):
            get_root_logger().error(err_type.format("categories"))
            return -1

        self.__support_visualize_annotations_tp_fp_fn_tn(self.analyzers[model].get_false_negative_ids, categories)

    def visualize_annotations_for_true_negative(self, model=None):
        """
        It shows the ground truth and the predictions based on the true negative analysis.

        Parameters
        ----------
        model: str, optional
            Name of the model used for the analysis. If not specified, it is considered the first provided. (default is None)
        """

        if self.dataset.task_type != TaskType.CLASSIFICATION_BINARY:
            get_root_logger().error("True negative visualization is supported only for binary classification")
            return -1

        if not self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer")
            return -1

        if model is None:
            model = list(self.analyzers.keys())[0]
        elif not isinstance(model, str):
            get_root_logger().error(err_type.format("model"))
            return -1
        elif model not in self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer for {} model".format(model))
            return -1

        self.__models_predictions = [model]

        category = self.dataset.get_categories_names()[0]
        self.__support_visualize_annotations_tp_fp_fn_tn(self.analyzers[model].get_true_negative_ids, [category])

    def visualize_annotations_for_error_type(self, error_type, categories=None, model=None):
        """
        It shows the ground truth and the predictions based on the false positive error type analysis.

        Parameters
        ----------
        error_type: ErrorType
            Error type to be included in the analysis. The types of error supported are: ErrorType.BACKGROUND, ErrorType.SIMILAR_CLASSES, ErrorType.OTHER.
        categories: list, optional
            List of categories to be included in the visualization. If not specified, all the categories are included. (default is None)
        model: str, optional
            Name of the model used for the analysis. If not specified, it is considered the first provided. (default is None)
        """
        self.__cam_visualization = False

        if not isinstance(error_type, ErrorType):
            get_root_logger().error(err_type.format("error_type"))
            return -1
        elif self.dataset.task_type == TaskType.CLASSIFICATION_BINARY:
            get_root_logger().error("Not supported for binary classification.")
            return -1
        elif error_type == ErrorType.LOCALIZATION:
            get_root_logger().error("Not supported error type.")
            return -1
        elif error_type == ErrorType.BACKGROUND and self.dataset.task_type != TaskType.CLASSIFICATION_MULTI_LABEL:
            get_root_logger().error("Background error type supported only for multi-label classification.")
            return -1

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            get_root_logger().error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        if not self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer")
            return -1

        if model is None:
            model = list(self.analyzers.keys())[0]
        elif not isinstance(model, str):
            get_root_logger().error(err_type.format("model"))
            return -1
        elif model not in self.analyzers:
            get_root_logger().error("Please make sure to provide the analyzer for {} model".format(model))
            return -1

        self.__models_predictions = [model]

        fp_ids_per_cat = self.analyzers[model].get_false_positive_errors_ids(categories)
        fp_ids = {"gt": [], "props": []}
        for cat in categories:
            if error_type.value in fp_ids_per_cat[cat]:
                fp_ids["gt"].extend(fp_ids_per_cat[cat][error_type.value]["gt"])
                fp_ids["props"].extend(fp_ids_per_cat[cat][error_type.value]["props"])
        observations = self.dataset.get_observations_from_ids(fp_ids["gt"])
        categories_ids = self.dataset.get_categories_id_from_names(categories)
        self.__ids_to_show = fp_ids
        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            self.__start_iterator(observations, category=list(categories_ids), show_predictions=True)
        else:
            self.__start_iterator(observations, show_predictions=True)

    def visualize_cams(self, categories=None, cams_categories=None, heatmap=True, threshold=0.5, show_predictions=False,
                       show_only_gt_cams=False, model=None):
        """
        It shows all the ground truth and the related Class Activation Maps of the model.

        Parameters
        ----------
        categories: list, optional
            List of categories to be included in the visualization. If not specified, all the categories are included. (default is None)
        cams_categories: list, optional
            List of CAMs categories to be included in the visualization. If not specified, all the categories are included. (default is None)
        heatmap: bool, optional
            Indicates whether to visualize the CAMs in heatmap mode or not. (default is True)
        threshold: float, optional
            Threshold used for the CAMs visualization. (default is 0.5)
        show_predictions: bool, optional
            Indicates whether to visualize also the predictions. (default is False)
        show_only_gt_cams: bool, optional
            If 'cams_categories' is not specified, indicates whether to show all the CAMs or only the ones related to the ground truth labels. (default is False)
        model: str, optional
            Name of the model used for the visualization. If not specified, it is considered the first provided. (default is None)
        """

        if not self.dataset.cams:
            get_root_logger().error("No CAMs available. Please make sure to provide the CAMs")
            return -1

        if model is None:
            model = list(self.dataset.cams.keys())[0]
        elif not isinstance(model, str):
            get_root_logger().error(err_type.format("model"))
            return -1
        elif model not in self.dataset.cams:
            get_root_logger().error("Please make sure to provide the CAMs for {} model".format(model))
            return -1

        self.__models_predictions = [model]

        if type(self.dataset) is not DatasetCAMs or self.dataset.cams is None:
            get_root_logger().error("CAMs not available. Please make sure to load the CAMs to the dataset.")
            return -1

        if not isinstance(heatmap, bool):
            get_root_logger().error(err_type.format("heatmap"))
            return -1

        if not isinstance(threshold, float):
            get_root_logger().error(err_type.format("threshold"))
            return -1

        if not isinstance(show_predictions, bool):
            get_root_logger().error(err_type.format("show_predictions"))
            return -1

        self.__ids_to_show = None
        self.__cam_visualization = True
        self.__heatmap = heatmap
        self.__cam_threshold = threshold
        self.__show_gt_cams = show_only_gt_cams

        if cams_categories is not None:
            ids = self.dataset.get_categories_id_from_names(cams_categories)
            self.__cams_categories = ids
        else:
            self.__cams_categories = []

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            get_root_logger().error(err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        observations = self.dataset.get_observations_from_categories(categories)
        self.__start_iterator(observations, show_predictions=show_predictions)

    def __support_visualize_annotations_tp_fp_fn_tn(self, get_ids_function, categories):
        """
        Support function for the visualization of false positive, false negative and true positive.
        Parameters
        ----------
        get_ids_function:
            function used to retrieve the ids to be shown
        categories: list, optional
            Categories to include. If None include all (default is None)
        """
        self.__cam_visualization = False
        categories_ds = self.dataset.get_categories_names()
        if categories is None:
            categories = [categories_ds[0]] if self.dataset.task_type == TaskType.CLASSIFICATION_BINARY else categories_ds
        elif not self.dataset.are_valid_categories(categories):
            return -1

        ids_per_cat = get_ids_function(categories)
        ids = {"gt": [], "props": []}
        for cat in categories:
            ids["gt"].extend(ids_per_cat[cat]["gt"])
            ids["props"].extend(ids_per_cat[cat]["props"])
        observations = self.dataset.get_observations_from_ids(ids["gt"])
        categories_ids = self.dataset.get_categories_id_from_names(categories)
        self.__ids_to_show = ids
        if self.dataset.task_type == TaskType.CLASSIFICATION_MULTI_LABEL:
            self.__start_iterator(observations, category=list(categories_ids), show_predictions=True)
        else:
            self.__start_iterator(observations, show_predictions=True)

    def __start_iterator(self, observations, category=None, show_predictions=False):
        """
        Starts the iteration over the observations

        Parameters
        ----------
        observations: pandas.DataFrame
            Observations to show
        category: str or array-like, optional
            Name of a specific category or list of categories names (default is None)
        show_predictions: bool, optional
            If True show also the predictions (default is False)
        """
        if show_predictions and not self.dataset.proposals:
            get_root_logger().warning("No proposals available. Please make sure to load proposals to the dataset.")
            show_predictions = False

        self.__current_observations = [i.to_dict() for b, i in observations.iterrows()]
        self.__show_predictions = show_predictions
        self.__current_category = category

        if len(self.__current_observations) == 0:
            print(labels_str.warn_no_images_criteria)
        else:
            self.__iterator = Iterator(self.__current_observations, show_name=False, image_display_function=self.__display_function)
            if self.__cam_visualization:
                children = self.__iterator.all_widgets.children
                cam_widgets = self.__create_cams_widgets()
                self.__iterator.all_widgets = VBox(children=[cam_widgets, VBox(children=children)])
            self.__iterator.start_iteration()
