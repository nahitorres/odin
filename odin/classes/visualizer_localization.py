import os
import cv2
import numpy as np
from ipywidgets import Checkbox, HBox, VBox
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from .dataset_localization import DatasetLocalization
from odin.utils import Iterator, get_root_logger
from .error_type import ErrorType
from .strings import err_type, err_value
from .visualizer_interface import VisualizerInterface
from odin.classes import strings as labels_str, TaskType, AnalyzerLocalization


class VisualizerLocalization(VisualizerInterface):

    def __init__(self, dataset, analyzers=None):
        """
        The VisualizerLocalization class can be used to visualize the ground truth and the predictions of localization tasks, such as object detection and instance segmentation.

        Parameters
        ----------
        dataset: DatasetLocalization
            Data set used for the visualization
        analyzers: list, optional
            List of the analyzers of different models. (default is None)
        """

        if not isinstance(dataset, DatasetLocalization):
            raise TypeError(err_type.format("dataset"))

        analyzers_dict = {}
        if analyzers is not None:
            if isinstance(analyzers, AnalyzerLocalization):
                analyzers = [analyzers]
            elif not isinstance(analyzers, list):
                raise TypeError(err_type.format("analyzers"))
            for analyzer in analyzers:
                if not isinstance(analyzer, AnalyzerLocalization):
                    raise TypeError("Invalid analyzers instances. Please, provide 'AnalyzerLocalization' type")

                analyzers_dict[analyzer._model_name] = analyzer

        super().__init__(dataset, analyzers_dict)

        self.__iterator = None

        self.__colors = {}
        self.__show_predictions = False
        self.__ids_to_show = None

        self.__show_gt = True
        self.__show_model_predictions = {}

        colors = ['red', 'violet', 'darkcyan', 'dodgerblue', 'blue', 'orange']
        if self.dataset.proposals:
            for i, m in enumerate(self.dataset.proposals):
                self.__show_model_predictions[m] = True
                self.__colors[m] = colors[i] if i < len(colors) else i % len(colors)

    def __reset_show_models_predictions(self, value=True):
        for model in self.__show_model_predictions:
            self.__show_model_predictions[model] = value

    def __create_visualization_widgets(self):
        gt_chkbox = Checkbox(value=True, description="Show GT")
        gt_chkbox.observe(self.visualization_widget_changed, names=['value'])
        chkboxes = [gt_chkbox]
        for model in self.dataset.proposals:
            if self.__show_model_predictions[model]:
                m_chkbox = Checkbox(value=self.__show_predictions, description=model)
                m_chkbox.observe(self.visualization_widget_changed, names=['value'])
                chkboxes.append(m_chkbox)
        return HBox(chkboxes)

    def visualization_widget_changed(self, widget):
        if widget["owner"].description == "Show GT":
            self.__show_gt = widget["new"]
        else:
            self.__show_model_predictions[widget["owner"].description] = widget["new"]

        self.__iterator.refresh_output()

    def __show_segmentation_annotations(self, anns, predictions):
        ax = plt.gca()
        if self.__show_gt:
            for i, ann in anns.iterrows():
                seg_points = ann["segmentation"]
                for pol in seg_points:
                    poly = [[float(pol[i]), float(pol[i + 1])] for i in range(0, len(pol), 2)]
                    np_poly = np.array(poly)

                ax.add_patch(
                    Polygon(np_poly, linestyle='--', fill=False, facecolor='none', edgecolor='green', linewidth=2))
                ax.text(x=seg_points[0][0], y=seg_points[0][1], s=ann['category_id'], color='white', fontsize=9,
                        horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor='green'))
        if self.__show_predictions:
            for model in self.__show_model_predictions:
                if self.__show_model_predictions[model]:
                    matching = self.analyzers[model].get_matching_dataframe() if self.analyzers else None
                    for i, pred, in predictions[model].iterrows():
                        iou = matching.loc[matching["det_id"] == pred["id"]]["iou"].values[0] if matching is not None else None
                        seg_points = pred["segmentation"]
                        poly = [[float(seg_points[i]), float(seg_points[i + 1])] for i in range(0, len(seg_points), 2)]
                        np_poly = np.array(poly)

                        ax.add_patch(
                            Polygon(np_poly, linestyle='--', fill=False, facecolor='none',
                                    edgecolor=self.__colors[model],
                                    linewidth=2))
                        idx = int(len(seg_points) / 2)
                        if idx % 2 == 0:
                            idx += 1

                        bbox_str = f"{pred['category_id']}, conf:{pred['confidence']:.2f}"
                        if iou is not None:
                            bbox_str += f", iou:{iou:.2f}"
                        ax.text(x=seg_points[idx - 1], y=seg_points[idx],
                                s=bbox_str, color='white', fontsize=9,
                                horizontalalignment='left', verticalalignment='top',
                                bbox=dict(facecolor=self.__colors[model]))

    def __show_bbox_annotations(self, anns, predictions):
        ax = plt.gca()
        if self.__show_gt:
            for i, ann in anns.iterrows():
                bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4, 2))

                ax.add_patch(Polygon(np_poly, linestyle='--', facecolor='none', edgecolor='green', linewidth=3))
                ax.text(x=bbox_x, y=bbox_y, s=ann['category_id'], color='white', fontsize=9, horizontalalignment='left',
                        verticalalignment='top', bbox=dict(facecolor='green'))

        if self.__show_predictions:
            for model in self.__show_model_predictions:
                if self.__show_model_predictions[model]:
                    matching = self.analyzers[model].get_matching_dataframe() if self.analyzers else None
                    for i, pred in predictions[model].iterrows():
                        iou = matching.loc[matching["det_id"] == pred["id"]]["iou"].values[0] if matching is not None else None
                        bbox_x, bbox_y, bbox_w, bbox_h = pred['bbox']
                        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                                [bbox_x + bbox_w, bbox_y]]
                        np_poly = np.array(poly).reshape((4, 2))

                        ax.add_patch(Polygon(np_poly, linestyle='--', facecolor='none', edgecolor=self.__colors[model],
                                             linewidth=3))
                        bbox_str = f"{pred['category_id']}, conf:{pred['confidence']:.2f}"
                        if iou is not None:
                            bbox_str += f", iou:{iou:.2f}"
                        ax.text(x=bbox_x + bbox_w, y=bbox_y,
                                s=bbox_str,
                                color='white', fontsize=9, horizontalalignment='left', verticalalignment='top',
                                bbox=dict(facecolor=self.__colors[model]))

    def __show_image(self, image_path, index):
        """
        Shows the image with the meta-annotations specified previously by the user

        Parameters
        ----------
        image_path: str
            Path of the image
        index: int
            Index of the image in the variable '__current_images'
        """
        im_id = self.__current_images[index]["id"]

        print("Image with id:{}".format(im_id))
        if not os.path.exists(image_path):
            print("Image path does not exist: " + image_path)
        else:
            plt.figure(figsize=(10, 10))
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            plt.imshow(img)

            predictions = {}
            if self.__ids_to_show is None:
                if self.__current_category is None:
                    anns = self.dataset.get_annotations_from_image(im_id)
                    if self.__show_predictions:
                        for model in self.dataset.proposals:
                            predictions[model] = self.dataset.get_proposals_from_image_id(im_id, model)
                elif type(self.__current_category) == list:
                    anns = self.dataset.get_annotations_from_image_and_categories(im_id, self.__current_category)
                    if self.__show_predictions:
                        for model in self.dataset.proposals:
                            predictions[model] = self.dataset.get_proposals_from_image_id_and_categories(im_id,
                                                                                                         self.__current_category,
                                                                                                         model)
                else:
                    anns = self.dataset.get_annotations_from_image_and_categories(im_id, [self.__current_category])
                    if self.__show_predictions:
                        for model in self.dataset.proposals:
                            predictions[model] = self.dataset.get_proposals_from_image_id_and_categories(im_id,
                                                                                                         [
                                                                                                             self.__current_category],
                                                                                                         model)

                if self.__current_meta_anotation is not None and self.__meta_annotation_value is not None:
                    anns = anns[
                        anns.index.get_level_values(self.__current_meta_anotation) == self.__meta_annotation_value]
            else:
                img_anns = self.dataset.get_annotations_from_image(im_id).copy()
                anns = img_anns.loc[img_anns["id"].isin(self.__ids_to_show["gt"])]
                for model in self.dataset.proposals:
                    img_predictions = self.dataset.get_proposals_from_image_id(im_id, model).copy()
                    predictions[model] = img_predictions.loc[img_predictions["id"].isin(self.__ids_to_show["props"])]

            if self.dataset.task_type == TaskType.INSTANCE_SEGMENTATION and 'segmentation' in anns.columns.values:
                self.__show_segmentation_annotations(anns, predictions)
            else:
                self.__show_bbox_annotations(anns, predictions)

            plt.axis('off')
            plt.show()

    def visualize_annotations_for_ids(self, gt_ids, show_predictions=False):
        self.__ids_to_show = None
        self.__reset_show_models_predictions()
        if isinstance(gt_ids, int):
            gt_ids = [gt_ids]
        elif not isinstance(gt_ids, list):
            get_root_logger().error(err_type.format("id"))
            return -1

        if not isinstance(show_predictions, bool):
            get_root_logger().error(err_type.format("show_predictions"))
            return -1

        images = self.dataset.get_images_id_with_path_from_ids(gt_ids)
        self.__start_iterator(images, show_predictions=show_predictions)

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
        self.__ids_to_show = None
        self.__reset_show_models_predictions()

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            get_root_logger().error(labels_str.err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        if not isinstance(show_predictions, bool):
            get_root_logger().error(labels_str.err_type.format("show_predictions"))
            return -1

        images = []
        added_img_ids = []
        for c in categories:
            ii = self.dataset.get_images_id_with_path_for_category(c)
            for v in ii:
                if v['id'] in added_img_ids:
                    continue
                images.append(v)
                added_img_ids.append(v['id'])

        category_ids = [self.dataset.get_category_id_from_name(c) for c in categories]

        self.__start_iterator(images, category=category_ids, show_predictions=show_predictions)

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
        self.__reset_show_models_predictions()

        if not self.dataset.are_analyses_with_properties_available():
            get_root_logger().error("No properties available")
            return -1

        if not isinstance(meta_annotation, str):
            get_root_logger().error(labels_str.err_type.format("meta_annotation"))
            return -1
        elif not self.dataset.are_valid_properties([meta_annotation]):
            return -1

        if not isinstance(meta_annotation_value, str) and not isinstance(meta_annotation_value, float):
            get_root_logger().error(labels_str.err_type.format("meta_annotation_value"))
            return -1
        elif not self.dataset.is_valid_property(meta_annotation, [meta_annotation_value]):
            return -1

        if not isinstance(show_predictions, bool):
            get_root_logger().error(labels_str.err_type.format("show_predictions"))
            return -1

        images = self.dataset.get_images_id_with_path_with_property_value(meta_annotation, meta_annotation_value)

        self.__start_iterator(images, meta_annotation=meta_annotation, meta_annotation_value=meta_annotation_value,
                              show_predictions=show_predictions)

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
        self.__reset_show_models_predictions()

        if not self.dataset.are_analyses_with_properties_available():
            get_root_logger().error("No properties available")
            return -1

        if not isinstance(category, str):
            get_root_logger().error(labels_str.err_type.format("category"))
            return -1
        elif not self.dataset.is_valid_category(category):
            get_root_logger().error(err_value.format("category", self.dataset.get_categories_names()))
            return -1

        if not isinstance(meta_annotation, str):
            get_root_logger().error(labels_str.err_type.format("meta_annotation"))
            return -1
        elif not self.dataset.are_valid_properties([meta_annotation]):
            return -1

        if not isinstance(meta_annotation_value, str) and not isinstance(meta_annotation_value, float):
            get_root_logger().error(labels_str.err_type.format("meta_annotation_value"))
            return -1
        elif not self.dataset.is_valid_property(meta_annotation, [meta_annotation_value]):
            return -1

        if not isinstance(show_predictions, bool):
            get_root_logger().error(labels_str.err_type.format("show_predictions"))
            return -1

        images = self.dataset.get_images_id_with_path_for_category_with_property_value(category, meta_annotation,
                                                                                       meta_annotation_value)
        category_id = self.dataset.get_category_id_from_name(category)
        self.__start_iterator(images, category=category_id, meta_annotation=meta_annotation,
                              meta_annotation_value=meta_annotation_value, show_predictions=show_predictions)

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

        self.__reset_show_models_predictions(value=False)
        self.__show_model_predictions[model] = True

        if categories is not None and not isinstance(categories, list):
            get_root_logger().error(labels_str.err_type.format("categories"))
            return -1

        self.__support_visualize_annotations_tp_fp_fn(self.analyzers[model].get_true_positive_ids, categories)

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

        self.__reset_show_models_predictions(value=False)
        self.__show_model_predictions[model] = True

        if categories is not None and not isinstance(categories, list):
            get_root_logger().error(labels_str.err_type.format("categories"))
            return -1

        self.__support_visualize_annotations_tp_fp_fn(self.analyzers[model].get_false_positive_ids, categories)

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

        self.__reset_show_models_predictions(value=False)
        self.__show_model_predictions[model] = True

        if categories is not None and not isinstance(categories, list):
            get_root_logger().error(labels_str.err_type.format("categories"))
            return -1

        self.__support_visualize_annotations_tp_fp_fn(self.analyzers[model].get_false_negative_ids, categories)

    def visualize_annotations_for_error_type(self, error_type, categories=None, model=None):
        """
        It shows the ground truth and the predictions based on the false positive error type analysis.

        Parameters
        ----------
        error_type: ErrorType
            Error type to be included in the analysis. The types of error supported are: ErrorType.BACKGROUND, ErrorType.LOCALIZATION, ErrorType.SIMILAR_CLASSES, ErrorType.OTHER.
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

        self.__reset_show_models_predictions(value=False)
        self.__show_model_predictions[model] = True

        if not isinstance(error_type, ErrorType):
            get_root_logger().error(labels_str.err_type.format("error_type"))
            return -1

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not isinstance(categories, list):
            get_root_logger().error(labels_str.err_type.format("categories"))
            return -1
        elif not self.dataset.are_valid_categories(categories):
            return -1

        ids_per_cat = self.analyzers[model].get_false_positive_errors_ids(categories)
        ids = {"gt": [], "props": []}
        for cat in categories:
            ids["gt"].extend(ids_per_cat[cat][error_type.value]["gt"])
            ids["props"].extend(ids_per_cat[cat][error_type.value]["props"])

        self.__ids_to_show = ids
        if error_type == ErrorType.BACKGROUND:
            images = self.dataset.get_images_id_with_path_from_proposals_ids(ids["props"], model)
        else:
            images = self.dataset.get_images_id_with_path_from_annotation_ids(ids["gt"])
        self.__start_iterator(images, show_predictions=True)

    def __support_visualize_annotations_tp_fp_fn(self, get_ids_function, categories):
        """
        Support function for the visualization of false positive, false negative and true positive.
        Parameters
        ----------
        get_ids_function:
            function used to retrieve the ids to be shown
        categories: list, optional
            Categories to include. If None include all (default is None)
        """

        if categories is None:
            categories = self.dataset.get_categories_names()
        elif not self.dataset.are_valid_categories(categories):
            return -1

        ids_per_cat = get_ids_function(categories)
        ids = {"gt": [], "props": []}
        for cat in categories:
            ids["gt"].extend(ids_per_cat[cat]["gt"])
            ids["props"].extend(ids_per_cat[cat]["props"])
        self.__ids_to_show = ids
        images = self.dataset.get_images_id_with_path_from_annotation_ids(ids["gt"])

        self.__start_iterator(images, show_predictions=True)

    def __start_iterator(self, images, category=None, meta_annotation=None, meta_annotation_value=None,
                         show_predictions=False):
        """
        Starts the iteration over the images

        Parameters
        ----------
        images: list
            Dict that contains the id and path of the images
        category: str or array-like, optional
            Name of a specific category or list of categories names (default is None)
        meta_annotation: optional
            Property name (default is None)
        meta_annotation_value: optional
            Property value name (default is None)
        show_predictions: bool, optional
            If True show also the predictions (default is False)
        """
        if show_predictions and self.dataset.proposals is None:
            get_root_logger().warning("No proposals available. Please make sure to load proposals to the dataset.")
            show_predictions = False
        self.__show_predictions = show_predictions
        self.__current_category = category
        self.__current_images = images
        self.__current_meta_anotation = meta_annotation
        self.__meta_annotation_value = meta_annotation_value
        self.__show_gt = True
        paths = [img["path"] for img in images]
        if len(paths) == 0:
            print(labels_str.warn_no_images_criteria)
        else:
            self.__iterator = Iterator(paths, show_name=False, image_display_function=self.__show_image)
            if show_predictions:
                children = self.__iterator.all_widgets.children
                widgets = self.__create_visualization_widgets()
                self.__iterator.all_widgets = VBox(children=[widgets, VBox(children=children)])
            self.__iterator.start_iteration()
