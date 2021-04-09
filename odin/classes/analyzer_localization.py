import copy
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from abc import ABC

from sklearn.metrics import auc

from odin.classes import DatasetLocalization
from odin.classes.analyzer_interface import AnalyzerInterface
from odin.utils import get_root_logger
from odin.utils.draw_utils import plot_reliability_diagram, plot_false_positive_errors, pie_plot
from odin.utils.utils import sg_intersection_over_union, bb_intersection_over_union

logger = get_root_logger()


class AnalyzerLocalization(AnalyzerInterface, ABC):

    is_segmentation = False

    matching_dict = None

    iou_thresh_weak = 0.1  # intersection/union threshold
    iou_thresh_strong = 0.5  # intersection/union threshold
    localization_level = "strong"  # week

    __SAVE_PNG_GRAPHS = True

    __normalized_number_of_images = 1000

    __valid_metrics = ['precision_score', 'recall_score', 'f1_score', 'average_precision_score', 'precision_recall_auc',
                       'f1_auc', 'average_precision_interpolated', 'custom']

    __valid_curves = ['precision_recall_curve', 'f1_curve']

    def __init__(self, detector_name, dataset, result_saving_path='./results/', use_normalization=True,
                 norm_factor_categories=None, norm_factors_properties=None, iou=None, conf_thresh=None,
                 metric='average_precision_score', save_graphs_as_png=True):

        if type(dataset) is not DatasetLocalization:
            raise TypeError("Invalid type for 'dataset'. Use DatasetLocalization.")
        if not dataset.get_property_keys():
            raise Exception("No properties available.")

        self.is_segmentation = dataset.is_segmentation_ds()

        if iou is not None:
            self.iou_thresh_strong = iou
        if conf_thresh is None:
            conf_thresh = 0.5

        self.__SAVE_PNG_GRAPHS = save_graphs_as_png

        super().__init__(detector_name, dataset, result_saving_path, use_normalization, norm_factor_categories,
                         norm_factors_properties, conf_thresh, metric, self.__valid_metrics, self.__valid_curves, False,
                         self.__SAVE_PNG_GRAPHS)

    def analyze_reliability(self, categories=None, num_bins=10):
        if num_bins < 2:
            logger.error("Minimum number of bins is 2")
            return
        if num_bins > 50:
            logger.error("Maximum number of bins is 50")
            return
        if categories is None:
            anns = self.dataset.get_annotations()
            proposals = self.dataset.get_proposals()
            matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)

            numpy_confidence, numpy_label = self.__support_reliability(matching)
            result = self._calculate_reliability(None, numpy_label, numpy_confidence, num_bins)
            plot_reliability_diagram(result, self.__SAVE_PNG_GRAPHS, self.result_saving_path, is_classification=False)
        else:
            if not self._are_valid_categories(categories):
                return
            for category in categories:
                anns = self.dataset.get_annotations()
                proposals = self.dataset.get_proposals_of_category(category)
                matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)
                numpy_confidence, numpy_label = self.__support_reliability(matching)
                result = self._calculate_reliability(None, numpy_label, numpy_confidence, num_bins)
                plot_reliability_diagram(result, self.__SAVE_PNG_GRAPHS, self.result_saving_path,
                                         is_classification=False, category=category)

    def base_report(self, metrics=None, categories=None, properties=None, show_categories=True, show_properties=True):
        default_metrics = ['precision_score', 'recall_score', 'f1_score', 'average_precision_score']
        return self._get_report_results(default_metrics, metrics, categories, properties, show_categories,
                                        show_properties)

    def analyze_false_positive_error_for_category(self, category, categories=None, metric=None):
        if categories is None or category not in categories:
            categories = [category]
        elif not self._are_valid_categories(categories):
            return

        if category not in self.dataset.get_categories_names():
            logger.error("Invalid category name")
            return

        if metric is None:
            metric = self.metric
        elif not self._is_valid_metric(metric):
            return

        error_dict_total = self._analyze_false_positive_errors(categories)
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
            list_ids = matching_tmp["det_id"].values

            det = self.dataset.get_proposals_by_ids(category, list_ids)
            anns = self.dataset.get_anns_for_category(self.dataset.get_category_id_from_name(category))

            self._set_normalized_number_of_images_for_categories()

            metric_value, _ = self._compute_metric(anns, det, matching_tmp, metric)

            count_error = len(error_dict[error])
            error_values.append([metric_value, count_error])

        plot_false_positive_errors(error_values, errors, category_metric_value, category, metric,
                                   self.result_saving_path, self.__SAVE_PNG_GRAPHS)

    def show_distribution_of_property(self, property_name):
        if property_name not in self.dataset.get_property_keys():
            logger.error(f"Property '{property_name}' not valid.")
            return
        property_name_to_show = self.dataset.get_display_name_of_property(property_name)

        values = self.dataset.get_values_for_property(property_name)
        display_names = [self.dataset.get_display_name_of_property_value(property_name, v) for v in values]

        anns = self.dataset.get_annotations()
        count = anns.groupby(property_name).size()
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

        labels = self.dataset.get_categories_names()
        for pv in values:
            sizes = []
            for cat_name in self.dataset.get_categories_names():
                cat_id = self.dataset.get_category_id_from_name(cat_name)
                sizes.append(len(self.dataset.get_annotations_of_class_with_property(cat_id, property_name, pv).index))

            title = "Distribution of {} among categories".format(pv)
            output_path = os.path.join(self.result_saving_path, "distribution_{}_in_categories.png".format(pv))
            pie_plot(sizes, labels, title, output_path, self.__SAVE_PNG_GRAPHS)

    def get_matching_dict(self):
        if self.matching_dict is None:
            proposals = self.dataset.get_proposals()
            anns = self.dataset.get_annotations()
            self.matching_dict = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)
        return self.matching_dict

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
                                             self.dataset.get_number_of_images()

    def _set_normalized_number_of_images_for_property_for_categories(self, property_name):
        self.__normalized_number_of_images = self._norm_factors[property_name] * \
                                             self._norm_factors["categories"] * \
                                             self.dataset.get_number_of_images()

    def _match_detection_with_ground_truth(self, gt, proposals, iou_thres):
        # Input
        # proposals is a list of proposals
        # gt is a list of objets with the location
        annotations_matched_id = []
        matching = []

        imgs = self.dataset.images
        anns = pd.merge(gt, imgs, how="left", left_on="image_id", right_on="id")
        anns = anns.groupby("image_id")

        props = proposals.sort_values(["confidence", "id"], ascending=[False, True])
        if self.dataset.match_on_filename:
            props["image_id"] = props[self.dataset.match_param_props].apply(lambda x: self.dataset.get_image_id_from_image_name(x))


        for index, det in props.iterrows():
            total_ious = []
            match_info = {"confidence": det["confidence"], "difficult": 0, "label": -1, "duplicated": 0, "iou": -1,
                          "det_id": det["id"], "ann_id": -1, "category_det": det["category_id"],
                          "category_ann": -1}
            # For each annotation for the image calculate the Intersection Over Union
            try:
                if self.dataset.match_on_filename:
                    img_id = det["image_id"]
                else:
                    img_id = det[self.dataset.match_param_props]
                for i, ann in anns.get_group(img_id).iterrows():
                    iou = self.__intersection_over_union(ann, det)
                    total_ious.append([iou, ann])
            except KeyError:
                pass

            # Sort iou by higher score to evaluate prediction
            total_ious = sorted(total_ious, key=lambda k: k[0], reverse=True)
            ious = [i for i in total_ious if i[1]["category_id"] == det["category_id"]]
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
        return pd.DataFrame(matching)

    def _compute_metric(self, gt, detections, matching, metric, is_micro_required=False):
        if metric == 'precision_score':
            return self._compute_metric_precision_score(gt, detections, matching)
        elif metric == 'recall_score':
            return self._compute_metric_recall_score(gt, detections, matching)
        elif metric == 'f1_score':
            return self._compute_metric_f1_score(gt, detections, matching)
        elif metric == 'average_precision_score':
            return self._compute_average_precision_score(gt, detections, matching)
        elif metric == 'precision_recall_auc':
            return self._compute_precision_recall_auc_curve(gt, detections, matching)
        elif metric == 'f1_auc':
            return self._compute_f1_auc_curve(gt, detections, matching)
        elif metric == 'average_precision_interpolated':
            return self._compute_average_precision_interpolated(gt, detections, matching)
        elif metric == 'custom':
            return self._evaluation_metric(gt, detections, matching, is_micro_required)
        else:
            raise NotImplementedError("metric '{}' unknown".format(metric))

    def _get_input_report(self, properties, show_properties_report):
        input_report = {"total": {}}
        anns = self.dataset.get_annotations()
        proposals = self.dataset.get_proposals()
        all_matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)

        input_report["total"]["all"] = {"anns": anns,
                                        "props": proposals,
                                        "matching": all_matching}
        for cat in self.dataset.get_categories_names():
            anns = self.dataset.get_anns_for_category(self.dataset.get_category_id_from_name(cat))
            proposals = self.dataset.get_proposals_of_category(cat)
            prop_ids = proposals["id"].values
            cat_matching = all_matching[all_matching["det_id"].isin(prop_ids)]
            input_report["total"][cat] = {"anns": anns,
                                          "props": proposals,
                                          "matching": cat_matching}
        if show_properties_report:
            for property in properties:
                property_values = self.dataset.get_values_for_property(property)
                for p_value in property_values:
                    prop_value = property + "_" + "{}".format(p_value)
                    input_report[prop_value] = {}
                    # macro
                    for cat in self.dataset.get_categories_names():
                        anns = self.dataset.get_annotations_of_class_with_property(self.dataset
                                                                                   .get_category_id_from_name(cat),
                                                                                   property, p_value)
                        ann_ids = anns["id"].values
                        proposals = self.dataset.get_proposals_of_category(cat)
                        prop_ids = proposals["id"].values
                        property_match = all_matching[(all_matching["det_id"].isin(prop_ids)) &
                                                      ((all_matching["label"] != 1) |
                                                       (all_matching["ann_id"].isin(ann_ids)))]
                        list_ids = property_match["det_id"].values
                        proposals = self.dataset.get_proposals_with_ids(list_ids)
                        input_report[prop_value][cat] = {"anns": anns,
                                                         "props": proposals,
                                                         "matching": property_match}
        return input_report

    def _calculate_report_for_metric(self, input_report, categories, properties, show_categories, show_properties,
                                     metric):
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
        tot_value = 0
        counter = 0
        for cat in self.dataset.get_categories_names():
            value, _ = self._compute_metric(input_report["total"][cat]["anns"],
                                               input_report["total"][cat]["props"],
                                               input_report["total"][cat]["matching"],
                                               metric)
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
                    tot_value = 0
                    counter = 0
                    for cat in self.dataset.get_categories_names():
                        value, _ = self._compute_metric(input_report[p_value][cat]["anns"],
                                                           input_report[p_value][cat]["props"],
                                                           input_report[p_value][cat]["matching"],
                                                           metric)
                        tot_value += value
                        counter += 1
                    results[p_value] = tot_value / counter
        return results

    def _compute_average_precision_interpolated(self, gt, detections, matching):
        n_anns = len(gt.index)
        n_normalized = self._get_normalized_number_of_images()

        confidence = matching["confidence"].values
        label = matching["label"].values

        return self._average_precision_normalized(confidence, label, n_anns, n_normalized)

    def _calculate_metric_for_category(self, category, metric):
        proposals = self.dataset.get_proposals_of_category(category)
        category_id = self.dataset.get_category_id_from_name(category)
        anns = self.dataset.get_anns_for_category(category_id)
        matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)
        self._set_normalized_number_of_images_for_categories()
        value, value_std = self._compute_metric(anns, proposals, matching, metric)
        return {"value": value, "std": value_std, "matching": matching}

    def _analyze_true_positive_for_categories(self, categories):
        tp_classes = defaultdict(int)
        anns = self.dataset.get_annotations()
        for category in categories:
            proposals = self.dataset.get_proposals_of_category(category)
            matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)

            matching = matching[(matching["label"] == 1) & (matching["confidence"] >= self.conf_thresh)]
            tp_classes[category] = len(matching.index)
        return tp_classes

    def _analyze_false_negative_for_categories(self, categories):
        fn_classes = defaultdict(int)
        for category in categories:
            anns = self.dataset.get_anns_for_category(self.dataset.get_category_id_from_name(category))
            proposals = self.dataset.get_proposals_of_category(category)
            matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)
            matching = matching[(matching["label"] == 1) & (matching["confidence"] >= self.conf_thresh)]
            fn_classes[category] = len(anns.index) - len(matching.index)
        return fn_classes

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
        anns = self.dataset.get_annotations()
        for category in categories:
            proposals = self.dataset.get_proposals_of_category(category)
            matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)

            matching = matching[(matching["label"] == -1) & (matching["confidence"] >= self.conf_thresh)]

            fp_classes[category] = len(matching.index)

            # background errors
            matching["bg_error"] = np.where(matching["iou"] <= self.iou_thresh_weak, 1, 0)
            bg_indexes = matching[matching["bg_error"] == 1]["det_id"].values

            # localization errors
            matching = matching[matching["iou"] > self.iou_thresh_weak]
            matching["loc_error"] = np.where((matching["category_det"] == matching["category_ann"]), 1, 0)
            localization_indexes = matching[matching["loc_error"] == 1]["det_id"].values

            # similar and other errors
            matching = matching[(matching["category_det"] != matching["category_ann"])]
            matching["similar_others_error"] = matching.apply(
                lambda x: 1 if self.dataset.is_similar(x["category_det"], x["category_ann"]) else 2, axis=1)
            similar_indexes = matching[matching["similar_others_error"] == 1]["det_id"].values
            other_indexes = matching[matching["similar_others_error"] == 2]["det_id"].values

            self.fp_errors[category] = {"localization": localization_indexes,
                                        "similar": similar_indexes,
                                        "other": other_indexes,
                                        "background": bg_indexes}
        self.fp_errors["distribution"] = fp_classes
        return self.fp_errors

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
        for value in possible_values:
            anns = self.dataset.get_annotations_of_class_with_property(category_id, property_name, value)
            ann_ids = anns["id"].values

            property_match = matching[(matching["label"] != 1) | (matching["ann_id"].isin(ann_ids))]
            list_ids = property_match["det_id"].values

            det = self.dataset.get_proposals_by_ids(category_name, list_ids)

            metricvalue, metric_std = self._compute_metric(anns, det, property_match, metric)
            properties_results[value] = {"value": metricvalue, "std": metric_std}
        return properties_results

    def __intersection_over_union(self, ann, det):
        if self.is_segmentation:
            try:
                iou = sg_intersection_over_union(ann['segmentation'][0], det['segmentation'], ann["height"],
                                                 ann["width"])
            except:
                h, w = self.dataset.get_height_width_from_image(ann["image_id"])
                iou = sg_intersection_over_union(ann['segmentation'][0], det['segmentation'], h, w)
        else:
            bbox = ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + \
                   ann['bbox'][3]
            iou = bb_intersection_over_union(bbox, det['bbox'])
        return iou

    def __support_reliability(self, matching):
        match = matching[(matching["iou"] >= self.iou_thresh_weak) & (matching["confidence"] >= self.conf_thresh)]

        numpy_confidence = match["confidence"].values
        numpy_label = match["label"].values
        return numpy_confidence, numpy_label

    def _calculate_reliability(self, y_true, y_pred, y_score, num_bins):
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        indices = np.digitize(y_score, bins, right=True)
        bin_precisions = np.zeros(num_bins, dtype=np.float)
        bin_confidences = np.zeros(num_bins, dtype=np.float)
        bin_counts = np.zeros(num_bins, dtype=np.int)

        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                v_selected = y_pred[selected]
                tp = np.cumsum(v_selected == 1)
                fp = np.cumsum(v_selected == -1)
                bin_precisions[b] = np.mean(np.divide(tp, np.add(tp, fp)))
                bin_confidences[b] = np.mean(y_score[selected])
                bin_counts[b] = len(selected)
        avg_prec = np.sum(bin_precisions * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

        gaps = bin_confidences - bin_precisions
        ece = np.sum(np.abs(gaps) * bin_counts) / np.sum(bin_counts)
        mce = np.max(np.abs(gaps))

        result = {'values': bin_precisions, 'gaps': gaps, 'counts': bin_counts, 'bins': bins,
                  'avg_value': avg_prec, 'avg_conf': avg_conf, 'ece': ece, 'mce': mce}
        return result

    def _support_metric_threshold(self, n_true_gt, n_normalized, gt_ord, det_ord, tp, fp, threshold):
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
        np.warnings.filterwarnings("ignore")
        try:
            tp_norm = tp * n_normalized / n_true_gt
        except ZeroDivisionError:
            tp_norm = 0
        return tp, tp_norm, fp

    def _compute_metric_precision_score(self, gt, detections, matching):
        n_anns, n_normalized, numpy_confidence, _, tp, fp = self._support_metric(gt, None, matching)
        tp, tp_norm, fp = self._support_metric_threshold(n_anns, n_normalized, None, numpy_confidence, tp, fp,
                                                         self.conf_thresh)

        precision, precision_norm = self._support_precision_score(tp, tp_norm, fp)

        if self._use_normalization:
            return precision_norm, None
        else:
            return precision, None

    def _compute_metric_recall_score(self, gt, detections, matching):
        n_anns, n_normalized, numpy_confidence, _, tp, fp = self._support_metric(gt, None, matching)
        tp, tp_norm, fp = self._support_metric_threshold(n_anns, n_normalized, None, numpy_confidence, tp, fp,
                                                         self.conf_thresh)
        fn = n_anns - tp

        recall, recall_norm = self._support_recall_score(tp, tp_norm, fn)
        if self._use_normalization:
            return recall_norm, None
        else:
            return recall, None

    def _compute_metric_f1_score(self, gt, detections, matching):
        n_anns, n_normalized, numpy_confidence, _, tp, fp = self._support_metric(gt, None, matching)
        tp, tp_norm, fp = self._support_metric_threshold(n_anns, n_normalized, None, numpy_confidence, tp, fp,
                                                         self.conf_thresh)

        fn = n_anns - tp

        f1, f1_norm = self._support_f1_score(tp, tp_norm, fp, fn)
        if self._use_normalization:
            return f1_norm, None
        else:
            return f1, None

    def _compute_curve_for_categories(self, categories, curve):
        results = {}
        self._set_normalized_number_of_images_for_categories()
        for category in categories:
            proposals = self.dataset.get_proposals_of_category(category)
            category_id = self.dataset.get_category_id_from_name(category)
            anns = self.dataset.get_anns_for_category(category_id)
            matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)
            if curve == "precision_recall_curve":
                x_values, y_values = self._compute_precision_recall_curve(anns, proposals, matching)
            elif curve == "f1_curve":
                x_values, y_values = self._compute_f1_curve(anns, proposals, matching)
            else:
                raise ValueError("Invalid curve name.")
            auc_value = auc(x_values, y_values)
            results[category] = {'auc': auc_value,
                                 'x': x_values,
                                 'y': y_values}
        return results

    def _compute_precision_recall_curve(self, gt, detections, matching):
        n_anns, n_normalized, numpy_confidence, numpy_label, tp, fp = self._support_metric(gt, None, matching)
        precision, precision_norm, recall, recall_norm = self._support_precision_recall_auc(len(detections), n_anns,
                                                                                            n_normalized,
                                                                                            numpy_confidence, tp, fp,
                                                                                            False)
        if self._use_normalization:
            return recall_norm, precision_norm
        else:
            return recall, precision

    def _compute_precision_recall_auc_curve(self, gt, detections, matching):
        recall, precision = self._compute_precision_recall_curve(gt, detections, matching)
        std_err = np.std(precision) / np.sqrt(precision.size)
        return auc(recall, precision), std_err

    def _compute_f1_curve(self, gt, detections, matching):
        n_anns, n_normalized, numpy_confidence, numpy_label, tp, fp = self._support_metric(gt, None, matching)
        precision, precision_norm, recall, recall_norm, rel_indexes = self._support_precision_recall(n_anns,
                                                                                                     n_normalized,
                                                                                                     numpy_confidence,
                                                                                                     tp, fp)
        return self._support_f1_curve(numpy_confidence, precision, precision_norm, recall, recall_norm, rel_indexes)

    def _compute_f1_auc_curve(self, gt, detections, matching):
        thresholds, f1 = self._compute_f1_curve(gt, detections, matching)
        std_err = np.std(f1) / np.sqrt(f1.size)
        return auc(thresholds, f1), std_err

    def _compute_average_precision_score(self, gt, detections, matching):
        n_anns, n_normalized, numpy_confidence, label, tp, fp = self._support_metric(gt, None, matching)

        metric_value, std_err = self._support_average_precision(len(detections), n_anns, n_normalized,
                                                                numpy_confidence, tp, fp, False)
        if np.isnan(metric_value):
            metric_value = 0

        return metric_value, std_err

    def _support_metric(self, gt, detections, matching):

        n_anns = len(gt.index)
        n_normalized = self._get_normalized_number_of_images()

        numpy_confidence = matching["confidence"].values
        numpy_label = matching["label"].values

        tp = np.cumsum((numpy_label == 1).astype(int))  # True positive cumsum
        fp = np.cumsum((numpy_label == -1).astype(int))  # False Positive cumsum
        return n_anns, n_normalized, numpy_confidence, numpy_label, tp, fp

    def _average_precision_normalized(self, confidence, label, n_anns, n_normalized):

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

        ap_normlized = np.mean(prec_norm[is_tp]) * recall[-1]
        ap_norm_std = np.std(np.concatenate((prec_norm[is_tp], missed))) / np.sqrt(n_anns)
        if self._use_normalization:
            return ap_normlized, ap_norm_std
        else:
            return ap, ap_std