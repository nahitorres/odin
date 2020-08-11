import os
import sys
import copy
import numpy as np

from emd_with_classes.utils import *
from emd_with_classes.utils.utils import bb_intersection_over_union, sg_intersection_over_union
from .dataset import Dataset
from collections import defaultdict
from emd_with_classes.utils.draw_utils import plot_false_positive_errors, plot_class_distribution_of_fp, make_multi_category_plot, display_sensitivity_impact_plot, pie_plot
from collections import  Counter
logger = get_root_logger()


class Analyzer:

    __detector_name = 'detector'
    result_saving_path = './results/'
    dataset = None

    iou_thresh_weak = 0.1  # intersection/union threshold
    iou_thresh_strong = 0.5  # intersection/union threshold

    localization_level = "strong"  # week
    matched_category = {}
    use_normalization = True

    saved_results = {}

    __SAVE_PNG_GRAPHS = True  # if set to false graphs will just be displayed

    fp_errors = None

    def __init__(self, detector_name, dataset, result_saving_path='./results/', norm_factor=None, iou=None):
        if not isinstance(dataset, Dataset):
            logger.error("dataset_instance variable is not instance of DatasetInterface", sys.exc_info()[0])
            raise

        self.dataset = dataset
        self.is_segmentation = dataset.is_segmentation_ds()
        if not os.path.exists(result_saving_path):
            os.mkdir(result_saving_path)
        self.result_saving_path = os.path.join(result_saving_path, detector_name)
        if not os.path.exists(self.result_saving_path):
            os.mkdir(self.result_saving_path)
        self.__detector_name = detector_name
        if norm_factor is None:
            self.normalizer_factor = 1 / len(self.dataset.get_categories_names())
        else:
            self.normalizer_factor = norm_factor
        self.saved_results = {}
        if not iou is None:
            self.iou_thresh_strong = iou
        self.use_normalization = True
        self.matching_dict = None

    def __get_normalized_number_of_images(self):
        return self.normalizer_factor * self.dataset.get_number_of_images()

    def __intersection_over_union(self, ann, det):
        if self.is_segmentation:
            h, w = self.dataset.get_height_with_from_image(ann["image_id"])
            iou = sg_intersection_over_union(ann['segmentation'][0], det['segmentation'], h, w)
        else:
            bbox = ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + \
                   ann['bbox'][3]
            iou = bb_intersection_over_union(bbox, det['bbox'])
        return iou

    def _match_detection_with_ground_truth(self, gt, proposals, iou_thres):
        # Input
        # proposals is a list of proposals
        # gt is a list of objets with the location
        annotations_matched_id = []
        matching = []

        proposals = sorted(proposals, key=lambda x: x['confidence'], reverse=True)
        for index_det, det in enumerate(proposals):
            total_ious = []
            match_info = {"confidence": det["confidence"], "difficult": 0, "label": -1, "duplicated": 0, "iou": -1,
                          "det_id": det["id"], "ann_id": -1, "category_det": det["category_id"],
                          "category_ann": -1, "image_name": det["image_name"]}
            # For each annotation for the image calculate the Intersection Over Union
            image_id = self.dataset.get_image_id_from_image_name(det["image_name"])
            for index_ann, ann in enumerate(gt):
                if ann["image_id"] != image_id:
                    continue
                iou = self.__intersection_over_union(ann, det)
                total_ious.append([iou, ann])
            # Sort iou by higher score to evaluate prediction
            total_ious = sorted(total_ious, key=lambda k: k[0], reverse=True)
            ious = [i for i in total_ious if i[1]["category_id"] == det["category_id"]]
            if len(ious) > 0:
                iou, ann = ious[0]
                match_info = {"confidence": det["confidence"], "difficult": 0, "label": -1,
                              "iou": iou, "det_id": det["id"], "ann_id": ann["id"],
                              "category_det": det["category_id"], "category_ann": ann["category_id"], "image_name": det["image_name"]}
                not_used = False
                for iou, ann in ious:
                    if not ann["id"] in annotations_matched_id:
                        not_used = True
                        break
                if not_used:
                    # Corroborate is correct detection
                    if iou >= iou_thres:
                        # Difficult annotations are ignored
                        if "difficult" in ann.keys() and ann["difficult"] == 1:
                            match_info = {"confidence": det["confidence"], "difficult": 1, "label": 0, "iou": iou,
                                          "det_id": det["id"], "ann_id": ann["id"], "category_det": det["category_id"],
                                          "category_ann": ann["category_id"], "image_name": det["image_name"]}
                        else:

                            match_info = {"confidence": det["confidence"], "difficult": 0, "label": 1, "iou": iou,
                                          "det_id": det["id"], "ann_id": ann["id"], "category_det": det["category_id"],
                                          "category_ann": ann["category_id"], "image_name": det["image_name"]}
                            annotations_matched_id.append(ann["id"])
            elif len(total_ious) > 0:
                iou, ann = total_ious[0]
                # Save the max iou for the detection for later analysis
                match_info = {"confidence": det["confidence"], "difficult": 0, "label": -1,  "iou": iou,
                              "det_id": det["id"], "ann_id": ann["id"], "category_det": det["category_id"],
                              "category_ann": ann["category_id"], "image_name": det["image_name"]}

            matching.append(match_info)

        return matching

    def _evaluation_metric(self, gt, detections, matching):

        confidence = []
        label = []

        for match in matching:
            confidence.append(match["confidence"])
            label.append(match["label"])

        n_anns = len(gt)
        n_normalized = self.__get_normalized_number_of_images()

        return self.__average_precision_normalized(confidence, label, n_anns, n_normalized)

    def __average_precision_normalized(self, confidence, label, n_anns, n_normalized):

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

        if self.use_normalization:
            return ap_normlized, ap_norm_std
        else:
            return ap, ap_std

    def __calculate_metric_for_category(self, category):
        proposals = self.dataset.get_proposals_of_category(category)

        category_id = self.dataset.get_category_id_from_name(category)
        anns = self.dataset.get_anns_for_category(category_id)

        matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)
        value, value_std = self._evaluation_metric(anns, proposals, matching)
        return {"value": value, "std": value_std, "matching": matching}

    def analyze_property(self, property_name, possible_values, labels=None, show=True):
        if labels is None:
            labels = []
            for p in possible_values:
                display_name = self.dataset.get_display_name_of_property_value(property_name, p)
                if display_name is None:
                    labels.append(p)
                else:
                    labels.append(display_name)
        categories = self.dataset.get_categories_names()

        for category in categories:
            category_id = self.dataset.get_category_id_from_name(category)
            if not (category in self.saved_results.keys() and 'all' in self.saved_results[category].keys()):
                if not category in self.saved_results.keys():
                    self.saved_results[category] = {}
                self.saved_results[category]['all'] = self.__calculate_metric_for_category(category)
            matching = self.saved_results[category]['all']["matching"]
            self.saved_results[category][property_name] = self.__calculate_metric_for_properties_of_category(category,
                                                                                                         category_id,
                                                                                                         property_name,
                                                                                                         possible_values,
                                                                                                         matching)

            title = "Analysis of {} property".format(property_name)

        if show:
            make_multi_category_plot(self.saved_results, property_name, labels, title, self.__SAVE_PNG_GRAPHS,
                                     self.result_saving_path)


    def __calculate_metric_for_properties_of_category(self, category_name, category_id, property_name, possible_values,
                                                  matching):
        if property_name in self.saved_results[category_name].keys():
            return self.saved_results[category_name][property_name]

        properties_results = {}
        for value in possible_values:

            anns = self.dataset.get_annotations_of_class_with_property(category_id, property_name, value)
            ann_ids = [ann["id"] for ann in anns]

            property_match = []
            list_ids = []
            for match in matching:
                if match["label"] == 1 and not match["ann_id"] in ann_ids:
                    continue
                property_match.append(match)
                list_ids.append(match["det_id"])
            det = self.dataset.get_proposals_by_ids(category_name, list_ids)

            metricvalue, metric_std = self._evaluation_metric(anns, det, property_match)
            properties_results[value] = {"value": metricvalue, "std": metric_std}
        return properties_results

    def analyze_properties(self, properties=None):
        if properties is None:
            properties = self.dataset.get_property_keys()
        else:
            properties = [p for p in properties if p in self.dataset.get_property_keys()]
        if len(properties) == 0:
            logger.warn("No properties to analyze")

        for pkey in properties:
            values = self.dataset.get_values_for_property(pkey)
            self.analyze_property(pkey, values)

    def analyze_sensitivity_impact_of_properties(self, properties=None):
        if properties is None:
            properties = self.dataset.get_property_keys()
        else:
            properties = [p for p in properties if p in self.dataset.get_property_keys()]
        if len(properties) == 0:
            logger.warn("No properties to analyze")
        display_names = [self.dataset.get_display_name_of_property(pkey) for pkey in properties]

        for pkey in properties:
            values = self.dataset.get_values_for_property(pkey)
            self.analyze_property(pkey, values, show=False)

        display_sensitivity_impact_plot(self.saved_results, self.result_saving_path, properties,
                                                   display_names, self.__SAVE_PNG_GRAPHS)


    def get_matching_dict(self):
        if self.matching_dict is None:
            proposals = self.dataset.get_proposals()
            anns = self.dataset.get_annotations()
            self.matching_dict = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)
        return self.matching_dict
    ## Analyze the false positive errors in the whole dataset
    def __analyze_false_positive_errors(self):
        if not self.fp_errors is None:
            return self.fp_errors
        self.fp_errors = {}


        fp_classes = defaultdict(int)
        for category in self.dataset.get_categories_names():
            proposals = self.dataset.get_proposals_of_category(category)
            anns = self.dataset.get_annotations()

            matching = self._match_detection_with_ground_truth(anns, proposals, self.iou_thresh_strong)

            localization_indexes, bg_indexes, other_indexes, similar_indexes = [], [], [], []

            for match in matching:
                if match["label"] == -1:
                    fp_classes[category] += 1
                    if match["iou"] > self.iou_thresh_weak:
                        if match["category_det"] == match["category_ann"]:
                            localization_indexes.append(match["det_id"])
                        else:
                            if self.dataset.is_similar(match["category_det"], match["category_ann"]):
                                similar_indexes.append(match["det_id"])
                            else:
                                other_indexes.append(match["det_id"])
                    else:
                        bg_indexes.append(match["det_id"])
            self.fp_errors[category] = {"localization": localization_indexes,
                                        "similar": similar_indexes,
                                        "other": other_indexes,
                                        "background": bg_indexes}
        self.fp_errors["distribution"] = fp_classes
        return self.fp_errors

    # Plots the distribution of false positive errors
    def get_fp_error_distribution(self):
        error_dict_total = self.__analyze_false_positive_errors()
        plot_class_distribution_of_fp(error_dict_total["distribution"], self.result_saving_path, self.__SAVE_PNG_GRAPHS)


    # Analyzes and plot the false positive errors of each category
    def analyze_false_positive_error_for_category(self, category):

        error_dict_total = self.__analyze_false_positive_errors()
        error_dict = error_dict_total[category]

        self.use_normalization = False
        values = self.__calculate_metric_for_category(category)
        self.use_normalization = True

        category_metric_value = values['value']
        matching = values["matching"]


        errors = ["localization", "similar", "other", "background"]
        error_values = []

        for error in errors:
            if len(error_dict[error]) == 0:
                error_values.append([category_metric_value, 0])
                continue

            # localization_error
            local_matching = []
            list_ids = []
            for match in matching:
                if match["det_id"] in error_dict[error]:
                    continue
                local_matching.append(match)
                list_ids.append(match["det_id"])
            det = self.dataset.get_proposals_by_ids(category, list_ids)
            anns = self.dataset.get_anns_for_category(self.dataset.get_category_id_from_name(category))
            self.use_normalization = False
            metric_value, _ = self._evaluation_metric(anns, det, local_matching)
            self.use_normalization = True

            count_error = len(error_dict[error])
            error_values.append([metric_value, count_error])

        plot_false_positive_errors(error_values, errors, category_metric_value, category, self.result_saving_path,
                                   self.__SAVE_PNG_GRAPHS)

    def analyze_false_positive_errors(self):
        self.get_fp_error_distribution()
        for category in self.dataset.get_categories_names():
            self.analyze_false_positive_error_for_category(category)

    def show_distribution_of_property(self, property_name):
        property_name_to_show = self.dataset.get_display_name_of_property(property_name)

        values = self.dataset.get_values_for_property(property_name)
        display_names = [self.dataset.get_display_name_of_property_value(property_name,v) for v in values]
        anns = self.dataset.get_annotations()


        c = Counter([ann[property_name] for ann in anns])
        sizes = []
        for pv in values:
            if not pv in c:
                sizes.append(0)
            else:
                sizes.append(c[pv])

        title = "Distribution of {}".format(property_name_to_show)
        output_path = os.path.join(self.result_saving_path, "distribution_total_{}.png".format(property_name))
        pie_plot(sizes, display_names, title, output_path, self.__SAVE_PNG_GRAPHS)

        labels = [c[:2] for c in self.dataset.get_categories_names()]
        for pv in values:
            sizes = []
            for cat_name in self.dataset.get_categories_names():
                cat_id = self.dataset.get_category_id_from_name(cat_name)
                sizes.append(len(self.dataset.get_annotations_of_class_with_property(cat_id, property_name, pv)))

            title = "Distribution of {} among categories".format(pv)
            output_path = os.path.join(self.result_saving_path, "distribution_{}_in_categories.png".format(pv))
            pie_plot(sizes, labels, title, output_path, self.__SAVE_PNG_GRAPHS)