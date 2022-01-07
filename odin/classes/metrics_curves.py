from enum import Enum


class Metrics(Enum):
    ACCURACY = "Accuracy"
    ERROR_RATE = "Error_Rate"
    PRECISION_SCORE = "Precision"
    RECALL_SCORE = "Recall"
    F1_SCORE = "F1"
    AVERAGE_PRECISION_SCORE = "Average_Precision"
    AVERAGE_PRECISION_INTERPOLATED = "Average_Precision_Interpolated"
    ROC_AUC = "ROC_AUC"
    PRECISION_RECALL_AUC = "Precision-Recall_AUC"
    F1_AUC = "F1_AUC"
    CAM_COMPONENT_IOU = "Component_IoU"
    CAM_GLOBAL_IOU = "Global_IoU"
    CAM_BBOX_COVERAGE = "Bbox_coverage"
    CAM_IRRELEVANT_ATTENTION = "Irrelevant attention"


class Curves(Enum):
    PRECISION_RECALL_CURVE = 'Precision-Recall_Curve'
    ROC_CURVE = 'ROC_Curve'
    F1_CURVE = 'F1_Curve'
