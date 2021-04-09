from enum import Enum


class Metrics(Enum):
    ACCURACY = "accuracy"
    PRECISION_SCORE = "precision_score"
    RECALL_SCORE = "recall_score"
    F1_SCORE = "f1_score"
    AVERAGE_PRECISION_SCORE = "average_precision_score"
    ROC_AUC = "roc_auc"
    PRECISION_RECALL_AUC = "precision_recall_auc"
    CUSTOM = "custom"


class Curves(Enum):
    PRECISION_RECALL_CURVE = 'precision_recall_curve'
    ROC_CURVE = 'roc_curve'
    F1_CURVE = 'f1_curve'
