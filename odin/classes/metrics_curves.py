from enum import Enum


class Metrics(Enum):
    ACCURACY = "Accuracy"
    ERROR_RATE = "Error_Rate"
    PRECISION_SCORE = "Precision"
    RECALL_SCORE = "Recall"
    F1_SCORE = "F1"
    F_BETA_SCORE = "Fbeta"
    FALSE_ALARM_RATE = "False alarm rate"
    MISS_ALARM_RATE = "Miss alarm rate"
    AVERAGE_PRECISION_SCORE = "Average_Precision"
    AVERAGE_PRECISION_INTERPOLATED = "Average_Precision_Interpolated"
    ROC_AUC = "ROC_AUC"
    PRECISION_RECALL_AUC = "Precision-Recall_AUC"
    F1_AUC = "F1_AUC"
    CAM_COMPONENT_IOU = "Component_IoU"
    CAM_GLOBAL_IOU = "Global_IoU"
    CAM_BBOX_COVERAGE = "Bbox_coverage"
    CAM_IRRELEVANT_ATTENTION = "Irrelevant attention"
    MAE = "Mean absolute error"
    MSE = "Mean squared error"
    NAB_SCORE = "NAB score"
    RMSE = "Root mean squared error"
    MAPE = "Mean absolute percentage error"
    MARRE = "Mean absolute ranged relative error"
    OPE = "Overall Percentage Error"
    RMSLE = "Root mean squared log error"
    SMAPE = "Symmetric Mean absolute percentage error"
    MATTHEWS_COEF = "Matthews correlation coefficient"
    COEFFICIENT_VARIATION = "Coefficient of variation"
    COEFFICIENT_DETERMINATION = "Coefficient of determination"


class Curves(Enum):
    PRECISION_RECALL_CURVE = 'Precision-Recall_Curve'
    ROC_CURVE = 'ROC_Curve'
    F1_CURVE = 'F1_Curve'
