from enum import Enum


class ErrorType(Enum):
    BACKGROUND = "background"
    LOCALIZATION = "localization"
    SIMILAR_CLASSES = "sim"
    DUPLICATED = "duplicated"
    LOC_SIM = "loc+sim"
    LOC_CLASS = "loc+class"
    CLASSIFICATION = "class"
