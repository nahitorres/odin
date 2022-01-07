from enum import Enum


class ErrorType(Enum):
    BACKGROUND = 0
    LOCALIZATION = 1
    SIMILAR_CLASSES = 2
    OTHER = 3
