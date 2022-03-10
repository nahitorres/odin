from enum import Enum


class TimeSeriesType(Enum):
    UNIVARIATE = 'univariate'
    MULTIVARIATE = 'multivariate'


class TSProposalsType(Enum):
    LABEL = 'label'
    REGRESSION = 'regression'
