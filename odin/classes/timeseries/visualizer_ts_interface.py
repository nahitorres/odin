import abc


class VisualizerTimeSeriesInterface(metaclass=abc.ABCMeta):

    def __init__(self, dataset, analyzers):
        self.dataset = dataset
        self.analyzers = analyzers

    @abc.abstractmethod
    def show(self):
        pass