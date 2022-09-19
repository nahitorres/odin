import abc


class ScalerInterface(metaclass=abc.ABCMeta):
    """Interface for all scaler objects.

    This interface defines all the methods that a scaler must implement.
    """
    
    @abc.abstractmethod
    def transform(self, x):
        pass
    
    @abc.abstractmethod
    def inverse_transform(self, x):
        pass