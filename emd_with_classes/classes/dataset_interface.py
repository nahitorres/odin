import abc


class DatasetInterface(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def dataset_type_name(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load') and
                callable(subclass.load) or
                NotImplemented)

    @abc.abstractmethod
    def load(self):
        """Method to load dataset into memory"""
        raise NotImplementedError
