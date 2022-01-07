from tabulate import tabulate
from odin.classes import strings as labels_str


class PropertyAnnotatorInterface:
    """
    This class is extended by the Annotator classes (one for each implemented annotation).
    Each subclass implements the function process_object that returns the annotation that must be added to the dataset.
    """
    def __init__(self, property_name, property_values, default_value=None):
        """
        property_values can be a list with all the possible values (e.g color case)
        or a range of integer (e.g object detection case) to be specified as range(x) or range(x, y)
        """
        self.property_name = property_name
        self.property_values = property_values
        self.DEFAULT_VALUE = default_value

    def process_object(self, data_object, dataset_abs_path=None):
        pass

    def print_results(self, dataset):
        """
        Print the results of the annotation for all the property values
        """
        if self.DEFAULT_VALUE is None:
            default_values = dataset[self.property_name].isnull().sum()
        else:
            default_values = len(dataset[dataset[self.property_name] == self.DEFAULT_VALUE])
        if default_values == len(dataset):
            print(f"No object has been annotated with property {self.property_name}")
        else:
            table = []
            for value in self.property_values:
                counter = len(dataset[dataset[self.property_name] == value])
                table.append([value, counter])
            print(tabulate(table, headers=[self.property_name, labels_str.info_ann_objects]))
