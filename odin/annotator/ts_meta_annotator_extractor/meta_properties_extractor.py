import json
import os
from enum import Enum
import pandas as pd
from tqdm.auto import tqdm
from odin.annotator.ts_meta_annotator_extractor import AnnotatorMonth, AnnotatorDuration

from odin.classes import strings as labels_str
from odin.classes.timeseries import DatasetTSAnomalyDetection
from odin.classes.safe_writer import SafeWriter
from odin.classes.strings import err_type
from odin.utils.env import get_root_logger

pd.options.mode.chained_assignment = None


class MetaProperties(Enum):
    MONTH = "month"
    DURATION = "duration"

class MetaPropertiesExtractor:

    def __init__(self,
                 dataset,
                 properties=None,  # list of properties to add
                 output_path="./metaproperties.csv",
                 exclude_columns = ["anomaly", "anomaly_window", "value"]
                 ):
        """
        The MetaPropertiesExtractor class allows the automatic extraction of meta-annotations from the observations

        Parameters
        ----------
        dataset: DatasetTSAnomalyDetection
            Data set used to perform meta-annotation extraction
        properties: MetaProperties, optional
            List of the meta-annotations to be included in the extraction. If None, no default meta-annotation
            will be extracted
        output_path: str, optional
            Path used to save the new generated annotated data set. If not provided, it is saved in the same location of the old data set. (default is None)
         exclude_columns: list, optional
             Columns of the original dataset that must be excluded from the meta-properties files
        """
        if not isinstance(dataset, DatasetTSAnomalyDetection):
            raise TypeError(err_type.format("dataset"))
        if properties is None:
            properties = []
        elif not isinstance(properties, list) or not all(isinstance(x, MetaProperties) for x in properties):
            raise TypeError(err_type.format("properties"))
        if output_path is not None and not isinstance(output_path, str):
            raise TypeError(err_type.format("output_path"))

        self.dataset = dataset
        self.annotators = self.__instantiate_annotators(properties)
        if len(self.annotators) == 0:
            get_root_logger().warning("No meta-annotator extractor selected")
            
        self.__create_new_properties_for_ds()
        self.name = dataset.dataset_path  # get original file name
        self.__set_output(output_path)
        
        self.output_path = output_path
        self.exclude_columns = exclude_columns

        print("{} {}".format(labels_str.info_new_ds, self.output_directory))

    def __create_new_properties_for_ds(self):
        """
        For each annotator, create a new column for the dataset,
        every element initialized to the corresponding default value
        """
        for a in self.annotators:
            self.dataset.observations[a.property_name] = a.DEFAULT_VALUE

    def __instantiate_annotators(self, properties):
        """
        Check which chosen properties are valid, for each instantiate the corresponding annotator.
        Return: True if all the chosen properties are valid, False otherwise
        """
        annotators = []
        for p in properties:
            if p == MetaProperties.MONTH:
                annotator = AnnotatorMonth()
                annotators.append(annotator)
                print(f"Property '{p.value}' will be add to the dataset")
            elif p == MetaProperties.DURATION:
                annotator = AnnotatorDuration()
                annotators.append(annotator)
                print(f"Property '{p.value}' will be add to the dataset")

            # here add new annotation if implemented
            else:
                print(f"Property '{p}' not valid")

        return annotators

    def __set_output(self, output_path):
        """
        Set the output path and the annotated dataset name.
        The name of the annotated dataset is the same of the original one.
        """
        if output_path is None:
            self.output_directory = "./metaproperties.csv"
        else:
            self.output_directory = output_path

        #self.file_path_for_json = os.path.join(self.output_directory, self.name + "_ANNOTATED.json")

    def start_annotation(self, single_row = True):
        """
        For each object in the dataset, perform all the possible annotations.
        If single_row is true, a single datum (e.g., a timestamp) is considered, otherwise the whole dataset is considered.
        """
        if len(self.annotators) == 0:
            get_root_logger().error("No meta-annotator extractor available. Please be sure to select one from the "
                                    "default or to provide your custom one")
            return
        
        properties = self.dataset.properties
        
        if single_row is True:
            for i, data_object in tqdm(self.dataset.observations.iterrows(), total=len(self.dataset.observations.index)):
                for annotator in self.annotators:
                    annotation = annotator.process_object(data_object)
                    self.dataset.properties.loc[self.dataset.properties.index == data_object._name, annotator.property_name] = annotation
        else:
            for annotator in self.annotators:
                if self.dataset.properties is None: # there are no loaded properties
                    self.dataset.properties = pd.DataFrame()
                    
                self.dataset.properties[annotator.property_name] = annotator.process_object(self.dataset.observations)
                    
                    

        self.__save_output()  # at the for end, save the output

    def __save_output(self):
        """
        Save the dataset annotated into the output file
        """
        print("End Annotation.")
        self.dataset.properties.to_csv(self.output_path)
        print("Output file saved.")

    def print_results(self):
        """
        For each property print the obtained results
        """
        for annotator in self.annotators:
            annotator.print_results(self.dataset.properties)

    def add_custom_property(self, annotator):
        """
        annotators is a list
        """
        self.annotators.append(annotator)
        print(f"Property '{annotator.property_name}' will be add to the dataset")

        # Create a new column for the dataset, every element initialized to a default value
        self.dataset.observations[annotator.property_name] = annotator.DEFAULT_VALUE
