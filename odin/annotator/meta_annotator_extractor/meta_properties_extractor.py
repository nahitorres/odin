import json
import os
from enum import Enum
import pandas as pd
from tqdm.auto import tqdm
from odin.annotator.meta_annotator_extractor import AnnotatorColor, AnnotatorFaces

from odin.classes import strings as labels_str, DatasetClassification
from odin.classes.safe_writer import SafeWriter
from odin.classes.strings import err_type
from odin.utils.env import get_root_logger

pd.options.mode.chained_assignment = None


class MetaProperties(Enum):
    COLOR = "color"
    FACES = "faces"


class MetaPropertiesExtractor:

    def __init__(self,
                 dataset,
                 properties=None,  # list of properties to add
                 output_path=None,
                 ):
        """
        The MetaPropertiesExtractor class allows the automatic extraction of meta-annotations from the observations

        Parameters
        ----------
        dataset: DatasetClassification
            Data set used to perform meta-annotation extraction
        properties: MetaProperties, optional
            List of the meta-annotations to be included in the extraction. If None, no default meta-annotation
            will be extracted
        output_path: str, optional
            Path used to save the new generated annotated data set. If not provided, it is saved in the same location of the old data set. (default is None)
        """
        if not isinstance(dataset, DatasetClassification):
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
        self.name = (dataset.dataset_root_param.split('/')[-1]).split('.')[0]  # get original file name
        self.__set_output(output_path)

        print("{} {}".format(labels_str.info_new_ds, self.file_path_for_json))

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
            if p == MetaProperties.COLOR:
                annotator = AnnotatorColor()
                annotators.append(annotator)
                print(f"Property '{p.value}' will be add to the dataset")
            elif p == MetaProperties.FACES:
                annotator = AnnotatorFaces()
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
            name = self.dataset.dataset_root_param
            self.output_directory = name.replace(os.path.basename(name), "")
        else:
            self.output_directory = output_path

        self.file_path_for_json = os.path.join(self.output_directory, self.name + "_ANNOTATED.json")

    def start_annotation(self):
        """
        For each object in the dataset, perform all the possible annotations.
        """
        if len(self.annotators) == 0:
            get_root_logger().error("No meta-annotator extractor available. Please be sure to select one from the "
                                    "default or to provide your custom one")
            return
        observations = self.dataset.get_all_observations()
        for i, data_object in tqdm(observations.iterrows(), total=len(observations.index)):
            for annotator in self.annotators:
                annotation = annotator.process_object(data_object, self.dataset.images_abs_path)
                self.dataset.observations.loc[self.dataset.observations["id"] == data_object["id"], annotator.property_name] = annotation

        self.__save_output()  # at the for end, save the output

    def __save_output(self):
        """
        Save the dataset annotated into the output file
        """
        print("End Annotation.")
        dataset_annotated = {
            "observations": json.loads(self.dataset.get_all_observations().to_json(orient='records'))
        }
        w = SafeWriter(os.path.join(self.file_path_for_json), 'w')
        w.write(json.dumps(dataset_annotated, indent=4))
        w.close()
        print("Output file saved.")

    def print_results(self):
        """
        For each property print the obtained results
        """
        for annotator in self.annotators:
            annotator.print_results(self.dataset.get_all_observations())

    def add_custom_property(self, annotator):
        """
        annotators is a list
        """
        self.annotators.append(annotator)
        print(f"Property '{annotator.property_name}' will be add to the dataset")

        # Create a new column for the dataset, every element initialized to a default value
        self.dataset.observations[annotator.property_name] = annotator.DEFAULT_VALUE
