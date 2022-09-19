import os
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from odin.annotator.meta_annotator_extractor import PropertyAnnotatorInterface


class AnnotatorFaces(PropertyAnnotatorInterface):
    """
    Subclass of PropertyAnnotatorInterface.
    Implement the faces annotation for each input image.
    """

    ONE = "0-1"
    FOUR = "2-4"
    MORE_FIVE = "5+"

    NAME = "faces"
    DEFAULT_VALUE = ""

    __face_detector = None

    def __init__(self):
        property_values = [self.ONE, self.FOUR, self.MORE_FIVE]
        self.__face_detector = MTCNN()
        super().__init__(self.NAME, property_values, self.DEFAULT_VALUE)

    def process_object(self, data_object, dataset_abs_path=None):
        """
        Checks the number of faces detected in the image and returns the corresponding range (0-1, 2-4, 5+)
        """
        try:
            object_path = os.path.join(dataset_abs_path, data_object['file_name'])
            pixels = plt.imread(object_path)

            # detect faces in the image
            faces = self.__face_detector.detect_faces(pixels)
            people = len(faces)
            if people >= 5:
                return self.MORE_FIVE
            if people >= 2:
                return self.FOUR
            return self.ONE
        except Exception:
            return self.DEFAULT_VALUE
