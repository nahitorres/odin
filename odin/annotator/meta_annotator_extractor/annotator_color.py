import os

import PIL
from PIL import ImageChops
from odin.annotator.meta_annotator_extractor import PropertyAnnotatorInterface


class AnnotatorColor(PropertyAnnotatorInterface):
    """
    Subclass of PropertyAnnotatorInterface.
    Implement the color annotation for each input image.
    """

    COLOR = "rgb"
    GREY_SCALE = "bw"
    NAME = "color"
    DEFAULT_VALUE = ""

    def __init__(self):
        property_values = [self.COLOR, self.GREY_SCALE]
        super().__init__(self.NAME, property_values, self.DEFAULT_VALUE)

    def process_object(self, data_object, dataset_abs_path=None):
        """
        Check if the input image is colored or black-and-white.
        Return "rgb" in the first case, "bw" in the second case.
        """
        try:
            object_path = os.path.join(dataset_abs_path, data_object['file_name'])
            image = PIL.Image.open(object_path)
            if image.mode == "1" or image.mode == "L":
                return self.GREY_SCALE
            rgb = image.split()
            # ImageChops.difference() returns the absolute value of the pixel-by-pixel difference between the two images
            # To be sure that the image is truly grayscale, we need to compare colors on every pixel
            # Image.getextrema() returns a 2-tuple containing the minimum and maximum pixel value
            if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] == 0 and \
                    ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] == 0:
                return self.GREY_SCALE

            return self.COLOR
        except (OSError, KeyError, TypeError, IOError):
            return self.DEFAULT_VALUE
