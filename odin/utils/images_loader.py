import os
import glob


class ImagesLoader:
    def __init__(self, images_path, images_extension):
        self.images_path = images_path
        self.images_extension = images_extension
        
    def get_images_array(self):
        return glob.glob(os.path.join(self.images_path, "*" + self.images_extension))
