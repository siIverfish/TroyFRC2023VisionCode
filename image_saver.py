""" Saves images to a folder. """

import os.path

import cv2 as cv

from icecream import ic

class ImageSaver:
    """ Handles logic for saving files to the test_images folder. """

    # Singleton, don't worry about this if you don't know what it is, i don't either
    def __new__(cls, *_args, **_kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(ImageSaver, cls).__new__(cls)
        return cls.instance

    def __init__(self, folder):
        self.folder = folder
        # create the folder to save the images to if it doesn't exist
        if not os.path.exists(f"test_images/{self.folder}"):
            os.makedirs(f"test_images/{self.folder}")
        # make sure we don't overwrite any files
        self.file_count = 0
        while os.path.exists(self.path):
            self.file_count += 1

    def save(self, image):
        """Saves an image to the test_images folder."""
        cv.imwrite(self.path, image)
        self.file_count += 1

    @property
    def path(self):
        """Returns the path of the next file to be saved."""
        return f"test_images/{self.folder}/test_{self.file_count}.jpg"
