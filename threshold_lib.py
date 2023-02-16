""" Saves and loads the threshold to a file."""

import json
import os.path
import numpy as np
from dataclasses import dataclass

import cv2 as cv

from icecream import ic

HUE_RANGE = 10

@dataclass
class Threshold:
    """
    Holds the lower and upper threshold color values for the objects.
    I like it this way but we could replace it with just passing static values to `process_object`.
    """

    lower: np.ndarray
    upper: np.ndarray

    def to_json(self):
        """Converts the threshold to a JSON serializable object."""
        return {"lower": self.lower.tolist(), "upper": self.upper.tolist()}

    @classmethod
    def from_json(cls, data):
        """Converts a JSON object to a threshold."""
        return cls(lower=np.array(data["lower"]), upper=np.array(data["upper"]))


def generate_starting_threshold(metadatum, path):
    """Looks at the image's center and generates a threshold around it"""
    image = cv.imread(f"test_images/{path}/{metadatum['image_name']}")
    color = image[metadatum["center_y"], metadatum["center_x"]]
    
    return Threshold(
        lower=np.array([color[0] - HUE_RANGE, 0, 0]),
        upper=np.array([color[0] + HUE_RANGE, 255, 255]),
    )

def center_color(metadatum, path):
    #I'm assuming the center_color returns an (R,G,B) tuple. I'm not sure if cv.imread(img)[y,x] returns this tuple, please lmk if I'm wrong.
    
    """Returns the RGB color value of the center of a given image (by metadatum)"""
    image = cv.imread(f"test_images/{path}/{metadatum['image_name']}")
    color = image[metadatum["center_y"], metadatum["center_x"]]
    return color
#TODO: Take the average color value of all of the pictures and return the Threshold() object.

def load_threshold(path, image_metadata=None):
    """Returns the initial threshold to test, it should return the average of all colors."""
    save_path = f"test_data/{path}/threshold.json"
    if os.path.isfile(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            threshold = Threshold.from_json(json.load(f))
    else:
        if image_metadata is None:
            raise ValueError("Threshold file does not exist.")
        else:
            """Takes the average color value over all pics"""
            ct = 0
            r,g,b = 0
            for i in image_metadata:
                r,g,b += center_color(i, path) 
                ct += 1
            r_avg = r/ct
            g_avg = g/ct
            b_avg = b/ct
            color_tuple = [r_avg,g_avg,b_avg]
            threshold = Threshold(
                lower=np.array([color[0] - HUE_RANGE, 0, 0]),
                upper=np.array([color[0] + HUE_RANGE, 255, 255]),
            )
            #please work
            
    ic(threshold)
    return threshold


def save_threshold(threshold, path):
    """Saves the threshold to a file."""
    save_path = f"test_data/{path}/threshold.json"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(threshold.to_json(), f)  
    except PermissionError as error:
        print(error)
        save_threshold(threshold, path)
