""" Saves and loads the threshold to a file."""

import json
import os.path
import numpy as np
import colorsys
from dataclasses import dataclass

import cv2 as cv

from icecream import ic

HUE_RANGE = 60

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


def generate_starting_threshold(image_metadata, path):
    """
    Looks at the center of the first image in metadatum and generates a threshold around it.
    Always uses a hue range of HUE_RANGE.
    """
    count = 0
    rgb_color_list = np.array([0, 0, 0])
    for i in image_metadata:
        rgb_color_list += center_color(i, path) 
        count += 1
    averages = rgb_color_list / count
    ic(averages)
    averages /= 255
    print("After dividing by 255:")
    ic(averages)
    averages = colorsys.rgb_to_hsv(averages[0], averages[1], averages[2])
    hue = int(averages[0] * 180)
    ic(hue)
    threshold = Threshold(
        lower=np.array([max(  0, hue - HUE_RANGE), 50,     50]),
        upper=np.array([min(180, hue + HUE_RANGE), 255, 255]),
    )
    print(" -------------- Generated starting threshold: ------------------ ")
    ic(threshold)
    return threshold


def center_color(metadatum, path):
    #I'm assuming the center_color returns an (R,G,B) tuple. I'm not sure if cv.imread(img)[y,x] returns this tuple, please lmk if I'm wrong.
    # oh god the image is in BGR ill add a reverse() to the end of the function
    """Returns the RGB color value of the center of a given image (by metadatum)"""
    ic(metadatum)
    image = cv.imread(f"test_images/{path}/{metadatum['image_name']}")
    color = image[metadatum["center_y"], metadatum["center_x"]]
    color = color[::-1]
    ic(color)
    return color

#TODO: Take the average color value of all of the pictures and return the Threshold() object.

def load_threshold(path):
    """Returns the initial threshold to test, it should return the average of all colors."""
    save_path = f"test_data/{path}/threshold.json"
    if os.path.isfile(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            return Threshold.from_json(json.load(f))
    else:
        return None
            


def save_threshold(threshold, path):
    """Saves the threshold to a file."""
    save_path = f"test_data/{path}/threshold.json"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(threshold.to_json(), f)  
    except PermissionError as error:
        print(error)
        save_threshold(threshold, path)
