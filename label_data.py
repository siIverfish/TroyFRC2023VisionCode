""" 
Loads data from test_images and gets a person to label it. 

Use the --path argument to specify which folder to label.
e.g. python label_data.py --path cone

First, it will load the data from test_data/<path>/data.json, and then it will load the images from test_images/<path>.
It shows the images one at a time, and the user can double-click on the center of the object in the image and then press `q` to move on to the next image.
"""

import os
import json
import argparse

import cv2 as cv

from icecream import ic

def get_number_from_string(string: str) -> int:
    """ Returns the number in a string. e.g. "test_123.jpg" -> 123 """
    return int("".join(c for c in string if c.isdigit()))

# Argument parser that takes a folder name. 
# args.path will be the folder name passed to the script
parser = argparse.ArgumentParser("Labels the data in the folder inside test_images given by --path")
parser.add_argument("--path", type=str, help="The folder inside test_images to label")
args = parser.parse_args()


images_dir_path = "test_images/" + args.path
data_path   = "test_data/" + args.path + "/data.json"

# Create the folder if it doesn't exist
if not os.path.exists("test_data/" + args.path):
    os.mkdir("test_data/" + args.path)

# Get the images and data, and sort them by the number in the filename
images = sorted(os.listdir(images_dir_path), key=get_number_from_string)

# get only the numbers from previous_data

try:
    with open(data_path, "r", encoding="utf-8") as f:
        # Loads the data from the json file previously saved
        user_center_data = json.load(f)
except FileNotFoundError:
    # if there is no saved data for this folder, just use an empty list, which will be saved later
    user_center_data = []

# Get all of the numbers of previously loaded images
already_labeled_numbers = [get_number_from_string(x["image_name"]) for x in user_center_data]


def save_click(event, x, y, flags, param):
    """ Saves the data when the user clicks. """
    if event == cv.EVENT_LBUTTONDBLCLK:
        ic(x, y, image)
        # Add the user's click to the data
        user_center_data.append({"center_x": x, "center_y": y, "image_name":image})
        with open(data_path, "w", encoding="utf-8") as f:
            # Save the data again.
            f.write(json.dumps(user_center_data, indent=4))


for image in images:
    # Skip the image if it has already been labeled
    if get_number_from_string(image) in already_labeled_numbers:
        continue
    ic(image)
    img = cv.imread(f"{images_dir_path}/{image}")
    cv.imshow("image", img)
    cv.setMouseCallback("image", save_click)
    cv.waitKey(0)
    cv.destroyAllWindows()