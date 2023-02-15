""" Tests a threshold against the images in test_images. """

import os.path
import json
import argparse

import numpy as np
from icecream import ic
import cv2 as cv

from objectdetection import get_object, Threshold
from contour_lib import get_contour_center
from threshold_lib import load_threshold, save_threshold, generate_starting_threshold

# This is an argument parser that gets the --path argument from the command line and saves it to args.path
parser = argparse.ArgumentParser("Tests a threshold against the images in test_images.")
parser.add_argument("--path", type=str, help="The path to the test data file.")
args = parser.parse_args()

# The path to the saved threshold and the path to the labeled data
save_path = f"test_data/{args.path}/threshold.json"
data_path = f"test_data/{args.path}/data.json"

DID_NOT_FIND_IMAGE_PENALTY = 40

def make_random_change(threshold):
    """
    Makes a random change to the threshold for testing.
    """
    return Threshold(
        lower=threshold.lower + np.random.randint(-10, 10, 3),
        upper=threshold.upper + np.random.randint(-10, 10, 3),
    )


def generate_test_thresholds(base_threshold):
    """
    Generates a bunch of random thresholds to test.
    These will be used to find the best threshold by incrementally making random changes 
    to the threshold and seeing if they are more effective.
    """
    for _ in range(5):
        threshold = make_random_change(base_threshold)
        # If the lower bound is greater than the upper bound, 
        # then the threshold is invalid and we should skip it.
        if not all(threshold.lower < threshold.upper):
            continue
        yield threshold

# load the labeled data
with open(data_path, "r", encoding="utf-8") as f:
    image_click_data = json.load(f)

# loads all of the images mentioned in the data
images = []
for data in image_click_data:
    image_name = data["image_name"]
    images.append(cv.imread(f"test_images/{args.path}/{image_name}"))


def rate_threshold(threshold):
    """
    Returns how many pixels off the threshold gets on average.
    This is used to test randomly generated thresholds for fitness.
    
    NOTE: The lower the score, the better the threshold. So a score of 0 is perfect and a score of 1,000 is terrible.
    A good threshold has a score of around 40.
    """
    amounts_off = []
    not_found = 0

    for datum, image in zip(image_click_data, images):
        largest_object = get_object(image, threshold)
        if largest_object is None:
            not_found += 1
            continue
        center = get_contour_center(largest_object)
        amounts_off.append(abs(center[0] - datum["center_x"]) + abs(center[1] - datum["center_y"]))
        
    # return the median of the amounts off
    score = np.median(amounts_off)
    # apply a penalty for not finding the object in an image.
    score += (not_found * DID_NOT_FIND_IMAGE_PENALTY) / len(images)
    # if the score is nan, then return infinity instead because this is a really really bad threshold.
    # This happens when the threshold never finds the object.
    if np.isnan(score):
        return float("inf")
    return score



def main():
    """
    Continously generates random thresholds and tests them against the labeled data.
    The best threshold is saved to a file so that it can be used in objectdetection.py.
    """
    # loads in the previous best threshold
    best = load_threshold(args.path)
    if best is None:
        # if there is no previous best threshold, then generate a rough starting threshold
        best = generate_starting_threshold(image_click_data, args.path)
        # Save the threshold so that we don't have to generate it again.
        save_threshold(best, args.path)
    # TODO: we could save the best threshold's score so that we don't have to rate it again.
    best_score = rate_threshold(best)
    if best_score is np.nan:
        best_score = float("inf")
    while True:
        # generate 5 random thresholds and see which one is the best
        for threshold in generate_test_thresholds(best):
            score = rate_threshold(threshold)
            # the lower the score, the better the threshold. 
            # I use <= instead of < because I want to allow small variations.
            if score <= best_score:
                # set the new best threshold
                best_score = score
                ic("NEW BEST:", threshold, score)
                # we don't have to save the threshold in a variable because it is saved in the file.
                save_threshold(best, args.path)
            ic(score)


if __name__ == "__main__":
    main()
