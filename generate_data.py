""" 
Opens a camera stream and allows you to save images to a folder by pressing 's'.

Saves the images to test_images/<path>/test_<number>.jpg
This is the simplest program in the project.
"""

import cv2 as cv
import argparse

from image_saver import ImageSaver

parser = argparse.ArgumentParser("Shows the camera stream and detects objects based on a threshold value.")
parser.add_argument("-p", "--path", type=str, help="The path to save the images to.")
args = parser.parse_args()

if args.path is None:
    raise ValueError("Please specify a path to save the images to.")

# modified version of infinite_frame_stream from objectdetection.py that returns the key pressed and the frame.
def infinite_frame_stream():
    """
    Returns a generator that yields frames from the webcam.
    Exits the program if the user presses 'q' and waits between frames.
    """
    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        if frame is None:
            print("Error reading frame")
            exit(1)
        # Exit if the user presses 'q'
        key = cv.waitKey(1)
        if key & 0xFF == ord("q"):
            exit(0)
        yield key, frame

def main():
    """ Opens a camera stream and allows you to save images to a folder. """
    for key, frame in infinite_frame_stream():
        cv.imshow("frame", frame)
        if key & 0xFF == ord("s"):
            print("Saving image...")
            # image_saver is a singleton, so it will only be created once.
            # it handles some logic for not overwriting previously saved images.
            ImageSaver(args.path).save(frame)

if __name__ == "__main__":
    main()
