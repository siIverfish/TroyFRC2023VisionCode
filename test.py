# import the opencv library
import cv2


# define a video capture object
vid = cv2.VideoCapture(0)

def exit_if_user_exits():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)

while True:
    _, frame = vid.read()
    cv2.imshow('frame', frame)
    exit_if_user_exits()

# After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()
