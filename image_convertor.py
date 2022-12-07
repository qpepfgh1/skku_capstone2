import cv2
import numpy as np

def pil_to_cv2(pil_image):
    # numpy Image
    numpy_image = np.array(pil_image)

    # cv2 Image
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return cv2_image

def pil_to_numpy(pil_image):
    # numpy Image
    numpy_image = np.array(pil_image)
    return numpy_image

def numpy_to_cv2(numpy_image):
    # cv2 Image
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return cv2_image