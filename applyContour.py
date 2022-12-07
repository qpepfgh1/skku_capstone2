import cv2
import imutils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import image_convertor

original_image = image_convertor.pil_to_cv2(Image.open('./dataset/ALL/proc/image/IMG_7.PNG'))
mask_image = image_convertor.pil_to_numpy(Image.open('./dataset/ALL/proc/mask/IMG_7.PNG'))

class_num = 2
mask_list = []
mask_list.append(np.where(mask_image == 255, 0, mask_image))
mask_list.append(np.where(mask_image == 50, 0, mask_image))

for i in range(class_num):
    uniqPixel = np.unique(mask_list[i])[1]
    if uniqPixel == 255:
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    gray = cv2.cvtColor(image_convertor.numpy_to_cv2(mask_list[i]), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # Image.open("./test.jpg").convert('RGB').show()

    # find the largest contour in the threshold image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea, default=0)

    # bounding box, and display the number of points in the contour
    cv2.drawContours(original_image, [c], -1, color, 5)

cv2.imshow('',original_image)
cv2.waitKey(0)