import apriltag
import cv2

image_cv = cv2.imread("./34139872896_defdb2f8d9_c.jpg")
if len(image_cv.shape) == 3:
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
else:
    gray = image_cv

detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
tags = detector.detect(gray)
