import dt_apriltags as apriltag
import cv2

image_cv = cv2.imread("data/cam.jpg")
if len(image_cv.shape) == 3:
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
else:
    gray = image_cv

detector = apriltag.Detector(families="tag16h5")
tags = detector.detect(gray)
print(len(tags))
