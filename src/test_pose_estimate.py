import dt_apriltags as apriltag
import cv2
import utils
from viam.components.camera import Camera


image_cv = cv2.imread("data/cam.jpg")
if len(image_cv.shape) == 3:
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
else:
    gray = image_cv

detector = apriltag.Detector(families="tag16h5")
tags = detector.detect(gray)
print(tags)

rsProperties = Camera.Properties(intrinsic_parameters={
  "width_px": 640,
  "height_px": 480,
  "focal_x_px": 608.7110595703125,
  "focal_y_px": 609.39044189453125,
  "center_x_px": 320.21575927734375,
  "center_y_px": 239.54310607910156,
})

z = 500

for tag in tags:
    for corner in tag.corners:
        pt = utils.pixel_to_point(rsProperties, corner[0], corner[1], z)
        print(pt)


