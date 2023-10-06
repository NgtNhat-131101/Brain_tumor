import cv2
import numpy as np
import json

with open('Br35H-Mask-RCNN\\Annotation\\annotations_train.json', 'r') as json_file:
    data = json.load(json_file)

image_width = 300
image_height = 200
image_path = "Br35H-Mask-RCNN\\TRAIN\\y101.jpg"

image = cv2.imread(image_path)

segmentation_image = np.zeros_like(image, dtype=np.uint8)

regions = data['y101.jpg24870']['regions']  

for region in regions:
    shape_attributes = region['shape_attributes']
    cx = int(shape_attributes["cx"])
    cy = int(shape_attributes["cy"])
    rx = int(shape_attributes["rx"])
    ry = int(shape_attributes["ry"])
    
    color = (0, 255, 0)  
    thickness = 2
    
    ellipse_points = cv2.ellipse2Poly((cx, cy), (rx, ry), 0, 0, 360, 10)
    cv2.fillPoly(segmentation_image, [ellipse_points], color)

cv2.imshow('image', image)
cv2.imshow('Segmentation Image', segmentation_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
