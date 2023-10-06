import cv2
import numpy as np
import json

# Đọc dữ liệu từ tệp JSON
with open('Br35H-Mask-RCNN\\Annotation\\annotations_train.json', 'r') as json_file:
    data = json.load(json_file)

# Kích thước ảnh (điều này phải phù hợp với kích thước ảnh thực tế)
image_width = 300
image_height = 200
image_path = "Br35H-Mask-RCNN\\TRAIN\\y101.jpg"

image = cv2.imread(image_path)

# Tạo ảnh segmentation trắng
segmentation_image = np.zeros_like(image, dtype=np.uint8)

# Trích xuất thông tin về các điểm dữ liệu từ dữ liệu JSON
regions = data['y101.jpg24870']['regions']  # Chọn tệp cụ thể theo tên tệp

for region in regions:
    shape_attributes = region['shape_attributes']
    cx = int(shape_attributes["cx"])
    cy = int(shape_attributes["cy"])
    rx = int(shape_attributes["rx"])
    ry = int(shape_attributes["ry"])
    
    # Màu và độ dày của đường vẽ
    color = (0, 255, 0)  # Màu xanh lá cây
    thickness = 2
    
    # Vẽ ellipse
    ellipse_points = cv2.ellipse2Poly((cx, cy), (rx, ry), 0, 0, 360, 10)
    cv2.fillPoly(segmentation_image, [ellipse_points], color)

# Hiển thị ảnh segmentation
cv2.imshow('image', image)
cv2.imshow('Segmentation Image', segmentation_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu ảnh segmentation thành một tệp PNG nếu cần
# cv2.imwrite('segmentation_result.png', segmentation_image)
