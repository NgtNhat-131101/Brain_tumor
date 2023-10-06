import json

# Đọc dữ liệu từ tệp JSON
with open('Br35H-Mask-RCNN\\Annotation\\annotations_train.json', 'r') as json_file:
    data = json.load(json_file)

# Kiểm tra từng item trong bộ dữ liệu
for image_name, image_data in data.items():
    print(image_name)
    print(image_data['filename'])
    break