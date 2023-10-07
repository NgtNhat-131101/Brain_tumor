import cv2
import os
import json
import tqdm
import numpy as np 

def preprocessing(data_folder, annotation_file, new_size=512):
    with open(annotation_file, 'r') as json_file:
        data = json.load(json_file)
    
    for image_filename, image_data in tqdm.tqdm(data.items()):
        image_path = os.path.join(data_folder, image_data['filename'])
        image = cv2.imread(image_path)
        
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        mask = np.zeros_like(image)
        image = cv2.resize(image, (new_size, new_size))
        
        regions = image_data['regions']
        
        if regions[0]['shape_attributes']['name'] == 'polygon':
            data_path = 'valid_data'
            image_train_path = os.path.join(data_path, image_data['filename'])
            cv2.imwrite(image_train_path, image)
            
            shape_attributes = regions[0]['shape_attributes']
            points_x = shape_attributes['all_points_x']
            points_y = shape_attributes['all_points_y']

            points = list(zip(points_x, points_y))
            points = np.array(points, dtype=np.int32)

            cv2.fillPoly(mask, [points], (255, 255, 255))
            mask = cv2.resize(mask, (new_size, new_size))

            mask_filename = os.path.splitext(image_filename)[0] + '_mask.png'
            mask_path = os.path.join('mask_valid', mask_filename)
            cv2.imwrite(mask_path, mask)

if __name__ == "__main__":
    # preprocessing(data_folder='Br35H-Mask-RCNN\\TRAIN',
    #             annotation_file='Br35H-Mask-RCNN\Annotation\\annotations_train.json')
    preprocessing(data_folder = 'Br35H-Mask-RCNN\\VAL',
                annotation_file='Br35H-Mask-RCNN\\Annotation\\annotations_val.json')