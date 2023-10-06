import cv2
import os
import json
import tqdm
import numpy as np 

def preprocessing(data_folder, annotation_file, new_size = 512):
    
    with open(annotation_file, 'r') as json_file:
        data = json.load(json_file)
    
    for image_filename, image_data in tqdm.tqdm(data.items()):

        image_path = os.path.join(data_folder, image_data['filename'])

        image = cv2.imread(image_path)

        image_width = image.shape[1]
        image_height = image.shape[0]
        
        mask = np.zeros_like(image)
        print(mask.shape)
        image = cv2.resize(image, (new_size, new_size))

        regions = image_data['regions']
        for region in regions:
            shape_attributes = region['shape_attributes']
            points_x = shape_attributes['all_points_x']
            points_y = shape_attributes['all_points_y']

            points = list(zip(points_x, points_y))
            points = np.array(points, dtype=np.int32)

            cv2.fillPoly(mask, [points], 255)
            mask =cv2.resize(mask, (new_size, new_size))
        result = cv2.hconcat([image, mask])
        cv2.imshow('original and mask', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == "__main__":
    preprocessing(data_folder = 'Br35H-Mask-RCNN\\TRAIN',
                  annotation_file = 'Br35H-Mask-RCNN\Annotation\\annotations_train.json')