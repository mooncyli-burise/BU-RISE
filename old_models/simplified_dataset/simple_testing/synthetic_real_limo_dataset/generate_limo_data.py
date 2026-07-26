import cv2
import numpy as np
import json
from config import DIMENSIONS

def generate_synthetic_data(x_classes, y_classes, angle_classes):
    ground_truth = []

    file_path = '/home/roboticslab/BU-RISE/simple_testing/synthetic_real_limo_dataset/real_limo.png'

    image = cv2.imread(file_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {file_path}")

    # Get image dimensions
    height, width = image.shape[:2]

    diff = height-width

    pad_left = (diff) // 2
    pad_right = (diff) // 2

    image = cv2.copyMakeBorder(
        image, 0, 0, pad_left, pad_right, 
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    width = height
    image_id = 0

    for i in range(1000//(x_classes*y_classes*angle_classes)+1):
        for y in range(-y_classes//2+1, y_classes//2+1):
            ty = y*height/y_classes-height/(y_classes*2)
            for x in range(-x_classes//2+1, x_classes//2+1):
                tx = x*height/x_classes-height/(x_classes*2)
                # Create translation matrix
                translation_matrix = np.float32([
                    [1, 0, tx],
                    [0, 1, ty]
                ])
                
                for angle in range(angle_classes):
                        # Define center, angle, and scale
                        center = (width // 2, height // 2)
                        theta = angle*(360/angle_classes)  # degrees
                        scale = 0.2 #random.uniform(0.1, 0.5)

                        # Create rotation matrix
                        rotation_matrix = cv2.getRotationMatrix2D(center, -theta, scale)
                        
                        # Apply rotation
                        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
                        # Apply translation on rotated image
                        translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))

                        idx = (y+1)*x_classes + x+1

                        ground_truth.append({
                            "pose": idx,
                            "orientation": theta
                        })
                        
                        # Save and show result
                        cv2.imwrite(f'simple_testing/synthetic_real_limo_dataset/images_'+DIMENSIONS+f'/transformed_limo_{image_id}.jpg', translated_image)
                        image_id += 1

    with open('simple_testing/synthetic_real_limo_dataset/ground_truth_'+DIMENSIONS+'.json', "w") as file:
        json.dump(ground_truth, file, indent=4)