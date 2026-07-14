# import cv2
# import numpy as np
# import random
# import json
# from util.normalize_pixel_coords import normalize_coords

# def generate_synthetic_data():
#     ground_truth = []

#     file_path = '/home/roboticslab/BU-RISE/simple_testing/synthetic_real_limo_dataset/real_limo.png'

#     image = cv2.imread(file_path)

#     if image is None:
#         raise FileNotFoundError(f"Could not load image: {file_path}")

#     # Get image dimensions
#     height, width = image.shape[:2]

#     diff = height-width

#     pad_left = (diff) // 2
#     pad_right = (diff) // 2

#     image = cv2.copyMakeBorder(
#         image, 0, 0, pad_left, pad_right, 
#         borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
#     )

#     width = height

#     for i in range(1000):
#         # --- Step 1: Rotation ---
#         # Define center, angle, and scale
#         center = (width // 2, height // 2)
#         angle = random.randint(0, 3)*90  # degrees
#         scale = 0.2 #random.uniform(0.1, 0.5)

#         # Create rotation matrix
#         rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        
#         # Apply rotation
#         rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
                
#         # --- Step 2: Translation ---
#         # Define translation offsets
#         x, y = random.randint(-1,2), random.randint(-1,2)
#         tx, ty = x*height/4-height/8, y*height/4-height/8  # Right and down
        
#         # Create translation matrix
#         translation_matrix = np.float32([
#             [1, 0, tx],
#             [0, 1, ty]
#         ])
                
#         # Apply translation on rotated image
#         translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
        
#         idx = (y+1)*4 + x+1

#         ground_truth.append({
#             "pose": idx,
#             "orientation": angle
#         })
        
#         # Save and show result
#         cv2.imwrite(f'simple_testing/synthetic_real_limo_dataset/images/transformed_limo_{i}.jpg', translated_image)

#     with open("simple_testing/synthetic_real_limo_dataset/ground_truth.json", "w") as file:
#         json.dump(ground_truth, file)


import cv2
import numpy as np
import json
from util.normalize_pixel_coords import normalize_coords

def generate_synthetic_data():
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

    for i in range(1000//64):
        for y in range(-1,3):
            ty = y*height/4-height/8
            for x in range(-1,3):
                tx = x*height/4-height/8
                # Create translation matrix
                translation_matrix = np.float32([
                    [1, 0, tx],
                    [0, 1, ty]
                ])
                
                for angle in range(4):
                        # Define center, angle, and scale
                        center = (width // 2, height // 2)
                        theta = angle*90  # degrees
                        scale = 0.2 #random.uniform(0.1, 0.5)

                        # Create rotation matrix
                        rotation_matrix = cv2.getRotationMatrix2D(center, -theta, scale)
                        
                        # Apply rotation
                        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
                        # Apply translation on rotated image
                        translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))

                        idx = (y+1)*4 + x+1

                        ground_truth.append({
                            "pose": idx,
                            "orientation": theta
                        })
                        
                        # Save and show result
                        cv2.imwrite(f'simple_testing/synthetic_real_limo_dataset/images/transformed_limo_{image_id}.jpg', translated_image)
                        image_id += 1

    with open("simple_testing/synthetic_real_limo_dataset/ground_truth.json", "w") as file:
        json.dump(ground_truth, file, indent=4)