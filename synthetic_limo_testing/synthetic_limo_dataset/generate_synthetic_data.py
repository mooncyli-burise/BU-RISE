import cv2
import numpy as np
import random
import json
from util.normalize_pixel_coords import normalize_coords
from config import WIDTH, HEIGHT

def generate_synthetic_dataset():
    transformed_images = []
    ground_truth = []

    file_path = '/home/mooncyli/BU-RISE/synthetic_limo_testing/synthetic_limo_dataset/real_limo.png'

    image = cv2.imread(file_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {file_path}")

    # Get image dimensions
    height, width = image.shape[:2]

    final_width = int(height/3*4)

    pad_hori = (final_width-width) // 2

    width = final_width

    image = cv2.copyMakeBorder(
        image, 0, 0, pad_hori, pad_hori, 
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # # set 0 degrees as up
    # center = (width // 2, height // 2)
    # angle = 90  # degrees
    # scale = 1.0

    # # Create rotation matrix
    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # # Apply rotation
    # rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # cv2.imwrite(f'synthetic_limo_dataset/original_limo.jpg', rotated_image)

    # 1. Gather all images and extract their physical creation timestamps
    for i in range(1000):
        # --- Step 1: Rotation ---
        # Define center, angle, and scale
        center = (width // 2, height // 2)
        angle = random.randint(0, 360)  # degrees
        scale = 0.25
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        
        # Apply rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        # resize to 160x120
        downscaled = cv2.resize(rotated_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

                
        # --- Step 2: Translation ---
        # Define translation offsets
        tx, ty = random.randint(-(WIDTH//2-WIDTH*0.25/2),(WIDTH//2-WIDTH*0.25/2)), random.randint(-(HEIGHT//2-HEIGHT*0.25/2),(HEIGHT//2-HEIGHT*0.25/2))  # Right and down
        
        # Create translation matrix
        translation_matrix = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])
                
        # Apply translation on rotated image
        translated_image = cv2.warpAffine(downscaled, translation_matrix, (WIDTH, HEIGHT))
        
        ground_truth.append({
            "center": normalize_coords(tx, ty, True),
            "orientation": angle
        })
        
        # Save and show result
        cv2.imwrite(f'synthetic_limo_testing/synthetic_limo_dataset/images/transformed_limo_{i}.jpg', translated_image)

    with open("synthetic_limo_testing/synthetic_limo_dataset/ground_truth.json", "w") as file:
        json.dump(ground_truth, file, indent = 4)