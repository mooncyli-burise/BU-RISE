import cv2
import numpy as np
import os
import random
import json
from util.normalize_pixel_coords import normalize_coords

#move outside of folder to run

transformed_images = []
ground_truth = []

file_path = '/home/roboticslab/BU-RISE/synthetic_limo_dataset/original_limo.jpg'

image = cv2.imread(file_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {file_path}")

# Get image dimensions
height, width = image.shape[:2]

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
for i in range(200):
    # --- Step 1: Rotation ---
    # Define center, angle, and scale
    center = (width // 2, height // 2)
    angle = random.randint(0, 360)  # degrees
    scale = random.uniform(0.25, 1)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
    # --- Step 2: Translation ---
    # Define translation offsets
    tx, ty = random.randint(-120,120), random.randint(-65,65)  # Right and down
    
    # Create translation matrix
    translation_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])
            
    # Apply translation on rotated image
    translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
    
    ground_truth.append({
        "center": normalize_coords(tx, ty),
        "orientation": angle
    })
    
    # Save and show result
    cv2.imwrite(f'synthetic_limo_dataset/images/transformed_limo_{i}.jpg', translated_image)

with open("synthetic_limo_dataset/ground_truth.json", "w") as file:
    json.dump(ground_truth, file)