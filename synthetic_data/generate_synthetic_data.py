import cv2
import numpy as np

i=0
filename = ''
 
# Load the image
image = cv2.imread('ex.jpg')
 
# Get image dimensions
height, width = image.shape[:2]
 
# --- Step 1: Rotation ---
# Define center, angle, and scale
center = (width // 2, height // 2)
angle = 45  # degrees
scale = 1.0
 
# Create rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
 
# Apply rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
 
# --- Step 2: Translation ---
# Define translation offsets
tx, ty = 100, 50  # Right and down
 
# Create translation matrix
translation_matrix = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])
 
# Apply translation on rotated image
translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
 
# Save and show result
cv2.imwrite(f'synthetic_data/transformed{i, filename}.jpg', translated_image)
cv2.imshow('Rotated and Translated', translated_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
