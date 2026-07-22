import cv2
import numpy as np
import random
import json
from util.normalize_pixel_coords import normalize_coords
from config import WIDTH, HEIGHT

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Adds Gaussian noise to an image.
    mean: Average change in pixel value (usually 0).
    sigma: Standard deviation (higher means more noise).
    """
    # Generate random Gaussian noise with the same shape as the image
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    
    # Add the noise to the original image
    noisy_image = image.astype(np.float32) + noise
    
    # Clip the pixel values to keep them in the valid [0, 255] range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def add_salt_and_pepper_noise(image, amount=0.04):
    """
    Adds Salt and Pepper noise to an image.
    amount: Percentage of pixels to affect (e.g., 0.04 = 4%).
    """
    noisy_image = image.copy()
    
    # Add Salt (White pixels)
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # Add Pepper (Black pixels)
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

def apply_random_augmentation(image):
    choice = random.choice(['gaussian', 'salt_pepper', 'both', 'clean'])
    
    if choice == 'gaussian':
        return add_gaussian_noise(image, sigma=random.randint(10, 40))
    elif choice == 'salt_pepper':
        return add_salt_and_pepper_noise(image, amount=random.uniform(0.01, 0.05))
    elif choice == 'both':
        # Apply Gaussian first, then Salt & Pepper on top
        img_gau = add_gaussian_noise(image, sigma=random.randint(10, 25))
        return add_salt_and_pepper_noise(img_gau, amount=random.uniform(0.01, 0.03))
    else:
        return image # Keep it clean

def generate_synthetic_dataset(size):
    ground_truth = []

    file_path = 'backbone_model/synthetic_limo_dataset/real_limo.png'

    # 1. Gather all images and extract their physical creation timestamps
    for i in range(size):
        # random color background (change to (0,0,0) for black)
        h = random.randint(0,179)
        s = random.randint(0,255//2)
        v = random.randint(0,255)

        hsv_pixel = np.uint8([[[h, s, v]]])

        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)

        b, g, r = map(int, bgr_pixel[0][0])
        color = (b, g, r)

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
            borderType=cv2.BORDER_CONSTANT, value=color
        )

        # --- Step 1: Rotation ---
        # Define center, angle, and scale
        center = (width // 2, height // 2)
        angle = random.randint(0, 359)  # degrees
        scale = random.uniform(0.20, 0.30)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        
        # Apply rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=color)

        # resize to 160x120
        downscaled = cv2.resize(rotated_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        robot_width = width*scale/final_width*WIDTH 
        robot_height = height*scale/height*HEIGHT
                
        # --- Step 2: Translation ---
        # Define translation offsets
        # half of frame size - half of robot size
        tx, ty = random.randint(-(WIDTH//2-robot_width//2),(WIDTH//2-robot_width//2)), random.randint(-(HEIGHT//2-robot_height//2),(HEIGHT//2-robot_height//2))  # Right and down
        
        # Create translation matrix
        translation_matrix = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])
                
        # Apply translation on rotated image
        translated_image = cv2.warpAffine(downscaled, translation_matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_CONSTANT, borderValue=color)
        
        noisy_image = apply_random_augmentation(translated_image)

        ground_truth.append({
            "center": normalize_coords(tx, ty, True),
            "orientation": angle
        })
        
        # Save and show result
        cv2.imwrite(f'backbone_model/synthetic_limo_dataset/images/transformed_limo_{i}.jpg', noisy_image)

    with open("backbone_model/synthetic_limo_dataset/ground_truth.json", "w") as file:
        json.dump(ground_truth, file, indent = 4)