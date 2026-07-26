import os
import re

def get_sorted_files(sequence_folder):
    image_files = []
    for file_name in os.listdir(sequence_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(sequence_folder, file_name))
    if not image_files:
        print("No images were found.")
        return []

    image_files = sorted(
        image_files,
        key=lambda filename: int(
            re.search(r"(\d+)(?=\.[^.]+$)", filename).group(1)
        ),
    )
    return image_files

def crop_to_ratio(image):
    height, width = image.shape[:2]
    crop_width = width * 3 // 4
    x_start = (width - crop_width) // 2   # 160
    x_end = x_start + crop_width          # 1120

    cropped = image[:, x_start:x_end]
    return cropped