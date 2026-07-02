from torchvision.transforms import v2 as T

def get_transform(train):
    transforms = []
    #50% chance of random horizontal flip if train
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    #converts datatype to floats and scaled to between 0.0 and 1.0
    transforms.append(T.ToDtype(torch.float, scale=True))
    #converts from tv_tensor to torch.Tensor
    transforms.append(T.ToPureTensor())
    #returns all transformations condensed into single callable sequence
    return T.Compose(transforms)


from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import os

def get_image_timestamp(image_path):
    """Extracts the exact date and time a photo was taken from EXIF data."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None
        
        # Look up EXIF codes to find the timestamp keys
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name in ["DateTimeOriginal", "DateTime"]:
                # Convert the EXIF string format 'YYYY:MM:DD HH:MM:SS' into a datetime object
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"Skipping {os.path.basename(image_path)} due to error: {e}")
    return None

def calculate_car_orientations(sequence_folder):
    """Calculates the 0 to 360 degree orientation for every image in the folder."""
    valid_images = []
    
    # 1. Gather all images and extract their physical creation timestamps
    for file_name in os.listdir(sequence_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(sequence_folder, file_name)
            timestamp = get_image_timestamp(file_path)
            if timestamp:
                valid_images.append({
                    "file_name": file_name,
                    "timestamp": timestamp
                })
                
    if not valid_images:
        print("No images with valid EXIF timestamps were found.")
        return []
        
    # 2. Sort chronologically to match the physical rotation of the turntable
    valid_images.sort(key=lambda x: x["timestamp"])
    
    total_images = len(valid_images)
    print(f"Found {total_images} images. Processing 360-degree rotation map...\n")
    
    # 3. Calculate relative angles based on time progression
    # The first chronological photo is assumed to be our baseline (0 degrees)
    results = []
    for index, img_data in enumerate(valid_images):
        # Evenly divide 360 degrees by the number of total photos captured in the sequence
        calculated_angle = (index / total_images) * 360.0
        
        # Keep the value clean to one decimal point
        calculated_angle = round(calculated_angle, 1)
        
        results.append({
            "file_name": img_data["file_name"],
            "timestamp": img_data["timestamp"].strftime("%H:%M:%S"),
            "orientation_deg": calculated_angle
        })
        
    return results

