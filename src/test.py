import util_functions
import os
from util_functions import get_image_timestamp, calculate_car_orientations

# --- EXECUTION ---
# Replace this path with your local folder containing an unzipped EPFL car sequence
TARGET_FOLDER = '/data/epfl-gims08/tripod-seq/'

if os.path.exists(TARGET_FOLDER):
    car_angles = calculate_car_orientations(TARGET_FOLDER)
    
    # Print out a sample checklist of the results
    print(f"{'Image File':<20} | {'Time Shot':<10} | {'Orientation Angle':<15}")
    print("-" * 52)
    for item in car_angles[:15]: # Shows the first 15 images as an example
        print(f"{item['file_name']:<20} | {item['timestamp']:<10} | {item['orientation_deg']}°")
    print("\n... Script completed successfully.")
else:
    print(f"Directory '{TARGET_FOLDER}' not found. Please update the TARGET_FOLDER path variable.")
