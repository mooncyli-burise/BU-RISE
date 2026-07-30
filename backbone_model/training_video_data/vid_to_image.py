import cv2
import os
import json

from localization.apriltag import AprilTag

def extract_frames(video_path, output_folder):
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    ground_truth = []

    while True:
        # Read the next frame
        success, frame = video.read()
        
        # If the frame was not successfully read, we reached the end of the video
        if not success:
            break

        gt = AprilTag.get_ground_truth(frame)
        if gt["class"]==1:
            ground_truth.append(gt)
            
            # Save the frame as a JPEG file
            frame_name = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            
            frame_count += 1

    with open("/workspace/backbone_model/training_video_data/ground_truth.json", "w") as file:
            json.dump(ground_truth, file, indent = 4)

        # Release the video capture object
    video.release()
    print(f"Successfully extracted {frame_count} frames.")