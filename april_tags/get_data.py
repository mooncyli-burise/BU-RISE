import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pupil_apriltags import Detector
from config import CAMERA_PARAMS, TAG_SIZE, APRILTAG_HEIGHT, APRILTAG_WIDTH
import math

detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

def get_apriltag_video(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tags = detector.detect(image, True, CAMERA_PARAMS, TAG_SIZE)
    # print(tags)

    # show_video_tags(tags, image)

    return tags

def get_apriltag_images(sequence_folder, tag_size = TAG_SIZE):
    image_files = []

    for file_name in os.listdir(sequence_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(sequence_folder, file_name))
    if not image_files:
        print("No images with valid EXIF timestamps were found.")
        return []

    all_tags = []

    for file in image_files:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        downscaled = cv2.resize(image, (APRILTAG_WIDTH, APRILTAG_HEIGHT), interpolation=cv2.INTER_AREA)
        tags = detector.detect(downscaled, True, CAMERA_PARAMS, tag_size)

        for i, tag in enumerate(tags):
            print(f"Tag {i} detected in {file}")

        # print(tags)

        show_image_tags(tags, downscaled)

        all_tags.append(tags)
 
    # When everything done, release the capture
    cv2.destroyAllWindows()
    return all_tags

def show_image_tags(tags, image):
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))
        
        rotation_matrix = tag.pose_R
        orientation = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / math.pi
        cx, cy = tag.center.astype(int)
        
        cv2.putText(color_img,
                f"{orientation} deg",
                (cx, cy+10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)
    
        cv2.circle(color_img, (cx, cy), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.putText(color_img,
                    f"({cx}, {cy}",
                    (cx, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)
        cv2.imshow('Detected tags', color_img)
        while True:
            if cv2.waitKey(1) == ord('q'):
                break

def show_video_tags(tags, image):
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))
        
        center = tag.pose_t[:2]
        rotation_matrix = tag.pose_R
        orientation = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / math.pi
        cx, cy = tag.center.astype(int)
        
        cv2.putText(color_img,
                f"{orientation} deg",
                (cx, cy+10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)
    
        cv2.circle(color_img, (cx, cy), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.putText(color_img,
                    f"({cx}, {cy}",
                    (cx, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)
    cv2.imshow('Detected tags', color_img)
    return color_img
