import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dt_apriltags import Detector
from config import CAMERA_PARAMS, TAG_SIZE

def get_apriltag_video(image):
    detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    tags = detector.detect(image, True, CAMERA_PARAMS, TAG_SIZE)
    print(tags)

    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))


    cv2.imshow('Detected tags', color_img)
    return tags

def get_apriltag_images(sequence_folder):
    detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)
    
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
        tags = detector.detect(image, True, CAMERA_PARAMS, TAG_SIZE)
        print(tags)

        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

            cv2.putText(color_img, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255))

        all_tags.append(tags)
        cv2.imshow('Detected tags', color_img)
             if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
    return all_tags