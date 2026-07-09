import torch
import cv2
from april_tags.get_data import get_apriltag_video, get_apriltag_images
from april_tags.create_ground_truth import create_ground_truth
from util.objects import cap

def test_video_detection():
    #init cam
    if not cap.isOpened():
        print("Failed to open camera")
        exit()

    while True:
        ret, frame = cap.read()

        # #convert frame to RGB and normalize pixel values to [0, 1] to match pytorch format
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # image = torch.from_numpy(frame_rgb)
        # image = image.permute(2,0,1)
        # image = image.float() / 255.0
        # # images = [image]

        tags = get_apriltag_video(frame)
        if tags:
            print("Translation:")
            print(tags[0].pose_t)

            print("Rotation:")
            print(tags[0].pose_R)
        else:
            print("No tags detected")

        #cv2.imshow("Testing", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

def test_image_detection(folder_path):
    all_tags = get_apriltag_images(folder_path)
    print(len(all_tags))
    print("tee es pose: ", all_tags[0][0].pose_t, "\n")
    print("rut ruh:", all_tags[0][0].pose_R, "\n")
    print(create_ground_truth(all_tags))