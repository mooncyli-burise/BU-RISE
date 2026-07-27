import cv2
import torch

import config
from util.files import crop_to_ratio

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.cap.read()

        #convert frame to RGB and normalize pixel values to [0, 1] to match pytorch format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def apriltag_resize(self, frame):
        # reseize to 640x480
        cropped = crop_to_ratio(frame)
        resized = cv2.resize(cropped, (config.APRILTAG_WIDTH, config.APRILTAG_HEIGHT), interpolation=cv2.INTER_AREA)
        return resized

    # prep for model
    def model_prep(self, frame, device):
        # resize to 160x120
        cropped = crop_to_ratio(frame)
        resized = cv2.resize(cropped, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)

        # change format for pytorch
        image = torch.from_numpy(resized)
        image = image.permute(2,0,1)
        image = image.float() / 255.0
        image = image.to(device)
        image = image.unsqueeze(0)

        return image