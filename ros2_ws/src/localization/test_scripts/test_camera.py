import cv2
from localization.camera import Camera

camera = Camera()

while True:
    frame = camera.get_frame()

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break