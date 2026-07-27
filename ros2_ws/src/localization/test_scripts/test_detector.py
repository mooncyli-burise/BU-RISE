from localization.detector import Detector
from localization.camera import Camera

model = "backbone_model/best_finetuning_model_lr1e-3.pth"

detector = Detector(model)
camera = Camera()

while True:
    frame = camera.get_frame()

    pose = detector.predict_pose(frame)

    print(pose)