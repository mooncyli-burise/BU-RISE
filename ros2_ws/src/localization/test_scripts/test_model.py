from localization.camera import Camera
from localization.model import Model

weights = "backbone_model/best_finetuning_model_lr1e-3.pth"

camera = Camera()
model = Model(weights)

while True:
    frame = camera.get_frame()

    center, heading = model.predict(frame)

    print(center)
    print(heading)