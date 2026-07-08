import torch
import cv2
from model.model import create_model
from train import device

#load trained model
eval_model = create_model()
eval_model.load_state_dict(torch.load("best_robot_detector.pth", map_location=device))
eval_model.to(device)
eval_model.eval()

images = []

while True:
    #init cam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    #convert frame to RGB and normalize pixel values to [0, 1] to match pytorch format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = torch.from_numpy(frame_rgb)
    image = image.permute(2,0,1)
    image = image.float() / 255.0
    image = image.to(device)
    images = [image]

    #run inference
    with torch.no_grad():
        outputs = eval_model(images)

    output = outputs[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    angles = output["orientation"].cpu().numpy()

    for box, score, angle in zip(boxes, scores, angles):

        if score < 0.5:
            continue

        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2)

        cv2.putText(frame,
                    f"{angle} deg",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)
        
    cv2.imshow("Robot Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()