from flask import Flask, request, Response
import cv2
import numpy as np

app = Flask(__name__)

latest_frame = None


@app.route("/upload", methods=["POST"])
def upload():
    global latest_frame

    jpg = request.data

    img = cv2.imdecode(
        np.frombuffer(jpg, np.uint8),
        cv2.IMREAD_COLOR
    )

    latest_frame = img

    return "OK"


def generate():
    global latest_frame

    while True:
        if latest_frame is None:
            continue

        _, jpg = cv2.imencode(".jpg", latest_frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg.tobytes()
            + b"\r\n"
        )


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


app.run(host="0.0.0.0", port=8081)
# http://localhost:8081/video