from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
from datetime import datetime

app = Flask(__name__)

# folders
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# load model
model = YOLO("models/best.pt")


@app.route("/", methods=["GET", "POST"])
def index():

    output_video = None

    if request.method == "POST":

        file = request.files["video"]

        if file:

            # save uploaded video
            upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(upload_path)

            # output path
            output_filename = "detected_" + file.filename
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            # open video
            cap = cv2.VideoCapture(upload_path)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # detect
                results = model(frame)

                annotated_frame = results[0].plot()

                # date & time
                current_time = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")

                cv2.putText(
                    annotated_frame,
                    current_time,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                out.write(annotated_frame)

            cap.release()
            out.release()

            output_video = "/" + output_path.replace("\\", "/")

    return render_template(
        "index.html",
        output_video=output_video
    )


if __name__ == "__main__":
    app.run(debug=True)
