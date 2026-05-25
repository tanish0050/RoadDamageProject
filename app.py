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

# load YOLO model
model = YOLO("models/best.pt")


@app.route("/", methods=["GET", "POST"])
def index():

    output_video = None
    output_image = None

    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html")

        # save uploaded file
        upload_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(upload_path)

        # extension
        ext = file.filename.split(".")[-1].lower()

        # ---------------- IMAGE DETECTION ----------------

        if ext in ["jpg", "jpeg", "png"]:

            image = cv2.imread(upload_path)

            results = model(
                image,
                conf=0.5
            )

            annotated_image = results[0].plot()

            current_time = datetime.now().strftime(
                "%d-%m-%Y  %I:%M:%S %p"
            )

            cv2.putText(
                annotated_image,
                current_time,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            output_image_name = "detected_image.jpg"

            output_image_path = os.path.join(
                OUTPUT_FOLDER,
                output_image_name
            )

            cv2.imwrite(
                output_image_path,
                annotated_image
            )

            output_image = "/" + output_image_path.replace("\\", "/")

        # ---------------- VIDEO DETECTION ----------------

        else:

            output_video_name = "detected_output.mp4"

            output_video_path = os.path.join(
                OUTPUT_FOLDER,
                output_video_name
            )

            cap = cv2.VideoCapture(upload_path)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0:
                fps = 20

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (width, height)
            )

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model(
                    frame,
                    conf=0.5
                )

                annotated_frame = results[0].plot()

                current_time = datetime.now().strftime(
                    "%d-%m-%Y  %I:%M:%S %p"
                )

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

            output_video = "/" + output_video_path.replace("\\", "/")

    return render_template(
        "index.html",
        output_video=output_video,
        output_image=output_image
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
