from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = YOLO("models/best.pt")


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["file"]

        if file:

            file_path = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(file_path)

            # FILE EXTENSION
            ext = file.filename.split(".")[-1].lower()

            # IMAGE DETECTION
            if ext in ["jpg", "jpeg", "png"]:

                image = cv2.imread(file_path)

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

                cv2.imshow(
                    "Road Damage Detection AI",
                    annotated_image
                )

                cv2.waitKey(0)

                cv2.destroyAllWindows()

            # VIDEO DETECTION
            else:

                cap = cv2.VideoCapture(file_path)

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame = cv2.resize(
                        frame,
                        (1000, 600)
                    )

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

                    cv2.imshow(
                        "Road Damage Detection AI",
                        annotated_frame
                    )

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()

                cv2.destroyAllWindows()

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
