from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = YOLO("models/best.pt")


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        video = request.files["video"]

        if video:

            # Save uploaded video
            video_path = os.path.join(
                UPLOAD_FOLDER,
                video.filename
            )

            video.save(video_path)

            # Open video
            cap = cv2.VideoCapture(video_path)

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # Resize frame
                frame = cv2.resize(frame, (900, 500))

                # YOLO detection
                results = model(frame, conf=0.5)

                # Draw detections
                annotated_frame = results[0].plot()

                # Show live detection
                cv2.imshow(
                    "Road Damage Detection",
                    annotated_frame
                )

                # Press Q to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

            cv2.destroyAllWindows()

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)