from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load trained YOLO model
model = YOLO("models/best.pt")


@app.route("/", methods=["GET", "POST"])
def index():

    output_video = None

    if request.method == "POST":

        # Get uploaded video
        video = request.files["video"]

        if video:

            # Save uploaded video
            input_path = os.path.join(
                UPLOAD_FOLDER,
                video.filename
            )

            video.save(input_path)

            # Output filename
            output_filename = "detected_" + video.filename

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_filename
            )

            # Open video
            cap = cv2.VideoCapture(input_path)

            # Video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Save processed video
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # Resize for faster processing
                frame = cv2.resize(frame, (640, 360))

                # YOLO detection
                results = model(frame, conf=0.5)

                # Draw detections
                annotated_frame = results[0].plot()

                # Resize back
                annotated_frame = cv2.resize(
                    annotated_frame,
                    (width, height)
                )

                # Write frame
                out.write(annotated_frame)

            cap.release()
            out.release()

            # Send output video to HTML
            output_video = "output/" + output_filename

    return render_template(
        "index.html",
        output_video=output_video
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
