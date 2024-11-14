from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained models
violence_model = load_model("E:\Project_model\H5_format\vgg16_violence_detection.h5")
# Replace this with your code to load the YOLO model
yolo_model = YOLO("E:\Project_model\H5_format\vgg16_violence_detection.h5")

# Function to integrate violence prediction and YOLO object detection
def detect_objects_and_predict_violence(input_path, output_path, yolo_model, confidence_threshold=0.60):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error opening video file")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Violence prediction
            resized_frame = cv2.resize(frame, (224, 224))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)
            prediction = violence_model.predict(input_frame)

            if prediction[0][0] > 0.5:
                violence_label = "Violence Detected"
                violence_color = (0, 0, 255)  # Red for detection
            else:
                violence_label = "No Violence Detected"
                violence_color = (0, 255, 0)  # Green for no detection

            # Draw label on the frame
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 32)
            draw.text((50, 50), violence_label, font=font, fill=violence_color)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # YOLO object detection logic
            yolo_results = yolo_model(frame)
            for result in yolo_results:
                classes = result.names
                cls = result.boxes.cls
                conf = result.boxes.conf
                detections = result.boxes.xyxy

                for pos, detection in enumerate(detections):
                    if conf[pos] >= confidence_threshold:
                        xmin, ymin, xmax, ymax = detection
                        label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                        color = (0, int(cls[pos] * 10), 255)

                        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                        cv2.putText(frame, label, (int(xmin), int(ymin) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            out.write(frame)

        cap.release()
        out.release()
        print(f"Output video saved as {output_path}")

    except Exception as e:
        print(f"Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(input_path)

    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")
    detect_objects_and_predict_violence(input_path, output_path, yolo_model)

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
