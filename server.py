from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import os

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv8 model
try:
    model = YOLO("yolov8n.pt")  # Use yolov8s.pt, yolov8m.pt, etc., for different sizes
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")

@app.route('/')
def home():
    return "YOLOv8 Flask API is running!"

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Check if an image is included in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Load the image
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')

        # Perform inference
        print("Running inference...")
        results = model(image, conf=0.25)  # Adjust confidence threshold if needed
        print(f"Detections: {results[0].boxes}")

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        for box in results[0].boxes:
            xyxy = box.xyxy[0]  # Bounding box coordinates
            cls = int(box.cls)  # Class index
            conf = float(box.conf)  # Convert Tensor to float

            # Draw the bounding box
            draw.rectangle(xyxy.tolist(), outline="red", width=3)
            # Add label and confidence score
            label = f"{model.names[cls]}: {conf:.2f}"
            draw.text((xyxy[0], xyxy[1] - 10), label, fill="red", font=40)

        # Save the processed image to a BytesIO object
        img_io = io.BytesIO()
        image.save(img_io, format="JPEG")
        img_io.seek(0)

        # Return the image with bounding boxes
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
