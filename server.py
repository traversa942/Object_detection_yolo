from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import base64

app = Flask(__name__)

# Load YOLOv8 model
try:
    model = YOLO("yolov8n.pt")  # Update the model file if using a different version
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")

@app.route('/')
def home():
    return "YOLOv8 Flask API is running!"

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Check if JSON body exists
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "No image data provided in JSON"}), 400

        # Decode Base64 image from JSON
        image_data = request.json['image']
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {e}"}), 400

        # Perform inference
        results = model(image, conf=0.25)

        # Load a font
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font_size = 20
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            cls = int(box.cls)
            conf = float(box.conf)

            # Draw the bounding box
            draw.rectangle(xyxy, outline="red", width=3)
            label = f"{model.names[cls]}: {conf:.2f}"
            draw.text((xyxy[0], xyxy[1] - 20), label, fill="red", font=font)

        # Save the processed image to a BytesIO object
        img_io = io.BytesIO()
        image.save(img_io, format="JPEG")
        img_io.seek(0)

        # Encode the processed image to Base64
        processed_image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # Return the Base64-encoded processed image
        return jsonify({"processed_image": processed_image_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
