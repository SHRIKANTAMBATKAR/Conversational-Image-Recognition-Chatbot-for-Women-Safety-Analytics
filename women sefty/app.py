import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Load the trained ML model
MODEL_PATH = "women_safety_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to preprocess the image
def process_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))  # Ensure the input size matches model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route to serve the chatbot UI
@app.route("/")
def index():
    return render_template("index.html")  # Ensure you have this file in templates folder

# Route to handle image upload and classification
@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        print("❌ No image found in request")
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(filepath)  # Save the uploaded image

    try:
        processed_img = process_image(image)
        prediction = model.predict(processed_img)[0][0]  # Get prediction score

        response = {"safety": "unknown", "message": "Something went wrong!", "image_url": filepath}

        if prediction > 0.7:
            response = {
                "safety": "unsafe",
                "message": "🚨 Woman is Unsafe! Immediate action required.",
                "suggestions": ["Call emergency services", "Share location with trusted contacts", "Move to a safer place"],
                "emergency_numbers": ["📞 112 (Emergency)", "📞 1091 (Women's Helpline)"],
                "confidence": round(prediction * 100, 2),
                "image_url": filepath
            }
        elif 0.4 < prediction <= 0.7:
            response = {
                "safety": "alert",
                "message": "⚠️ Woman might be in an uncomfortable situation. Stay alert.",
                "suggestions": ["Avoid isolated areas", "Stay connected with friends", "Be aware of surroundings"],
                "confidence": round(prediction * 100, 2),
                "image_url": filepath
            }
        else:
            response = {
                "safety": "safe",
                "message": "✅ Woman is Safe. No immediate danger detected.",
                "suggestions": ["Stay cautious but no threats detected", "Keep safety apps active"],
                "confidence": round((1 - prediction) * 100, 2),
                "image_url": filepath
            }

        print(f"📢 Response: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return jsonify({"error": "Failed to analyze image"}), 500

if __name__ == "__main__":
    app.run(debug=True)
