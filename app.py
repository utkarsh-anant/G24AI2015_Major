from flask import Flask, request, render_template_string, redirect, url_for
import joblib
import numpy as np
from PIL import Image
import io
import os
import traceback

app = Flask(__name__)

MODEL_PATH = "savedmodel.pth"

# Try load model
try:
    model = joblib.load(MODEL_PATH)
    LOAD_ERR = None
except Exception as e:
    model = None
    LOAD_ERR = str(e)

HTML = """
<!doctype html>
<title>ML Model - Upload</title>
<h2>Upload face image (jpg/png) — will be resized to 64x64 grayscale</h2>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=file accept="image/*" required>
  <input type=submit value='Upload and Predict'>
</form>
{% if result is not none %}
  <h3>Predicted class: {{ result }}</h3>
{% endif %}
{% if error %}
  <pre style="color:red">{{ error }}</pre>
{% endif %}
"""

def preprocess_image(file_bytes):
    # open image, convert to grayscale and resize 64x64, same as Olivetti
    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((64,64))
    arr = np.asarray(img, dtype=np.float32) / 255.0   # normalize to 0-1
    flat = arr.flatten()
    return flat

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML, result=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML, result=None, error="Model not loaded: " + (LOAD_ERR or "unknown"))
    try:
        if 'file' not in request.files:
            return render_template_string(HTML, result=None, error="No file uploaded")
        f = request.files['file']
        b = f.read()
        x = preprocess_image(b)
        # model expects 2D array
        pred = model.predict([x])[0]
        return render_template_string(HTML, result=int(pred), error=None)
    except Exception as e:
        return render_template_string(HTML, result=None, error=traceback.format_exc())

if __name__ == "__main__":
    # bind to 0.0.0.0 for Docker/K8s
    app.run(host="0.0.0.0", port=5000)
