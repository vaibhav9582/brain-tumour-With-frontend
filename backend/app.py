from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# Load your model
model = load_model("braintumor.h5")

# Same labels you used in training
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

IMAGE_SIZE = 150  # same as Kaggle

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    # Convert uploaded image to OpenCV array
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    # NOTE: No normalization because you did not use img = img/255 in training
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)
    class_index = int(np.argmax(pred))
    class_name = labels[class_index]

    return jsonify({
        "class_index": class_index,
        "class_name": class_name
    })

if __name__ == "__main__":
    app.run(debug=True)
