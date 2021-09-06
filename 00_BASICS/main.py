
# Turning off the warnings
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras

import PIL.Image as Image
import numpy as np
import io

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array

from flask import Flask, request, jsonify, make_response
app = Flask(__name__)
model = None

import flask
# Loading the model
def load_model():
    global model
    model = ResNet50(weights="imagenet")

def prepare_image(image, target):
    # if the image is not RGB then convert it to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the image to desired shape
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files.get("image").read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image
            image = prepare_image(image, target=(224, 224))
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            for (imageID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            data["success"] = True
    return jsonify(data)


@app.route('/', methods=["GET", "POST"])
def hello():
    return "Hello world"

if __name__ == '__main__':
    print("loading the model please await....")
    load_model()
    app.run(host="localhost", port=3001)

