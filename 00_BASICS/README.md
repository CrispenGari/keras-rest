### Keras Rest Basics 
We are going to cover much ground on how we can create a REST server using
flask and keras. We will be following [this keras blog post](https://blog.keras.io/category/tutorials.html).

### Environment setup.
1. Keras Tensorflow Installation

````shell
pip install tensorflow
````

2. We are also going to need the following packages

````shell
pip install flask gevent requests pillow
````

3. Checking if tensorflow is installed properly
````python
# Turning off the warnings
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)
````

We are going to have three functions which are:

1. `load_model`:
    * Used to load our trained Keras model and prepare it for inference

````python
def load_model():
    global model
    model = ResNet50(weights="imagenet")
````
2. ``prepare_image:``
This function will prepare our image by:
   * converting the image to RGB
   * Resizes it to ``224x224`` pixels (the input spatial dimensions for ResNet)
   * Preprocesses the array via mean subtraction and scaling
    
````python
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
````

Now that we have these function the function that follows is the 
3. ``predict``:
This is the endpoint where the predictions are served at `/predict`.
   
````python
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

    return make_response(jsonify(data))
````

### Important Notes
The logic of loading the model is very important here, we only want to load the model once, when the script first run, **We will never want to load the model in the predict function, this is causes unnecessary computation because the model will be loaded at every call of the ``/predict`` route**. This is the reason why I load the model as follows, before the application is started:

````python
if __name__ == '__main__':
    print("loading the model please await....")
    load_model()
    app.run(debug=True, host="localhost", port=3001)
````

### Making request.
Note that all request we are going to make will be to the following route:

````
http://localhost:3001/predict
````
This request should be a `POST` request.

1. [cURL](https://curl.haxx.se/)
We are going to use ``cuRL`` to make request to the server for example we can run the following command to get our "hello world" route using the GET method:
   
```shell
curl -X GET "http://localhost:3001/"
```
To make a prediction on the `dog.png` image we will run the following command:

````shell
curl -X POST -F image=@dog.jpg "http://localhost:3001/predict"
````

> **Important Note:** Note that when making prediction, we should turn off the debug mode by changing the `app.run()` method to:

```python
if __name__ == '__main__':
    print("loading the model please await....")
    load_model()
    app.run(host="localhost", port=3001)
```

### Consuming the Keras REST API programmatically

We are going to create a python file that will make API request to our server using the `requests` module in python.

The file will look as follows:
````python
# import the necessary packages
import requests
import json

# initialize the Keras REST API endpoint URL along with the input  image path
KERAS_REST_API_URL = "http://localhost:3001/predict"
IMAGE_PATH = "dog.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}
# submit the request
response = requests.post(KERAS_REST_API_URL,
                         files=payload).json()
print(json.dumps(response, indent=2))
````

Running the `request.py` we will get the the following resposnse
```json
{
  "predictions": [
    {
      "label": "Walker_hound",
      "probability": 0.1518935114145279
    },
    {
      "label": "English_foxhound",
      "probability": 0.10193058848381042
    },
    {
      "label": "brown_bear",
      "probability": 0.07398248463869095
    },
    {
      "label": "Doberman",
      "probability": 0.05965857952833176
    },
    {
      "label": "bluetick",
      "probability": 0.04833292216062546
    }
  ],
  "success": true
}
```

### Using Postman

1. First of all you should change the method to `POST`
2. Paste this http://localhost:3001/predict url
3. Change from `raw` to ``form-data``
4. Under Key change from `text` to file
5. Select a file under value
6. Note that the key in our case is image
7. Click send

You will get the following response:

```json
{
    "predictions": [
        {
            "label": "Walker_hound",
            "probability": 0.1518935114145279
        },
        {
            "label": "English_foxhound",
            "probability": 0.10193058848381042
        },
        {
            "label": "brown_bear",
            "probability": 0.07398248463869095
        },
        {
            "label": "Doberman",
            "probability": 0.05965857952833176
        },
        {
            "label": "bluetick",
            "probability": 0.04833292216062546
        }
    ],
    "success": true
}
```

That's all for today, Next we are going to look at the [this](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/) post and see how we can do the same task at a scalable level.
