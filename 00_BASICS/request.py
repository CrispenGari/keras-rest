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