# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Send a sample image to the Flask REST API and print the returned YOLOv3 detections."""

import pprint

import requests

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov3-tiny"
IMAGE = "zidane.jpg"

# Read image
with open(IMAGE, "rb") as f:
    image_data = f.read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)
