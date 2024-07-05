# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Run a Flask REST API exposing one or more YOLOv5s models."""

import argparse
import io

import torch
from flask import Flask, request
from PIL import Image

app = Flask(__name__)
models = {}

DETECTION_URL = "/v1/object-detection/<model>"


@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    """
    Predicts objects in an image using YOLOv5s models exposed via a Flask REST API; expects an 'image' file in the POST
    request.

    Args:
      model (str): The name of the YOLOv5s model to be used for prediction, specified in the URL.

    Returns:
      flask.Response: JSON response containing detection results, with bounding box coordinates, confidence scores,
      and class labels.

    Notes:
      - Ensure the model specified in the URL is loaded and registered in the `models` dictionary before making requests.
      - The POST request must include an image file under the key 'image' for successful processing.

    Example:
      ```python
      curl -X POST http://127.0.0.1:5000/v1/object-detection/yolov5s -F image=@path/to/your/image.jpg
      ```
    """
    if request.method != "POST":
        return

    if request.files.get("image"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=640)  # reduce size=320 for faster inference
            return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv3 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--model", nargs="+", default=["yolov5s"], help="model(s) to run, i.e. --model yolov5n yolov5s")
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load("ultralytics/yolov5", m, force_reload=True, skip_validation=True)

    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
