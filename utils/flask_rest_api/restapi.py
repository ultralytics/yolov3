# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run a Flask REST API exposing one or more YOLOv3 models."""

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
    """Predicts objects in an image using YOLOv3 models exposed via Flask REST API; expects 'image' file in POST
    request.
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
            return results.pandas().xywhn[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv3 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument(
        "--model", nargs="+", default=["yolov3-tiny"], help="model(s) to run, i.e. --model yolov3-tiny yolov3"
    )
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load("ultralytics/yolov3", m, force_reload=True, skip_validation=True)

    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
