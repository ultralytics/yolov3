<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Flask REST API Example for YOLO Models

[Representational State Transfer (REST)](https://en.wikipedia.org/wiki/Representational_state_transfer) [Application Programming Interfaces (APIs)](https://developer.mozilla.org/en-US/docs/Web/API) are a standard way to expose [Machine Learning (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) models, allowing other services or applications to interact with them over a network. This directory provides an example REST API built using the [Flask](https://palletsprojects.com/projects/flask/) microframework to serve predictions from an [Ultralytics YOLOv3](https://docs.ultralytics.com/models/yolov3/) model, potentially loaded via [PyTorch Hub](https://pytorch.org/hub/) or other standard PyTorch methods.

Deploying models via APIs is a crucial step in [MLOps](https://www.ultralytics.com/glossary/machine-learning-operations-mlops) and enables integration into larger systems. You can explore various [model deployment options](https://docs.ultralytics.com/guides/model-deployment-options/) for different scenarios.

## 🔧 Requirements

Ensure you have the necessary Python packages installed. The primary requirement is Flask.

Install Flask using pip:

```shell
pip install Flask torch torchvision
```

_Note: `torch` and `torchvision` are required for loading and running PyTorch-based models like YOLOv3._

## ▶️ Run the API

Once Flask and dependencies are installed, you can start the API server.

Execute the Python script:

```shell
python restapi.py --port 5000
```

The API server will start listening on the specified port (default is 5000).

## 🚀 Make a Prediction Request

You can send prediction requests to the running API using tools like [`curl`](https://curl.se/) or scripting languages.

Send a POST request with an image file (`zidane.jpg` in this example) to the `/v1/object-detection/yolov3` endpoint:

```shell
curl -X POST -F image=@zidane.jpg 'http://localhost:5000/v1/object-detection/yolov3'
```

_Ensure `zidane.jpg` (or your test image) is present in the directory where you run the `curl` command._

## 📄 Understand the Response

The API processes the image and returns the [object detection](https://www.ultralytics.com/glossary/object-detection) results in [JSON](https://www.ultralytics.com/glossary/json) format. Each object detected includes its class ID, confidence score, bounding box coordinates (normalized), and class name.

Example JSON response:

```json
[
  {
    "class": 0,
    "confidence": 0.8900438547,
    "height": 0.9318675399,
    "name": "person",
    "width": 0.3264600933,
    "xcenter": 0.7438579798,
    "ycenter": 0.5207948685
  },
  {
    "class": 0,
    "confidence": 0.8440024257,
    "height": 0.7155083418,
    "name": "person",
    "width": 0.6546785235,
    "xcenter": 0.427829951,
    "ycenter": 0.6334488392
  },
  {
    "class": 27,
    "confidence": 0.3771208823,
    "height": 0.3902671337,
    "name": "tie",
    "width": 0.0696444362,
    "xcenter": 0.3675483763,
    "ycenter": 0.7991207838
  },
  {
    "class": 27,
    "confidence": 0.3527112305,
    "height": 0.1540903747,
    "name": "tie",
    "width": 0.0336618312,
    "xcenter": 0.7814827561,
    "ycenter": 0.5065554976
  }
]
```

An example Python script (`example_request.py`) demonstrating how to send requests using the popular [requests](https://requests.readthedocs.io/en/latest/) library is also included in this directory.

## 🤝 Contributing

Contributions to enhance this example or add support for other Ultralytics models are welcome! Please see the main Ultralytics [CONTRIBUTING](https://docs.ultralytics.com/help/contributing/) guide for more information on how to get involved.
