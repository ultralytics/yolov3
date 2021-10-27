import torch
from IPython.display import Image, clear_output  # to display images

import os
from pathlib import Path
import numpy as np
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier
from models import *
from utils.datasets import *
import json
import csv
import datetime

import os

ROOT_FOLDER = os.path.join("Assignment3")

DATA_FOLDER = os.path.join(ROOT_FOLDER, "Data")

FRAMES_DIR = os.path.join(DATA_FOLDER, "Frames")
# IMAGE_MODEL_INFERENCE_DIR = os.path.join(DATA_FOLDER, "Image_model_inference")

VIDEO_DIR = os.path.join(DATA_FOLDER, "Videos")
video_name = os.listdir(VIDEO_DIR)[0]

VIDEO_TO_INFER_DIR = os.path.join(VIDEO_DIR, video_name)

VIDEO__MODEL_INFERENCE_DIR = os.path.join(DATA_FOLDER, "Video_model_inference")

FRAMES_TO_INFER_JSON = os.path.join(ROOT_FOLDER, "JSON", "frames_to_infer.json")

SAVE_RESULTS_DIR = os.path.join(ROOT_FOLDER, "Results")

SAVE_VIDEO_DIR = os.path.join(SAVE_RESULTS_DIR, video_name)
CSV_DIR = os.path.join(SAVE_RESULTS_DIR, "results.csv")

# Instruction: insert the path of the weight to one of trained model from above
# WEIGHTS = os.path.join(RESULTS_FOLDER, "exp", "weights", "best.pt")
WEIGHTS = os.path.join(ROOT_FOLDER, "Weights", "best.pt")

# Device to use (e.g. "0", "1", "2"... or "cpu")
DEVICE = "0"

# Intended image size must be in multiples of 32
# Image will be resized for training
IMAGE_SIZE = 640


class ImageDetection:
    def __init__(self, objectType, bbox):
        self.objectType = objectType
        self.bbox = bbox


def numKayakVesselAndBbox(listOfAllObjectsDetected):
    resultList = list()

    numKayak = 0
    numVessel = 0

    vesselBboxString = ""
    kayakBboxString = ""

    for objectDetected in listOfAllObjectsDetected:
        objectType = objectDetected.objectType
        objectBbox = objectDetected.bbox

        if objectType == 'vessel':
            numVessel += 1
            vesselBbox = [str(round(bboxCoord, 4)) for bboxCoord in objectBbox]
            vesselBboxString += f'{"_".join(vesselBbox)};'

        else:
            numKayak += 1
            kayakBbox = [str(round(bboxCoord, 4)) for bboxCoord in objectBbox]
            kayakBboxString += f'{"_".join(kayakBbox)};'

    resultList.append(numVessel)
    resultList.append(numKayak)
    resultList.append(vesselBboxString)
    resultList.append(kayakBboxString)

    return resultList


def saveResultsToCsv(dataToSaveToCsv):
    header = ['frame_index', 'no_ships', 'no_kayaks', 'ships_coordinates', 'kayaks_coordinates']

    with open(CSV_DIR, 'w', encoding='UTF8', newline='') as csvFile:
        writer = csv.writer(csvFile)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(dataToSaveToCsv)

    print(f'Data saved to {CSV_DIR}')


def getFramesToInfer():
    jsonFile = open(FRAMES_TO_INFER_JSON, )
    jsonFileLoaded = json.load(jsonFile)
    framesToInfer = jsonFileLoaded["frames_to_infer"]
    return framesToInfer


def modelInference(framesToInfer):
    print('Starting object detection on video')
    startTime = datetime.datetime.now()

    # Set device using in os.environ['CUDA_VISIBLE_DEVICES']
    device = select_device(DEVICE)

    # Loads weight of model
    model = attempt_load(WEIGHTS, map_location=device)

    # Gets stride size of model
    stride = int(model.stride.max())

    # Makes sure image size is a multiple of stride size
    imgsz = check_img_size(IMAGE_SIZE, s=stride)

    # Get class names
    names = model.module.names if hasattr(model, 'module') else model.names

    cap = cv2.VideoCapture(VIDEO_TO_INFER_DIR)
    _, img0 = cap.read()

    save_path = os.path.join(SAVE_VIDEO_DIR, os.path.split(VIDEO_TO_INFER_DIR)[-1])

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))

    print(f'fps {fps}')
    print(f'w x h = {w} x {h}')

    # the list to store (frameNumber, imageDetection)
    allDetectionList = list()

    while img0 is not None:

        # Padded resize
        img = letterbox(img0, new_shape=IMAGE_SIZE)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred)

        vessel_number = 1
        kayak_number = 1

        currentFrameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)  # returns a float
        currentFrameNumber = int(currentFrameNumber)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = img0  ##### Ganti im0s menjadi img0

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                currentDetectionList = list()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    # label = f'{names[c]} {conf:.2f}'
                    if names[c] == 'vessel':
                        label = f'vessel_{vessel_number}'
                        vessel_number += 1
                    else:
                        label = f'kayak_{kayak_number}'
                        kayak_number += 1

                    # plot Bbox on image
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

                    # bounding box
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                    # we only do this for frames we want to infer
                    if currentFrameNumber in framesToInfer:
                        print(f'Doing inference on frame number {currentFrameNumber}')

                        # Create ImageDetection object to store object class and bbox
                        currentDetection = ImageDetection(names[c], xywh)

                        # add to current list of dections
                        currentDetectionList.append(currentDetection)

                if currentDetectionList is not None:
                    allDetectionList.append([currentFrameNumber,
                                             numKayakVesselAndBbox(currentDetectionList)]
                                            )

        vid_writer.write(im0)
        _, img0 = cap.read()

    vid_writer.release()

    endTime = datetime.datetime.now()
    totalTime = endTime - startTime

    print(f'total time for model inference on video: {totalTime}')
    print()

    print('Saving results for frames of interest to csv now')
    saveResultsToCsv(allDetectionList)
    print('Done saving results for frames of interest.')
    print(f'Saved at {CSV_DIR}')

    return save_path


if "__name__" == "__main__":
    framesToInfer = getFramesToInfer()
    modelInference(framesToInfer)

    csv_read = pd.read_csv(CSV_DIR)
    csv_read.head()
