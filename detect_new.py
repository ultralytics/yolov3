# import libraries
import argparse
from pathlib import Path
import os

import torch
# from IPython.display import Image, clear_output

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

# root directory
CWD = os.getcwd()
ROOT_FOLDER = os.path.join(CWD, "Assignment3")
DATA_FOLDER = os.path.join(ROOT_FOLDER, "Data")

# adding arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='horizon_1_ship.avi', help='source')
parser.add_argument('--frames', type=str, default='horizon_1_ship.json', help='frames to infer json format')
opt = parser.parse_args()

video_name = opt.source
VIDEO_TO_INFER_DIR = os.path.join(DATA_FOLDER, video_name)

FRAMES_TO_INFER_JSON = os.path.join(ROOT_FOLDER, "JSON", opt.frames)

# outputs
SAVE_RESULTS_DIR = os.path.join(ROOT_FOLDER, "Results")
SAVE_VIDEO_DIR = os.path.join(SAVE_RESULTS_DIR, 'Video')  # need to add this subfolder
video_name_no_ext = video_name.replace(".avi", "")
CSV_DIR = os.path.join(SAVE_RESULTS_DIR, f"{video_name_no_ext}.csv")

# weights
WEIGHTS = os.path.join(ROOT_FOLDER, "Weights", opt.weights)

# Device to use (e.g. "0", "1", "2"... or "cpu")
if torch.cuda.is_available():
    DEVICE = "0"
else:
    DEVICE = "cpu"
print(f'Device {DEVICE}')

# Intended image size must be in multiples of 32
# Image will be resized for training
IMAGE_SIZE = 640


# define ImageDetection class
class ImageDetection:
    def __init__(self, objectType, bbox, confidence):
        self.objectType = objectType
        self.bbox = bbox
        self.confidence = confidence


# define all helper functions
# get object count
def numKayakVesselAndBbox(listOfAllObjectsDetected):
    resultList = list()

    numKayak = 0
    numVessel = 0

    vesselBboxString = ""
    kayakBboxString = ""

    for objectDetected in listOfAllObjectsDetected:
        objectType = objectDetected.objectType
        objectBbox = objectDetected.bbox
        objectConfident = objectDetected.confidence

        if objectType == 'vessel':
            numVessel += 1
            vesselBbox = [str(round(bboxCoord, 4)) for bboxCoord in objectBbox]
            vesselBboxString += f'{"_".join(vesselBbox)}'
            vesselBboxString += f"_{objectConfident};"

        else:
            numKayak += 1
            kayakBbox = [str(round(bboxCoord, 4)) for bboxCoord in objectBbox]
            kayakBboxString += f'{"_".join(kayakBbox)}'
            kayakBboxString += f"_{objectConfident};"

    resultList.append(numVessel)
    resultList.append(numKayak)
    resultList.append(vesselBboxString)
    resultList.append(kayakBboxString)
    print(resultList)
    return resultList


# helper function to save results
def saveResultsToCsv(dataToSaveToCsv):
    header = ['frame_index', 'no_ships', 'no_kayaks', 'ships_coordinates', 'kayaks_coordinates']

    with open(CSV_DIR, 'w', encoding='UTF8', newline='') as csvFile:
        writer = csv.writer(csvFile)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(dataToSaveToCsv)

    print(f'Data saved to {CSV_DIR}')


# getting frames from json input
def getFramesToInfer():
    jsonFile = open(FRAMES_TO_INFER_JSON, )
    jsonFileLoaded = json.load(jsonFile)
    framesToInfer = jsonFileLoaded["frames_to_infer"]
    return framesToInfer


# main function for video inference
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
    save_path = save_path.replace('.avi', '.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f'fps {fps}')
    print(f'w x h = {w} x {h}')

    # the list to store (frameNumber, imageDetection)
    allDetectionList = list()

    count = 0

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
        currentFrameNumber = int(currentFrameNumber) - 1  # frames are 0-index based

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
                        conf = f'{conf:.2f}'
                        currentDetection = ImageDetection(names[c], xywh, conf)

                        # add to current list of dections
                        currentDetectionList.append(currentDetection)

                if currentFrameNumber in framesToInfer:
                    currentListAdd = list()
                    currentListAdd.append(currentFrameNumber)
                    currentListAdd.extend(numKayakVesselAndBbox(currentDetectionList))
                    allDetectionList.append(currentListAdd)

        vid_writer.write(im0)
        
        # we skip some frames
        count += 1
        if count % 3 == 0 and count not in getFramesToInfer():
            count += 1
        else:
            count = count
        cap.set(1, count)
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


# main
if __name__ == "__main__":
    framesToInfer = getFramesToInfer()
    modelInference(framesToInfer)