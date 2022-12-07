import os
import glob
import torch
import cv2

def IoUArea(x1,y1,w1,h1,x2,y2,w2,h2):
    x_start_point1 = x1-w1/2
    y_start_point1 = y1-h1/2
    x_end_point1 = x1+w1/2
    y_end_point1 = y1+h1/2

    x_start_point2 = x2-w2/2
    y_start_point2 = y2-h2/2
    x_end_point2 = x2+w2/2
    y_end_point2 = y2+h2/2

    x_start_point = max(x_start_point1, x_start_point2)
    y_start_point = max(y_start_point1, y_start_point2)
    x_end_point = min(x_end_point1, x_end_point2)
    y_end_point = min(y_end_point1, y_end_point2)

    return (x_end_point-x_start_point)*(y_end_point-y_start_point)

def count_people(IoU_thresh, Confidence_thresh):
    labels_path = '/home/andread98/yolov3/MyWork/data_mask_test/ground_truth_people'
    # labels_path = '/home/andread98/yolov3/MyWork/data_mask_test/prediction'
    labels = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

    # Counter for the people in the ground truth
    people_ground_truth = 0
    # Counter for the people in the prediction
    people_prediction = 0
    # Intersection count
    people_intersection = 0

    for label in labels:
        # ground truth path
        ground_truth_path = '/home/andread98/yolov3/MyWork/data_mask_test/ground_truth_people/' + label
        # ground_truth_path = '/home/andread98/yolov3/MyWork/data_mask_test/prediction/' + label
        # prediction path
        prediction_path = '/home/andread98/yolov3/MyWork/data_mask_test/prediction_attack/' + label

        with open(ground_truth_path) as fp:
            ground_truth = []
            # x,y,w,h da estrarre per ogni riga del txt 
            for line in fp:
                numbers = []
                #Append the list of numbers to the result array
                #Convert each number to an integer
                #Split each line of whitespace
                numbers.extend( [float(item) for item in line.split()])
                ground_truth.append(numbers)
        
        # print(f'Ground truth: {ground_truth}')
        people_ground_truth += len(ground_truth)

        with open(prediction_path) as fp:
            prediction = []
            # x,y,w,h da estrarre per ogni riga del txt 
            for line in fp:
                numbers = []
                #Append the list of numbers to the result array
                #Convert each number to an integer
                #Split each line of whitespace
                numbers.extend( [float(item) for item in line.split()])
                prediction.append(numbers)

        # print(f'Prediction: {prediction}')
        people_prediction += len(prediction)

        for i in range(len(ground_truth)):
            line_ground_truth = ground_truth[i]
            _, x_gt, y_gt, w_gt, h_gt = line_ground_truth
            # x_gt, y_gt, w_gt, h_gt, prob, _ = line_ground_truth

            area_ground_truth = w_gt*h_gt

            max_area_ratio = 0 
            max_pos = None

            for j in range(len(prediction)):
                line_prediction = prediction[j]
                x_pr, y_pr, w_pr, h_pr, prob, _ = line_prediction
                area_prediction = w_pr*h_pr

                area_intersection = IoUArea(x_gt, y_gt, w_gt, h_gt, x_pr, y_pr, w_pr, h_pr)

                area_union = area_ground_truth + area_prediction - area_intersection

                if area_intersection/area_union > max_area_ratio:
                    max_area_ratio = area_intersection/area_union
                    max_pos = j

            if max_area_ratio > IoU_thresh and prediction[max_pos][4] > Confidence_thresh:
                del prediction[max_pos]
                people_intersection += 1
        if prediction:
            print(f'Not detected {prediction} in image {label}')

    print(f'People in the ground truth: {people_ground_truth}')
    print(f'People in the prediction: {people_prediction}')
    print(f'People in the intersection: {people_intersection}')

    return people_intersection