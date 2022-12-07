import torch
import torch.nn as nn

# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import numpy as np
# import os
# import fnmatch
# from torch.utils.data import Dataset
# from PIL import Image
# import random
# import math
# import torchvision.transforms.functional as TF
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
# import torch.optim as optim
# from torch.autograd import Variable


####################################################################################################
####################################################################################################
### Loss Function ###

class max_prob_class(nn.Module):
    """max_prob_class: extracts max class probability for a specific class from YOLO output.
    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
    The loss function compute the product between the objectness score and the class specific probability
    """

    def __init__(self, cls_id):
        super().__init__()
        self.cls_id = cls_id

    def forward(self, outputs):
        # Compute the batch dimension
        batch_dim = outputs[1][0].shape[0]

        # Get the propabilities we need from the tensors
        yolo_output1 = outputs[1][0]
        yolo_output1 = torch.reshape(yolo_output1, (batch_dim,3*80*80,85)).sigmoid()

        yolo_output2 = outputs[1][1]
        yolo_output2 = torch.reshape(yolo_output2, (batch_dim,3*40*40,85)).sigmoid()

        yolo_output3 = outputs[1][2]
        yolo_output3 = torch.reshape(yolo_output3, (batch_dim,3*20*20,85)).sigmoid()

        final_tensor = torch.cat((yolo_output1,yolo_output2,yolo_output3), dim=1)

        objectness = final_tensor[:, :, 4]
        cond_prob_targeted_class = final_tensor[:, :, 5+self.cls_id]

        confs_if_object_normal = objectness*cond_prob_targeted_class

        max_conf, _ = torch.max(confs_if_object_normal, dim=1)

        return max_conf

####################################################################################################

class max_prob_class2(nn.Module):
    """max_prob_class2: extracts max class probability for a specific class from YOLO output.
    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
    The loss function considers just the class specific probability, regardless of the objectness score
    """

    def __init__(self, cls_id):
        super().__init__()
        self.cls_id = cls_id

    def forward(self, outputs):
        # Compute the batch dimension
        batch_dim = outputs[1][0].shape[0]

        # Get the propabilities we need from the tensors
        yolo_output1 = outputs[1][0]
        yolo_output1 = torch.reshape(yolo_output1, (batch_dim,3*80*80,85))

        yolo_output2 = outputs[1][1]
        yolo_output2 = torch.reshape(yolo_output2, (batch_dim,3*40*40,85))

        yolo_output3 = outputs[1][2]
        yolo_output3 = torch.reshape(yolo_output3, (batch_dim,3*20*20,85))

        final_tensor = torch.cat((yolo_output1,yolo_output2,yolo_output3), dim=1)

        cond_prob_targeted_class = final_tensor[:, :, 5+self.cls_id]

        max_conf, _ = torch.max(cond_prob_targeted_class, dim=1)

        return max_conf

####################################################################################################

class max_prob_obj(nn.Module):
    """max_prob_obj: extracts max class probability from YOLO output.
    Module providing the functionality necessary to extract the max object probability from YOLO output.
    The loss function considers just the objectness score, without considering the class
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        # Compute the batch dimension
        batch_dim = outputs[1][0].shape[0]

        # Get the propabilities we need from the tensors
        yolo_output1 = outputs[1][0]
        yolo_output1 = torch.reshape(yolo_output1, (batch_dim,3*80*80,85))

        yolo_output2 = outputs[1][1]
        yolo_output2 = torch.reshape(yolo_output2, (batch_dim,3*40*40,85))

        yolo_output3 = outputs[1][2]
        yolo_output3 = torch.reshape(yolo_output3, (batch_dim,3*20*20,85))

        final_tensor = torch.cat((yolo_output1,yolo_output2,yolo_output3), dim=1)

        objectness = final_tensor[:, :, 4]

        max_conf, _ = torch.max(objectness, dim=1)

        return max_conf

####################################################################################################

class new_loss_tprob(nn.Module):
    """MaxProbExtractor: extracts max class probability for the max objectness score from YOLO output.
    Module providing the functionality necessary to extract the max class probability of the object with the max probability
    from YOLO output. The loss function doesn't use the product between objectness score and class score
    """

    def __init__(self, cls_id):
        super().__init__()
        self.cls_id = cls_id

    def forward(self, outputs):
        # Compute the batch dimension
        batch_dim = outputs[1][0].shape[0]

        # Get the propabilities we need from the tensors
        yolo_output1 = outputs[1][0]
        yolo_output1 = torch.reshape(yolo_output1, (batch_dim,3*80*80,85))

        yolo_output2 = outputs[1][1]
        yolo_output2 = torch.reshape(yolo_output2, (batch_dim,3*40*40,85))

        yolo_output3 = outputs[1][2]
        yolo_output3 = torch.reshape(yolo_output3, (batch_dim,3*20*20,85))

        final_tensor = torch.cat((yolo_output1,yolo_output2,yolo_output3), dim=1)

        objectness = final_tensor[:, :, 4]
        _ , index = torch.max(objectness, dim=1)

        cond_prob_targeted_class = final_tensor[:, :, 5+self.cls_id]
        max_cond_prob_targeted_class = cond_prob_targeted_class[:,index]
        max, _ = torch.max(max_cond_prob_targeted_class, dim=1)
        max

        return max
