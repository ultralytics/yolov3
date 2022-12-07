import math
import random
import sys
from xmlrpc.server import DocXMLRPCRequestHandler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from perlin_noise import Inverted_Perlin_Noise_Creator, Perlin_Noise_Creator

from median_pool import MedianPool2d

factor = math.sqrt(2)/2

# Here we have the tile function and the 6 base functions

class Tile_Creator:

    def __init__(self, list_of_shape):
        # self.list_classes_tile = [Tile_Creator_Circle, Tile_Creator_Ellipse, Tile_Creator_Square,
        #                           Tile_Creator_Rectangle, Tile_Creator_Triangle, Tile_Creator_Trapezoid]
        self.list_classes_tile = [Tile_Creator_Ellipse, Tile_Creator_Rectangle, Tile_Creator_Triangle, Tile_Creator_Trapezoid]
        # self.number_of_params = [5,6,5,6,6,6]
        self.number_of_params = [6,6,6,6]
        self.list_of_shape = list_of_shape
        self.Centroids_Creator()

    def __call__(self, dim, params):
        flag = 0
        self.params_navigator = 0
        mask_tot = torch.zeros((3,dim,dim))
        color = torch.zeros((3,dim,dim))

        # going through the list of the number of shapes
        for i in range(len(self.list_of_shape)):
            # check if we have to use some shape (is there a specific shape?)
            if self.list_of_shape[i] != 0:
                # define the specific tile creator
                tile_creator_shape = self.list_classes_tile[i]()
                # add the needed number of shapes
                for j in range(self.list_of_shape[i]):
                    tile, color, mask = tile_creator_shape(dim,params[self.params_navigator:self.params_navigator+self.number_of_params[i]])
                    mask_tot += mask
                    self.params_navigator += self.number_of_params[i]
                    if flag == 0:
                        tensor = tile
                        color_tot = color
                        flag = 1
                    else:
                        tensor = tensor*(1-mask) + tile*mask

        mask_tot.data.clamp_(0, 1)
        return tensor, color_tot, mask_tot

    def Centroids_Creator(self):
        k = sum(self.list_of_shape)
        centroids = initialize(k)
        self.centroids = [torch.from_numpy(item).float() for item in centroids]

    def Params_Creator(self):
        #define the base of the tile
        params = []
        count = 0

        # going through the list of the number of shapes
        for i in range(len(self.list_of_shape)):
            # check if we have to use some shape
            if self.list_of_shape[i] != 0:
                # define the specific tile creator
                tile_creator_shape = self.list_classes_tile[i]()
                # add the needed number of shapes
                for j in range(self.list_of_shape[i]):
                    params = params + tile_creator_shape.Params_Creator(self.centroids[count])
                    count += 1

        return params

    def Params_Clamp(self,params):
        self.params_navigator = 0

        # going through the list of the number of shapes
        for i in range(len(self.list_of_shape)):
            # check if we have to use some shape (are there circles?)
            if self.list_of_shape[i] != 0:
                # define the specific tile creator
                tile_creator_shape = self.list_classes_tile[i]()
                # add the needed number of shapes
                for _ in range(self.list_of_shape[i]):
                    params[self.params_navigator:self.params_navigator+self.number_of_params[i]] = tile_creator_shape.Params_Clamp(params[self.params_navigator:self.params_navigator+self.number_of_params[i]])
                    self.params_navigator += self.number_of_params[i]

        return params

    def Give_Color_Perlin(self,params):
        self.params_navigator = 0

        # returning the color first external color
        for i in range(len(self.list_of_shape)):
            # check if we have to use some shape (are there circles?)
            if self.list_of_shape[i] != 0:
                # define the specific tile creator
                tile_creator_shape = self.list_classes_tile[i]()
                return tile_creator_shape.Give_Color_Perlin(params[self.params_navigator:self.params_navigator+self.number_of_params[i]])

####################################################################################################
####################################################################################################

# function to compute euclidean distance
def distance(p1, p2):
	return np.sum((p1 - p2)**2)

# initialization algorithm
def initialize(k):

    # Create the 500 random points in the space [0,1]x[0,1]
    data = np.random.uniform(low=0,high=1,size=(500,2))

    np.random.shuffle(data)
    centroids = []
    centroids.append(data[np.random.randint(data.shape[0]), :])

	## compute remaining k - 1 centroids
    for c_id in range(k - 1):

        ## initialize a list to store distances of data
        ## points from nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            ## compute distance of 'point' from each of the previously
            ## selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
    return centroids

####################################################################################################
####################################################################################################

class Tile_Creator_Circle:

    def __init__(self):
        pass

    def __call__(self, dim, params):
        # Distances Tensor
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.sqrt((i-dim*params[4])**2 + (j-dim*params[3])**2) - (dim/2)*params[0]
        coeff = image.sigmoid()

        # Creation of the colors tensors
        color1_image = params[1].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[2].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        return coeff*color1_image + (1-coeff)*color2_image, color1_image, (1-coeff)

    def Params_Creator(self, centroids):
        a = torch.tensor(0.33)
        a.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        # x = torch.tensor(random.uniform(0,1))
        # x.requires_grad_(True)
        # y = torch.tensor(random.uniform(0,1))
        # y.requires_grad_(True)
        x = centroids[0]
        x.requires_grad_(True)
        y = centroids[1]
        y.requires_grad_(True)

        params = [a, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        # Defining the shape characteristics
        params[0].data.clamp_(0, 0.95)
        # Defining the colors: background and geometrical shape
        params[1].data.clamp_(0, 1)
        params[2].data.clamp_(0, 1)
        # Defining the coordinates of the geometrical shape
        params[3].data.clamp_(params[0]/2 + 0.025, 1-params[0]/2 - 0.025)
        params[4].data.clamp_(params[0]/2 + 0.025, 1-params[0]/2 - 0.025)

        return params

    def Give_Color_Perlin(self,params):
        return params[1]

####################################################################################################

class Tile_Creator_Ellipse:

    def __init__(self):
        pass

    def __call__(self,dim, params):
        # Distances Tensor
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = ((i-dim*params[5])*(dim/2)*params[1])**2 + ((j-dim*params[4])*(dim/2)*params[0])**2-((dim/2)*params[0]*(dim/2)*params[1])**2
        coeff = image.sigmoid()

        # Creation of the colors tensors
        color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        return coeff*color1_image + (1-coeff)*color2_image, color1_image, (1-coeff)

    def Params_Creator(self, centroids):
        a = torch.tensor(0.33)
        a.requires_grad_(True)
        b = torch.tensor(0.33)
        b.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        # x = torch.tensor(random.uniform(0,1))
        # x.requires_grad_(True)
        # y = torch.tensor(random.uniform(0,1))
        # y.requires_grad_(True)
        x = centroids[0]
        x.requires_grad_(True)
        y = centroids[1]
        y.requires_grad_(True)


        params = [a, b, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        # Defining the shape characteristics
        params[0].data.clamp_(0, 0.95)
        params[1].data.clamp_(0, 0.95)
        # Defining the colors: background and geometrical shape
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        # Defining the coordinates of the geometrical shape
        params[4].data.clamp_(params[1]/2 + 0.025, 1-params[1]/2 - 0.025)
        params[5].data.clamp_(params[0]/2 + 0.025, 1-params[0]/2 - 0.025)

        return params

    def Give_Color_Perlin(self,params):
        return params[2]

####################################################################################################

class Tile_Creator_Square:

    def __init__(self):
        pass

    def __call__(self, dim, params):
        # Tensore delle distanze
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs((i-dim*params[4])/params[0]),torch.abs((j-dim*params[3])/params[0])) - dim/2
        coeff = image.sigmoid()

        # Creation of the colors tensors
        color1_image = params[1].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[2].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        return coeff*color1_image + (1-coeff)*color2_image, color1_image, (1-coeff)

    def Params_Creator(self, centroids):
        a = torch.tensor(0.33)
        a.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        # x = torch.tensor(random.uniform(0,1))
        # x.requires_grad_(True)
        # y = torch.tensor(random.uniform(0,1))
        # y.requires_grad_(True)
        x = centroids[0]
        x.requires_grad_(True)
        y = centroids[1]
        y.requires_grad_(True)

        params = [a, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        # Defining the shape characteristics
        params[0].data.clamp_(0, 0.95)
        # Defining the colors: background and geometrical shape
        params[1].data.clamp_(0, 1)
        params[2].data.clamp_(0, 1)
        # Defining the coordinates of the geometrical shape
        params[3].data.clamp_(params[0]/2 + 0.025, 1-params[0]/2 - 0.025)
        params[4].data.clamp_(params[0]/2 + 0.025, 1-params[0]/2 - 0.025)
        return params

    def Give_Color_Perlin(self,params):
        return params[1]

####################################################################################################

class Tile_Creator_Rectangle:

    def __init__(self):
        pass

    def __call__(self,dim, params):
        # Distances Tensor
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs( (i-dim*params[5])/params[0] ), torch.abs( (j-dim*params[4])/params[1] ) )- dim/2
        coeff = image.sigmoid()

        # Creation of the colors tensors
        color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        return coeff*color1_image + (1-coeff)*color2_image, color1_image, (1-coeff)

    def Params_Creator(self, centroids):
        a = torch.tensor(0.33)
        a.requires_grad_(True)
        b = torch.tensor(0.33)
        b.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        # x = torch.tensor(random.uniform(0,1))
        # x.requires_grad_(True)
        # y = torch.tensor(random.uniform(0,1))
        # y.requires_grad_(True)
        x = centroids[0]
        x.requires_grad_(True)
        y = centroids[1]
        y.requires_grad_(True)

        params = [a, b, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        # Defining the shape characteristics
        params[0].data.clamp_(0, 0.95)
        params[1].data.clamp_(0, 0.95)
        # Defining the colors: background and geometrical shape
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        # Defining the coordinates of the geometrical shape
        params[4].data.clamp_(params[1]/2 + 0.025, 1-params[1]/2 - 0.025)
        params[5].data.clamp_(params[0]/2 + 0.025, 1-params[0]/2 - 0.025)

        return params

    def Give_Color_Perlin(self,params):
        return params[2]

####################################################################################################

class Tile_Creator_Triangle:

    def __init__(self):
        pass

    def __call__(self, dim, params):
        # Distances Tensor
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(-(i-dim*params[5])/params[1]),torch.abs(2*(j-dim*params[4])/params[0]) + (i-dim*params[5])/params[1]) - dim/2
        coeff = image.sigmoid()

        # Creation of the colors tensors
        color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        return coeff*color1_image + (1-coeff)*color2_image, color1_image, (1-coeff)

    def Params_Creator(self, centroids):
        a = torch.tensor(0.33)
        a.requires_grad_(True)
        b = torch.tensor(0.33)
        b.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        # x = torch.tensor(random.uniform(0,1))
        # x.requires_grad_(True)
        # y = torch.tensor(random.uniform(0,1))
        # y.requires_grad_(True)
        x = centroids[0]
        x.requires_grad_(True)
        y = centroids[1]
        y.requires_grad_(True)

        params = [a, b, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        # Defining the shape characteristics
        params[0].data.clamp_(0, 0.95)
        params[1].data.clamp_(0, 0.95)
        # Defining the colors: background and geometrical shape
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        # Defining the coordinates of the geometrical shape
        params[4].data.clamp_(params[1]/2 + 0.025, 1-params[1]/2 - 0.025)
        params[5].data.clamp_(params[0]/2 + 0.025, 1-params[0]/2 - 0.025)

        return params

    def Give_Color_Perlin(self,params):
        return params[2]

# class Tile_Creator_Triangle(object):

#     def __init__(self):
#         pass

#     def __call__(self, dim, params):
#         # Distances Tensor
#         image = torch.zeros((3,dim,dim))
#         for i in range(dim):
#             for j in range(dim):
#                 image[:,i,j] = torch.max(torch.abs(-(i-dim*params[5])/params[1]),torch.abs(2*(j-dim*params[4])/params[0]) + (i-dim*params[5])/params[1]) - dim/2
#         coeff = image.sigmoid()

#         # Creation of the colors tensors
#         color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
#         color1_image = color1_image.expand(-1, dim, dim)

#         color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
#         color2_image = color2_image.expand(-1, dim, dim)

#         tile = coeff*color1_image + (1-coeff)*color2_image
#         external_color = color1_image
#         mask = (1-coeff)

#         # Translation
#         delta_x = params[4] - 0.5
#         delta_y = params[5] - 0.5



#         # Rotation
#         angle = random.uniform(0,360)
#         tile_rotated = TF.rotate(tile_translated, angle)

#         # Translation



#     def Params_Creator(self, centroids):
#         a = torch.tensor(0.33)
#         a.requires_grad_(True)
#         b = torch.tensor(0.33)
#         b.requires_grad_(True)
#         color1 = torch.tensor([0.5,0.5,0.5])
#         color1.requires_grad_(True)
#         color2 = torch.tensor([1,0.5,0.5])
#         color2.requires_grad_(True)
#         # x = torch.tensor(random.uniform(0,1))
#         # x.requires_grad_(True)
#         # y = torch.tensor(random.uniform(0,1))
#         # y.requires_grad_(True)
#         x = centroids[0]
#         x.requires_grad_(True)
#         y = centroids[1]
#         y.requires_grad_(True)

#         params = [a, b, color1, color2, x, y]
#         return params

#     def Params_Clamp(self,params):
#         # Defining the shape characteristics
#         params[0].data.clamp_(0, 0.95)
#         params[1].data.clamp_(0, 0.95)
#         # Defining the colors: background and geometrical shape
#         params[2].data.clamp_(0, 1)
#         params[3].data.clamp_(0, 1)
#         # Defining the coordinates of the geometrical shape
#         params[4].data.clamp_(params[1]/2 + 0.025, 1-params[1]/2 - 0.025)
#         params[5].data.clamp_(params[0]/2 + 0.025, 1-params[0]/2 - 0.025)

#         return params

#     def Give_Color_Perlin(self,params):
#         return params[2]

####################################################################################################

class Tile_Creator_Trapezoid:

    def __init__(self):
        pass

    def __call__(self,dim, params):
        # Distances Tensor
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(2*(i-dim*params[5])/params[1]),torch.abs(3*(j-dim*params[4])/params[0]) + (i-dim*params[5])/params[1]) - dim
        coeff = image.sigmoid()

        # Creation of the colors tensors
        color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        return coeff*color1_image + (1-coeff)*color2_image, color1_image, (1-coeff)

    def Params_Creator(self, centroids):
        a = torch.tensor(0.33)
        a.requires_grad_(True)
        b = torch.tensor(0.5)
        b.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        # x = torch.tensor(random.uniform(0,1))
        # x.requires_grad_(True)
        # y = torch.tensor(random.uniform(0,1))
        # y.requires_grad_(True)
        x = centroids[0]
        x.requires_grad_(True)
        y = centroids[1]
        y.requires_grad_(True)

        params = [a, b, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        # Defining the shape characteristics
        params[0].data.clamp_(0, 0.95)
        params[1].data.clamp_(0, 0.95)
        # Defining the colors: background and geometrical shape
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        # Defining the coordinates of the geometrical shape
        params[4].data.clamp_(params[1]/2 + 0.025, 1-params[1]/2 - 0.025)
        params[5].data.clamp_(params[0]/2 + 0.025, 1-params[0]/2 - 0.025)

        return params

    def Give_Color_Perlin(self,params):
        return params[2]
