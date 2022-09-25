import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
import math
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from median_pool import MedianPool2d
import math
from perlin_noise import Perlin_Noise_Creator, Inverted_Perlin_Noise_Creator
factor = math.sqrt(2)/2

# Here we have the tile function and the 6 base functions

class Tile_Creator(object):
    
    def __init__(self, list_of_shape):
        self.list_classes_tile = [Tile_Creator_Circle, Tile_Creator_Ellipse, Tile_Creator_Square, 
                                  Tile_Creator_Rectangle, Tile_Creator_Triangle, Tile_Creator_Trapezoid]
        self.number_of_params = [5,6,5,6,6,6]
        self.list_of_shape = list_of_shape
        self.params_navigator = 0 

    def __call__(self, dim, params):
        flag = 0
        mask_tot = torch.zeros((3,dim,dim))
        color = torch.zeros((3,dim,dim))
        
        # going through the list of the number of shapes 
        for i in range(len(self.list_of_shape)):
            # check if we have to use some shape (are there circles?)
            if self.list_of_shape[i] != 0:
                # define the specific tile creator
                tile_creator_shape = self.list_classes_tile[i]()
                # add the needed number of shapes
                for _ in range(self.list_of_shape[i]):
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

                    

    def Params_Creator(self):
        #define the base of the tile 
        params = []
        
        # going through the list of the number of shapes 
        for i in range(len(self.list_of_shape)):
            # check if we have to use some shape 
            if self.list_of_shape[i] != 0:
                # define the specific tile creator
                tile_creator_shape = self.list_classes_tile[i]()
                # add the needed number of shapes
                for j in range(self.list_of_shape[i]):
                    params = params + tile_creator_shape.Params_Creator()
        
        return params
    
    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, 1)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        params[4].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[1]

####################################################################################################
####################################################################################################

class Tile_Creator_Circle(object):
    
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

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        color1 = torch.tensor([1,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,1])
        color2.requires_grad_(True)
        x = torch.tensor(random.uniform(0,1))
        x.requires_grad_(True)
        y = torch.tensor(random.uniform(0,1))
        y.requires_grad_(True)

        params = [a, color1, color2, x, y]
        return params
    
    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, 1)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        params[4].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[1]

####################################################################################################

class Tile_Creator_Ellipse(object):

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

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        b = torch.tensor(0.8)
        b.requires_grad_(True)
        color1 = torch.tensor([1,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,1])
        color2.requires_grad_(True)
        x = torch.tensor(random.uniform(0,1))
        x.requires_grad_(True)
        y = torch.tensor(random.uniform(0,1))
        y.requires_grad_(True)

        params = [a, b, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        params[4].data.clamp_(0, 1)
        params[5].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]

####################################################################################################

class Tile_Creator_Square(object):

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

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        color1 = torch.tensor([1.0,1.0,1.0])
        color1.requires_grad_(True)
        color2 = torch.tensor([1.0,1.0,1.0])
        color2.requires_grad_(True)
        x = torch.tensor(random.uniform(0,1))
        x.requires_grad_(True)
        y = torch.tensor(random.uniform(0,1))
        y.requires_grad_(True)

        params = [a, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, 1)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        params[4].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[1]

####################################################################################################

class Tile_Creator_Rectangle(object):

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

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        b = torch.tensor(0.25)
        b.requires_grad_(True)
        color1 = torch.tensor([1,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,1])
        color2.requires_grad_(True)
        x = torch.tensor(random.uniform(0,1))
        x.requires_grad_(True)
        y = torch.tensor(random.uniform(0,1))
        y.requires_grad_(True)

        params = [a, b, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 
        params[4].data.clamp_(0, 1)
        params[5].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]

####################################################################################################

class Tile_Creator_Triangle(object):

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

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        b = torch.tensor(0.5)
        b.requires_grad_(True)
        color1 = torch.tensor([1,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,1])
        color2.requires_grad_(True)
        x = torch.tensor(random.uniform(0,1))
        x.requires_grad_(True)
        y = torch.tensor(random.uniform(0,1))
        y.requires_grad_(True)

        params = [a, b, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 
        params[4].data.clamp_(0, 1)
        params[5].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]

####################################################################################################

class Tile_Creator_Trapezoid(object):

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

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        b = torch.tensor(0.5)
        b.requires_grad_(True)
        color1 = torch.tensor([1,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,1])
        color2.requires_grad_(True)
        x = torch.tensor(random.uniform(0,1))
        x.requires_grad_(True)
        y = torch.tensor(random.uniform(0,1))
        y.requires_grad_(True)

        params = [a, b, color1, color2, x, y]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 
        params[4].data.clamp_(0, 1)
        params[5].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]