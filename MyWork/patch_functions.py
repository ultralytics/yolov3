# from this import d
# from turtle import forward
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
# import os
# import fnmatch
# from torch.utils.data import Dataset
# from PIL import Image
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
# import torch.optim as optim
# from torch.autograd import Variable

####################################################################################################
####################################################################################################
### Patch Generator Functions ###

class Fractal_Patch_Generator(nn.Module):

    def __init__(self, dim_patch, dim_image, max_dim, tile_class, angle_type, BackgroundStyle, mask_function = None):
        super(Fractal_Patch_Generator, self).__init__()
        #Dimension of the patch, smallest component of the attack
        self.dim_patch = dim_patch 
        #Dimension of the image we want to create
        self.dim_image = dim_image
        #Max dimension for the patch
        self.max_dim = max_dim
        #Tile function
        self.tile_class = tile_class
        #Mask function
        self.mask_function = mask_function
        #Angle Type
        self.angle_type = angle_type
        #Background Style
        self.BackgroundStyle = BackgroundStyle

        #Check if they are compatible
        if self.dim_image % self.dim_patch != 0:
            raise Exception('Patch and Image do not have compatible dimensions. Please select the image as a multiple of patch.')

        # How many lines in the grid?
        self.dim_grid = int(self.dim_image/self.dim_patch)
    
    def populate(self, params):

        if self.BackgroundStyle == 0: # Normal situation with a plain color
            self.patches = []
            self.ex_colors = []
            self.masks = []

            for i in range(1,self.max_dim+1):
                patch, ex_color, _ = self.tile_class(self.dim_patch*i, params)
                mask = self.mask_function(self.dim_patch*i)

                self.patches.append(patch)
                self.ex_colors.append(ex_color)
                self.masks.append(mask)
        
        else: # Using the perlin noise in the background
            self.patches = []
            self.masks = []
            # Creation of the perlin noise with the function in perlin_noise.py
            if self.BackgroundStyle == 1:
                self.perlin_noise = Perlin_Noise_Creator(self.dim_image, self.tile_class.Give_Color_Perlin(params))
            if self.BackgroundStyle == 2:
                self.perlin_noise = Inverted_Perlin_Noise_Creator(self.dim_image, self.tile_class.Give_Color_Perlin(params))

            for i in range(1,self.max_dim+1):
                patch, _, mask = self.tile_class(self.dim_patch*i, params)
                self.patches.append(patch)
                self.masks.append(mask)            
    
    def application(self):
        #Creation of the complete image
        self.image = torch.rand((3,self.dim_image,self.dim_image))
        #Creation of the bool vector 
        self.bool_matrix = np.ones((self.dim_grid,self.dim_grid), dtype=int)
        #Creation of the index vector
        self.index_vector = np.arange(0,(self.dim_grid**2))
        #Creation of the shuffled version
        shuffled_index_vector = np.random.choice(self.index_vector, size=self.dim_grid**2, replace=False)
        self.complete_mask = torch.zeros((3,self.dim_image,self.dim_image))

        for index in shuffled_index_vector:
            #Translate the index in coordinates
            i = int(index/self.dim_grid) #row
            j = index%self.dim_grid      #column

            #Check if the corner is still available
            if self.bool_matrix[i][j]:
                av_dim = self.available_dimensions(i,j)
                
                #Choose randomly the dimension
                chosen_dim = int(np.random.choice(av_dim, 1))
                # print(chosen_dim)

                #Change the bool in the bool matrix in False, no longer available
                self.bool_matrix[i:i+chosen_dim,j:j+chosen_dim] = 0

                if self.angle_type == 0:
                    self.image[:,i*self.dim_patch:(i+chosen_dim)*self.dim_patch,j*self.dim_patch:(j+chosen_dim)*self.dim_patch] = self.patches[chosen_dim-1]

                elif self.angle_type == 1:
                    angle = random.choice([0,90,180,270])
                    #Apply the patch to the image
                    self.image[:,i*self.dim_patch:(i+chosen_dim)*self.dim_patch,j*self.dim_patch:(j+chosen_dim)*self.dim_patch] = TF.rotate(self.patches[chosen_dim-1],angle)

                else:
                    #Select the angle
                    angle = random.uniform(0,360)

                    

                    if self.BackgroundStyle == 0: # Normal case
                        #Rotate
                        out = TF.rotate(self.patches[chosen_dim-1], angle)
                        # Getting the background color
                        color = self.ex_colors[chosen_dim-1]

                        # Color in the angles
                        mask = self.masks[chosen_dim-1]
                        out[mask] = color[mask]

                        # Apply the patch to the image
                        self.image[:,i*self.dim_patch:(i+chosen_dim)*self.dim_patch,j*self.dim_patch:(j+chosen_dim)*self.dim_patch] = out

                    else: # Perlin Noise
                        #Rotate
                        out = TF.rotate(self.patches[chosen_dim-1], angle)
                        # Rotate the mask
                        out2 = TF.rotate(self.masks[chosen_dim-1], angle)
                        
                        # Apply the patch to the image  
                        self.image[:,i*self.dim_patch:(i+chosen_dim)*self.dim_patch,j*self.dim_patch:(j+chosen_dim)*self.dim_patch] = out
                        
                        # Tiling all the tile together to create to complete mask 
                        self.complete_mask[:,i*self.dim_patch:(i+chosen_dim)*self.dim_patch,j*self.dim_patch:(j+chosen_dim)*self.dim_patch] = out2

        if self.BackgroundStyle == 0:
            return self.image, self.complete_mask
        elif self.BackgroundStyle == 1 or self.BackgroundStyle == 2:
            self.image = self.complete_mask*self.image + (1-self.complete_mask)*self.perlin_noise
            return self.image, self.complete_mask
        else:
            return  self.complete_mask*self.image, self.complete_mask
        
    def available_dimensions(self,i,j):
        #It is always possible to put the smallest version of the patch
        av_dim = [1]

        for dim in range(2, self.max_dim+1):
            sub_matrix = self.bool_matrix[i:i+dim,j:j+dim]
            if np.sum(sub_matrix) == dim**2:
                av_dim.append(dim)
            else:
                break
        
        return av_dim

####################################################################################################
####################################################################################################
### Mask Creator ###

def Mask_Creator(dim):
    mask = torch.ones((3,dim,dim), dtype=torch.bool)
    for i in range(dim):
        for j in range(dim):
            if torch.sqrt(torch.tensor((i-(dim-1)/2)**2 + (j-(dim-1)/2)**2)) - (dim/2) < 0:
                mask[:,i,j] = False
    return mask

####################################################################################################
class Tile_Creator_Circle(object):
    
    def __init__(self):
        pass

    def __call__(self, dim, params):
        # Distances Tensor 
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.sqrt(torch.tensor((i-(dim-1)/2)**2 + (j-(dim-1)/2)**2)) - (dim/2)*params[0]
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
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)

        params = [a, color1, color2]
        return params
    
    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, 1)
        params[2].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[1]

#-----------------------------------------------------------------------------------------------------

class Tile_Creator_Double_Circle(object):
    
    def __init__(self):
        pass

    def __call__(self, dim, params):
        # Distances Tensor 
        image = torch.zeros((3,dim,dim))
        image2 = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.sqrt(torch.tensor((i-(dim-1)/2)**2 + (j-(dim-1)/2)**2)) - (dim/2)*params[0]
                image2[:,i,j] = torch.sqrt(torch.tensor((i-(dim-1)/2)**2 + (j-(dim-1)/2)**2)) - (dim/2)*params[0]*params[1]
        coeff = image.sigmoid()
        coeff2 = image2.sigmoid()

        # Creation of the colors tensors 
        color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        color3_image = params[4].unsqueeze(-1).unsqueeze(-1)
        color3_image = color3_image.expand(-1, dim, dim)

        return coeff*color1_image + (coeff2-coeff)*color2_image + (1 - coeff2)*color3_image, color1_image, (1-coeff)


    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        a2 = torch.tensor(0.50)
        a2.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        color3 = torch.tensor([0.5,0.5,0.5])
        color3.requires_grad_(True)

        params = [a, a2, color1, color2, color3]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, 1)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        params[4].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]

####################################################################################################

class Tile_Creator_Ellipse(object):

    def __init__(self):
        pass
    
    def __call__(self,dim, params):
        # Distances Tensor
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = ((i-(dim-1)/2)*(dim/2)*params[1])**2 + ((j-(dim-1)/2)*(dim/2)*params[0])**2-((dim/2)*params[0]*(dim/2)*params[1])**2
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
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)

        params = [a, b, color1, color2]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]

#-----------------------------------------------------------------------------------------------------

class Tile_Creator_Double_Ellipse(object):
    
    def __init__(self):
        pass

    def __call__(self,dim, params):
        # Distances Tensor 
        image = torch.zeros((3,dim,dim))
        image2 = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = ((i-(dim-1)/2)*(dim/2)*params[1])**2 + ((j-(dim-1)/2)*(dim/2)*params[0])**2-((dim/2)*params[0]*(dim/2)*params[1])**2
                image2[:,i,j] = ((i-(dim-1)/2)*(dim/2)*params[1]*params[3])**2 + ((j-(dim-1)/2)*(dim/2)*params[0]*params[2])**2-((dim/2)*params[0]*params[2]*(dim/2)*params[1]*params[3])**2
        coeff = image.sigmoid()
        coeff2 = image2.sigmoid()

        # Creation of the colors tensors
        color1_image = params[4].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[5].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        color3_image = params[6].unsqueeze(-1).unsqueeze(-1)
        color3_image = color3_image.expand(-1, dim, dim)

        return coeff*color1_image + (coeff2-coeff)*color2_image + (1 - coeff2)*color3_image, color1_image, (1-coeff)

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        a2 = torch.tensor(0.50)
        a2.requires_grad_(True)
        b = torch.tensor(0.50)
        b.requires_grad_(True)
        b2 = torch.tensor(0.50)
        b2.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        color3 = torch.tensor([0.5,0.5,0.5])
        color3.requires_grad_(True)

        params = [a, b, a2, b2, color1, color2, color3]

        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 
        params[4].data.clamp_(0, 1)
        params[5].data.clamp_(0, 1) 
        params[6].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[4]

####################################################################################################

class Tile_Creator_Square(object):

    def __init__(self):
        pass
    
    def __call__(self, dim, params):
        # Tensore delle distanze 
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(torch.tensor(i-(dim-1)/2)/params[0]),torch.abs(torch.tensor(j-(dim-1)/2))/params[0]) - dim/2
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
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)

        params = [a, color1, color2]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, 1)
        params[2].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[1]

#-----------------------------------------------------------------------------------------------------

class Tile_Creator_Double_Square(object):
    
    def __init__(self):
        pass

    def __call__(self, dim, params):
        # Distances Tensor 
        image = torch.zeros((3,dim,dim))
        image2 = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(torch.tensor(i-(dim-1)/2)/params[0]),torch.abs(torch.tensor(j-(dim-1)/2))/params[0]) - dim/2
                image2[:,i,j] = torch.max(torch.abs(torch.tensor(i-(dim-1)/2)/(params[0]*params[1])),torch.abs(torch.tensor(j-(dim-1)/2))/(params[0]*params[1])) - dim/2
        coeff = image.sigmoid()
        coeff2 = image2.sigmoid()

        # Creation of the colors tensors 
        color1_image = params[2].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[3].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        color3_image = params[4].unsqueeze(-1).unsqueeze(-1)
        color3_image = color3_image.expand(-1, dim, dim)

        return coeff*color1_image + (coeff2-coeff)*color2_image + (1 - coeff2)*color3_image, color1_image, (1-coeff)


    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        a2 = torch.tensor(0.5)
        a2.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        color3 = torch.tensor([0.5,0.5,0.5])
        color3.requires_grad_(True)

        params = [a, a2, color1, color2, color3]
        return params
    
    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, 1)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1)
        params[4].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]

####################################################################################################

class Tile_Creator_Rectangle(object):

    def __init__(self):
        pass
    
    def __call__(self,dim, params):
        # Distances Tensor
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(torch.tensor(i-(dim-1)/2)/params[0]),torch.abs(torch.tensor(j-(dim-1)/2))/params[1]) - dim/2
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
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)

        params = [a, b, color1, color2]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]

#-----------------------------------------------------------------------------------------------------

class Tile_Creator_Double_Rectangle(object):
    
    def __init__(self):
        pass

    def __call__(self,dim, params):
        # Distances Tensor 
        image = torch.zeros((3,dim,dim))
        image2 = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(torch.tensor(i-(dim-1)/2)/params[0]),torch.abs(torch.tensor(j-(dim-1)/2))/params[1]) - dim/2
                image2[:,i,j] = torch.max(torch.abs(torch.tensor(i-(dim-1)/2)/(params[0]*params[2])),torch.abs(torch.tensor(j-(dim-1)/2))/(params[1]*params[3])) - dim/2
        coeff = image.sigmoid()
        coeff2 = image2.sigmoid()

        # Creation of the colors tensors
        color1_image = params[4].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[5].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        color3_image = params[6].unsqueeze(-1).unsqueeze(-1)
        color3_image = color3_image.expand(-1, dim, dim)

        return coeff*color1_image + (coeff2-coeff)*color2_image + (1 - coeff2)*color3_image, color1_image, (1-coeff)

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        a2 = torch.tensor(0.50)
        a2.requires_grad_(True)
        b = torch.tensor(0.50)
        b.requires_grad_(True)
        b2 = torch.tensor(0.50)
        b2.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        color3 = torch.tensor([0.5,0.5,0.5])
        color3.requires_grad_(True)

        params = [a, b, a2, b2, color1, color2, color3]

        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 
        params[4].data.clamp_(0, 1)
        params[5].data.clamp_(0, 1) 
        params[6].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[4]

####################################################################################################

class Tile_Creator_Triangle(object):

    def __init__(self):
        pass

    def __call__(self, dim, params):
        # Distances Tensor
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(-(i-(dim-1)/2)/params[1]),torch.abs(2*(j-(dim-1)/2)/params[0]) + (i-(dim-1)/2)/params[1]) - dim/2
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
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)

        params = [a, b, color1, color2]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]

#-----------------------------------------------------------------------------------------------------

class Tile_Creator_Double_Triangle(object):
    
    def __init__(self):
        pass

    def __call__(self,dim, params):
        # Distances Tensor 
        image = torch.zeros((3,dim,dim))
        image2 = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(-(i-(dim-1)/2)/params[1]),torch.abs(2*(j-(dim-1)/2)/params[0]) + (i-(dim-1)/2)/params[1]) - dim/2
                image2[:,i,j] = torch.max(torch.abs(-(i-(dim-1)/2)/(params[1]*params[3])),torch.abs(2*(j-(dim-1)/2)/(params[0]*params[2])) + (i-(dim-1)/2)/(params[1]*params[3])) - dim/2
        coeff = image.sigmoid()
        coeff2 = image2.sigmoid()

        # Creation of the colors tensors
        color1_image = params[4].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[5].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        color3_image = params[6].unsqueeze(-1).unsqueeze(-1)
        color3_image = color3_image.expand(-1, dim, dim)

        return coeff*color1_image + (coeff2-coeff)*color2_image + (1 - coeff2)*color3_image, color1_image, (1-coeff)

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        a2 = torch.tensor(0.5)
        a2.requires_grad_(True)
        b = torch.tensor(0.50)
        b.requires_grad_(True)
        b2 = torch.tensor(0.50)
        b2.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        color3 = torch.tensor([0.5,0.5,0.5])
        color3.requires_grad_(True)

        params = [a, b, a2, b2, color1, color2, color3]

        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 
        params[4].data.clamp_(0, 1)
        params[5].data.clamp_(0, 1) 
        params[6].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[4]

####################################################################################################

class Tile_Creator_Trapezoid(object):

    def __init__(self):
        pass

    def __call__(self,dim, params):
        # Distances Tensor 
        image = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(2*(i-(dim-1)/2)/params[1]),torch.abs(3*(j-(dim-1)/2)/params[0]) + (i-(dim-1)/2)/params[1]) - dim
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
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)

        params = [a, b, color1, color2]
        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 

        return params
    
    def Give_Color_Perlin(self,params):
        return params[2]

#-----------------------------------------------------------------------------------------------------

class Tile_Creator_Double_Trapezoid(object):
    
    def __init__(self):
        pass

    def __call__(self,dim, params):
        # Distances Tensor 
        image = torch.zeros((3,dim,dim))
        image2 = torch.zeros((3,dim,dim))
        for i in range(dim):
            for j in range(dim):
                image[:,i,j] = torch.max(torch.abs(2*(i-(dim-1)/2)/params[1]),torch.abs(3*(j-(dim-1)/2)/params[0]) + (i-(dim-1)/2)/params[1]) - dim
                image2[:,i,j] = torch.max(torch.abs(2*(i-(dim-1)/2)/(params[1]*params[3])),torch.abs(3*(j-(dim-1)/2)/(params[0]*params[2])) + (i-(dim-1)/2)/(params[1]*params[3])) - dim
        coeff = image.sigmoid()
        coeff2 = image2.sigmoid()

        # Creation of the colors tensors
        color1_image = params[4].unsqueeze(-1).unsqueeze(-1)
        color1_image = color1_image.expand(-1, dim, dim)

        color2_image = params[5].unsqueeze(-1).unsqueeze(-1)
        color2_image = color2_image.expand(-1, dim, dim)

        color3_image = params[6].unsqueeze(-1).unsqueeze(-1)
        color3_image = color3_image.expand(-1, dim, dim)

        return coeff*color1_image + (coeff2-coeff)*color2_image + (1 - coeff2)*color3_image, color1_image, (1-coeff)

    def Params_Creator(self):
        a = torch.tensor(0.50)
        a.requires_grad_(True)
        a2 = torch.tensor(0.50)
        a2.requires_grad_(True)
        b = torch.tensor(0.50)
        b.requires_grad_(True)
        b2 = torch.tensor(0.50)
        b2.requires_grad_(True)
        color1 = torch.tensor([0.5,0.5,0.5])
        color1.requires_grad_(True)
        color2 = torch.tensor([0.5,0.5,0.5])
        color2.requires_grad_(True)
        color3 = torch.tensor([0.5,0.5,0.5])
        color3.requires_grad_(True)

        params = [a, b, a2, b2, color1, color2, color3]

        return params

    def Params_Clamp(self,params):
        params[0].data.clamp_(0, factor)
        params[1].data.clamp_(0, factor)
        params[2].data.clamp_(0, 1)
        params[3].data.clamp_(0, 1) 
        params[4].data.clamp_(0, 1)
        params[5].data.clamp_(0, 1) 
        params[6].data.clamp_(0, 1)

        return params
    
    def Give_Color_Perlin(self,params):
        return params[4]

####################################################################################################
####################################################################################################

class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        
        # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        # self.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

        
        self.min_contrast = 0.99
        self.max_contrast = 1.01
        self.min_brightness = -0.01
        self.max_brightness = 0.01
        self.noise_factor = 0.010
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)  # kernel_size = 7? see again
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''
    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):

        use_cuda = 1

        #adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))  # pre-processing on the image with 1 more dimension: 1 x 3 x 300 x 300, see median_pool.py

        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2  # img_size = 416, adv_patch size = patch_size in adv_examples.py, = 300
        # print('pad =' + str(pad)) # pad = 0.5*(416 - 300) = 58

        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        # print('adv_patch in load_data.py, PatchTransforme, size =' + str(adv_patch.size()))
        # adv_patch in load_data.py, PatchTransforme, size =torch.Size([1, 1, 3, 300, 300]), tot 5 dimensions

        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        # print('adv_batch in load_data.py, PatchTransforme, size =' + str(adv_batch.size()))
        # adv_batch in load_data.py, PatchTransforme, size =torch.Size([6, 14, 3, 300, 300])

        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
        # print('batch_size in load_data.py, PatchTransforme, size =' + str(batch_size))
        # batch_size in load_data.py, PatchTransforme, size =torch.Size([6, 14])

        # Contrast, brightness and noise transforms

        # Create random contrast tensor

        if use_cuda:
            contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        else:
            contrast = torch.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            # Fills self tensor (here 6 x 14) with numbers sampled from the continuous uniform distribution: 1/(max_contrast - min_contrast)

        # print('contrast1 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast1 in load_data.py, PatchTransforme, size =torch.Size([6, 14])

        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print('contrast2 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast2 in load_data.py, PatchTransforme, size =torch.Size([6, 14, 1, 1, 1])

        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        # print('contrast3 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast3 in load_data.py, PatchTransforme, size =torch.Size([6, 14, 3, 300, 300])

        # lines 206-221 could be replaced by:
        # contrast = torch.FloatTensor(adv_batch).uniform_(self.min_contrast, self.max_contrast)
        # print('contrast4 in load_data.py, PatchTransforme, size =' + str(contrast.size()))

        if use_cuda:
            contrast = contrast.cuda()
        else:
            contrast = contrast
#_________________________________________________________________________________________________________________________________________________
        # Create random brightness tensor
        if use_cuda:
            brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        else:
            brightness = torch.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)

        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        # lines 227-239 could be replaced by:
        # brightness = torch.FloatTensor(adv_batch).uniform_(self.min_brightness, self.max_brightness)
        # print('brightness in load_data.py, PatchTransforme, size =' + str(brightness.size()))

        if use_cuda:
            brightness = brightness.cuda()
        else:
            brightness = brightness

# _____________________________________________________________________________________________________________________________________________
        # Create random noise tensor
        if use_cuda:
            noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        else:
            noise = torch.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # dim: 6 x 14 x 3 x 300 x 300
#______________________________________________________________________________________________________________________________________________
        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)  # keep all elements in the range 0.000001-0.99999 (real numbers since FLoatTensor)
        # dim: 6 x 14 x 3 x 300 x 300

#______________________________________________________________________________________________________________________________________________
        # Where the label class_ids is 1 we don't want a patch (padding) --> fill mask with zero's

        cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # Consider just the first 'column' of lab_batch, where we can
                                                    # discriminate between detected person (or 'yes person') and 'no person')
                                                    # in this way, sensible data about x, y, w and h of the rectangles are not used for building the mask

        # NB torch.narrow returns a new tensor that is a narrowed version of input tensor. The dimension dim is input from start to start + length.
        # The returned tensor and input tensor share the same underlying storage.

        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # 6 x 14 x 3 x 300 x 300

        if use_cuda:
            msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask
        else:
            msk_batch = torch.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  # take a matrix of 1s, subtract that of the labels so that
                                                                                # we can have 0s where there is no person detected,
                                                                                # obtained by doing 1-1=0

        # NB! Now the mask has 1s 'above', where the labels data are sensible since they represent detected persons, and 0s where there are no detections
        # In this way, multiplying the adv_batch to this mask, built from the lab_batch tensor, allows to target only detected persons and nothing else,
        # i.e. pad with zeros the rest
#_______________________________________________________________________________________________________________________________________________
        # Pad patch and mask to image dimensions with zeros
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)  # dim 6 x 14 x 3 x 416 x 416
        msk_batch = mypad(msk_batch)  # dim 6 x 14 x 3 x 416 x 416

        # NB you see only zeros when you print it because they are all surrounding the patch to pad it to image dimensions (3 x 416 x 416)

#_______________________________________________________________________________________________________________________________________________
        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))  # dim = 6*14 = 84
        if do_rotate:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            else:
                angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)

        else:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)
            else:
                angle = torch.FloatTensor(anglesize).fill_(0)
#_______________________________________________________________________________________________________________________________________________
        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)  # 300

        if use_cuda:
            lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        else:
            lab_batch_scaled = torch.FloatTensor(lab_batch.size()).fill_(0)  # dim 6 x 14 x 5

        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size

        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # np.prod(batch_size) = 4*16 = 84
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # used to get off_x
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # used to get off_y

        if(rand_loc):
            if use_cuda:
                off_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))
                off_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))
            else:
                off_x = targetoff_x * (torch.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                off_y = targetoff_y * (torch.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))

            target_x = target_x + off_x
            target_y = target_y + off_y

        target_y = target_y - 0.05

        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size() # 6 x 14 x 3 x 416 x 416
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 16
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 16

        tx = (-target_x+0.5)*2
        ty = (-target_y+0.5)*2

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation, rescale matrix
        if use_cuda:
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        else:
            theta = torch.FloatTensor(anglesize, 2, 3).fill_(0) # dim 84 x 2 x 3 (N x 2 x 3) required by F.affine_grid

        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale

        grid = F.affine_grid(theta, adv_batch.shape)  # adv_batch should be of type N x C x Hin x Win. Output is N x Hg x Wg x 2

        adv_batch_t = F.grid_sample(adv_batch, grid)  # computes the output using input values and pixel locations from grid.
        msk_batch_t = F.grid_sample(msk_batch, grid)  # Output has dim N x C x Hg x Wg
        # print(adv_batch_t.size()) dim 84 x 3 x 416 x 416
        # print(msk_batch_t.size()) dim 84 x 3 x 416 x 416


        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2

        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)

        '''
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4]) # 4 x 16 x 3 x 416 x 416
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        #img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        #img = transforms.ToPILImage()(img)
        #img.show()
        #exit()

        # print((adv_batch_t * msk_batch_t).size()) dim = 6 x 14 x 3 x 416 x 416

        return adv_batch_t * msk_batch_t  # It is as if I have passed adv_batch_t "filtered" by the mask itself

####################################################################################################
####################################################################################################

class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):

        advs = torch.unbind(adv_batch, 1)  # Returns a tuple of all slices along a given dimension, already without it.
        # print(np.shape(advs)) # dim = (14,) --> it indicates TODO 14 copies of the adv patch: one for each detected person (random number)
        # plus the remaining to get a total = max_lab (i.e. 14)

        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)  # the output tensor has elements belonging to img_batch if adv == 0, else belonging to adv
            # dim img_batch = 6 x 3 x 416 x 416

            # you put one after the other your 14 adv_patches on the image. When you meet those which are totally 0, i.e. those that do not
            # correspond to a detected object in the image, you keep your image as it is (do nothing). Otherwise, you will have your scaled, rotated and
            # well-positioned patch corresponding to one of the detected objects of the image. I think its pixels are 0s where there is not the object, and =/= 0
            # where there is the object, with appropriate affine properties. Here, you substitute image pixels with adv pixels.
            # At the end of the 14th cycle you have attached your patches to all detected regions of the image 'layer by layer', for all images in the batch (6).
        return img_batch

class PatchApplierMask(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """
    def __init__(self,BackgroundStyle):
        super(PatchApplierMask, self).__init__()
        self.BackgroundStyle = BackgroundStyle

    def forward(self, img_batch, masks_batch, adv_patch, mask_attack = None):
        if self.BackgroundStyle == 3: 
            att_img_batch = img_batch*((1-masks_batch) + masks_batch*(1-mask_attack))+ adv_patch*masks_batch*mask_attack
            return att_img_batch
        else:
            att_img_batch = img_batch*(1-masks_batch) + adv_patch*masks_batch
            return att_img_batch