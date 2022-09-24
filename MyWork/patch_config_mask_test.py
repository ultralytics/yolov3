from torch import optim
from patch_functions import *
from loss_functions import *
from dataset_functions import *


class test(object):
    """
    Using the loss max_prob_class
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.patch_name = 'max_prob_class'

        self.img_size = 640

        # Dataloader informations
        self.img_dir_test = '/home/andread98/yolov3/MyWork/data_mask_test'    # test images location
        self.mask_dir_test = '/home/andread98/yolov3/MyWork/data_mask_test/mask'  # test labels location
        self.batch_size_test = 1
        self.n_iterations = 1

        # Loss function 
        self.loss_function = max_prob_class(0)

        # Patch functions
        # self.list_classes_tile = [Tile_Creator_Circle, Tile_Creator_Ellipse, Tile_Creator_Square, Tile_Creator_Rectangle, Tile_Creator_Triangle, Tile_Creator_Trapezoid]
        self.list_classes_tile = [Tile_Creator_Circle, Tile_Creator_Ellipse, Tile_Creator_Square, Tile_Creator_Rectangle, Tile_Creator_Triangle, Tile_Creator_Trapezoid, Tile_Creator_Double_Circle, Tile_Creator_Double_Ellipse, Tile_Creator_Double_Square, Tile_Creator_Double_Rectangle, Tile_Creator_Double_Triangle, Tile_Creator_Double_Trapezoid]
        # self.list_function_params = [Params_Creator_Circle, Params_Creator_Ellipse, Params_Creator_Square, Params_Creator_Rectangle, Params_Creator_Triangle, Params_Creator_Trapezoid]
        self.mask_function = Mask_Creator

        self.dim_tile = 16
        self.dim_patch = 640
        self.mul_fact = 4 # multiplying factor, how big can the tile be wrt the dim_tile 

        # 0: no rotation of the tile
        # 1: rotation of a multiple of 90
        # 2: random rotation. Attention, then the radius has to be at most sart(2)/2
        self.rotation_mode = 2


patch_configs = {
    "test": test
}