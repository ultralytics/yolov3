from torch import optim
from patch_functions import *
from loss_functions import *
from dataset_functions import *


class standard(object):
    """
    Using the loss max_prob_class
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.patch_name = 'standard'

        self.img_size = 640

        # Dataloader informations
        self.img_dir_test = '/home/andread98/yolov3/MyWork/data_mask_test'    # test images location
        self.mask_dir_test = '/home/andread98/yolov3/MyWork/data_mask_test/mask'  # test labels location
        self.batch_size_test = 1
        # self.n_iterations = 1
        self.BackgroundStyle = 0
        self.number_for_name = 12
        # self.list_of_shape = [3,0,0,0,0,0]

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
        self.rotation_mode = 1


class perlin_noise(standard):
    """
    Using the loss max_prob_class, perlin noise application
    """
    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'perlin_noise'
        self.BackgroundStyle = 1
        self.number_for_name = 16
        


class perlin_noise_inverted(standard):
    """
    Using the loss max_prob_class, inverted perlin noise application
    """
    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'perlin_noise_inverted'
        self.BackgroundStyle = 2
        self.number_for_name = 25

class ghost(standard):
    """
    Using the loss max_prob_class, ghost application
    """
    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'ghost'
        self.BackgroundStyle = 3
        self.number_for_name = 9

patch_configs = {
    "standard": standard,
    "perlin_noise": perlin_noise,
    "perlin_noise_inverted": perlin_noise_inverted,
    "ghost": ghost
}
