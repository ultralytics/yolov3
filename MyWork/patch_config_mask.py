from torch import optim

from dataset_functions import *
from loss_functions import *
from patch_functions import *


class standard:
    """
    Using the loss max_prob_class, standard application
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.patch_name = 'standard'

        self.img_size = 640

        # Dataloader informations
        self.img_dir = '/home/andread98/yolov3/MyWork/data_mask'    # train images location
        self.mask_dir = '/home/andread98/yolov3/MyWork/data_mask/mask'  # train labels location
        self.batch_size = 20
        self.n_iterations = 40
        self.BackgroundStyle = 0

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

        # Optimizer informations
        self.start_learning_rate = 0.1
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        # reduce learning rate when a metric has stopped learning (keras??)
        # In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
        # in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.

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

patch_configs = {
    "standard": standard,
    "perlin_noise": perlin_noise,
    "perlin_noise_inverted": perlin_noise_inverted,
    "ghost": ghost
}

# class loss1(object):
#     """
#     Using the loss max_prob_class
#     """

#     def __init__(self):
#         """
#         Set the defaults.
#         """
#         self.patch_name = 'standard'

#         self.img_size = 640

#         # Dataloader informations
#         self.img_dir = '/home/andread98/yolov3/MyWork/data_mask'    # train images location
#         self.mask_dir = '/home/andread98/yolov3/MyWork/data_mask/mask'  # train labels location
#         self.batch_size = 20
#         self.n_iterations = 200

#         # Loss function
#         self.loss_function = max_prob_class(0)

#         # Patch functions
#         # self.list_classes_tile = [Tile_Creator_Circle, Tile_Creator_Ellipse, Tile_Creator_Square, Tile_Creator_Rectangle, Tile_Creator_Triangle, Tile_Creator_Trapezoid]
#         self.list_classes_tile = [Tile_Creator_Circle, Tile_Creator_Ellipse, Tile_Creator_Square, Tile_Creator_Rectangle, Tile_Creator_Triangle, Tile_Creator_Trapezoid, Tile_Creator_Double_Circle, Tile_Creator_Double_Ellipse, Tile_Creator_Double_Square, Tile_Creator_Double_Rectangle, Tile_Creator_Double_Triangle, Tile_Creator_Double_Trapezoid]
#         # self.list_function_params = [Params_Creator_Circle, Params_Creator_Ellipse, Params_Creator_Square, Params_Creator_Rectangle, Params_Creator_Triangle, Params_Creator_Trapezoid]
#         self.mask_function = Mask_Creator

#         self.dim_tile = 16
#         self.dim_patch = 640
#         self.mul_fact = 4 # multiplying factor, how big can the tile be wrt the dim_tile

#         # 0: no rotation of the tile
#         # 1: rotation of a multiple of 90
#         # 2: random rotation. Attention, then the radius has to be at most sart(2)/2
#         self.rotation_mode = 2

#         # Optimizer informations
#         self.start_learning_rate = 0.03
#         self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
#         # reduce learning rate when a metric has stopped learning (keras??)
#         # In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
#         # in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.

# class loss2(loss1):
#     """
#     Using the loss max_prob_class2
#     """
#     def __init__(self):
#         """
#         Change stuff...
#         """
#         super().__init__()

#         self.patch_name = 'perlin_noise'
#         self.loss_function = max_prob_class2(0)


# class loss3(loss1):
#     """
#     Using the loss max_prob_obj
#     """
#     def __init__(self):
#         """
#         Change stuff...
#         """
#         super().__init__()

#         self.patch_name = 'perlin_noise_inverted'
#         self.loss_function = max_prob_obj()

# class loss4(loss1):
#     """
#     Using the loss new_loss_tprob
#     """
#     def __init__(self):
#         """
#         Change stuff...
#         """
#         super().__init__()

#         self.patch_name = 'ghost'
#         self.loss_function = new_loss_tprob()


# patch_configs = {
#     "max_prob_class": loss1,
#     "max_prob_class2": loss2,
#     "max_prob_obj": loss3,
#     "new_loss_tprob": loss4
# }
