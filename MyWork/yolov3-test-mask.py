# Caricare il modello test, senza autoshape = False

# Usare 20 immagini come test set
# Creare il dataloader con batch = 1 
# il modello test non prende come input dei tensori

# Caricare ogni tensore trovato dopo l'ottimizzazione
# Per ognuno calcolare la media del massimo di ogni foto

# Create a txt and every line is '2_max_prob_class.pt accuracy'

"""
Testing code for Adversarial patch
"""

from tqdm import tqdm
import matplotlib.pyplot as plt

from patch_functions import *
from loss_functions import *
from dataset_functions import *
import torch.optim as optim
import patch_config_mask_test as patch_config_mask_test
from tile_functions import Tile_Creator
from count_people import count_people
import math

from torch import autograd
from torchvision import transforms
import time
import os
import cv2

def classifier_attack(attack_path, dimension, multiplier):
    # Open the attack
    attack_PIL = Image.open(attack_path)
    # attack_PIL.show()

    # Resize the image
    new_image_PIL = attack_PIL.resize((dimension, dimension))
    # new_image.show()

    # Transform the PIL image to a numpy array
    new_image_np = np.asarray(new_image_PIL)

    # Tile the image
    tiling_np = np.tile(new_image_np, (multiplier, multiplier,1))
    # Numpy to PIL
    tiling_PIL = Image.fromarray(np.uint8(tiling_np))
    # PIL to tensor
    tiling_torch = transform1(tiling_PIL)
    
    return tiling_torch.unsqueeze(0)

configurations = [
    [2,0,0,0],
    [0,2,0,0],
    [0,0,2,0],
    [0,0,0,2],

    [1,1,0,0],
    [0,1,1,0],
    [0,0,1,1],
    [1,0,0,1],
    [0,1,0,1],
    [1,0,1,0],

    [3,0,0,0],
    [0,3,0,0],
    [0,0,3,0],
    [0,0,0,3],

    [2,1,0,0],
    [2,0,1,0],
    [2,0,0,1],

    [1,2,0,0],
    [0,2,1,0],
    [0,2,0,1],

    [1,0,2,0],
    [0,1,2,0],
    [0,0,2,1],

    [1,0,0,2],
    [0,1,0,2],
    [0,0,1,2]

]

images_path = '/home/andread98/yolov3/MyWork/data_mask_test/prediction_attack/'

# Transforming from PIL to Tensor
transform1 = transforms.ToTensor()

# Transforming from Tensor to PIL
transform2 = transforms.ToPILImage()

if __name__ == '__main__':

    class PatchTester(object):

        def __init__(self, flag, mode = None):

            self.flag = flag

            # Select the confing file 
            self.config = patch_config_mask_test.patch_configs['standard']()  # select the mode for the patch

            # Device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Model
            self.model_test = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or yolov3-spp, yolov3-tiny, custom

            # print(self.config.BackgroundStyle)
            if use_cuda:
                self.model_test = self.model_test.eval().to(self.device)  
                self.patch_applier = PatchApplierMask(self.config.BackgroundStyle).to(self.device)

            else:
                self.model_test = self.model_test.eval()
                self.patch_applier = PatchApplierMask(self.config.BackgroundStyle)
            
        def test(self, date, IoU_thresh, Confidence_thresh):
            """
            Optimize a patch to generate an adversarial example.
            :return: Nothing
            """
            txt_result_path = '/home/andread98/yolov3/MyWork/test_results/' + date + '/'+ 'IoU_thresh_' + str(IoU_thresh) + '_Confidence_thresh_' + str(Confidence_thresh) + '.txt'
            with open(txt_result_path, 'a') as f:
                test_loader = torch.utils.data.DataLoader(
                            VOCmask(self.config.img_dir_test, self.config.mask_dir_test, self.config.img_size,
                                        shuffle=True),
                            batch_size=self.config.batch_size_test,
                            shuffle=True,
                            num_workers=10)
                self.iteration_length = len(test_loader)
                
                 
            #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
                BB_path = '/home/andread98/yolov3/MyWork/BB-results'
                # if self.flag:
                #     # modes = ['baseline', 'white', 'black', 'random']
                #     modes = ['baseline', 'random']
                #     for mode in modes:
                #         print(mode)
                #         directory = BB_path + '/' + mode
                #         if not os.path.exists(directory):
                #             os.makedirs(directory)
                #         for i_batch, (img_batch, masks_batch, img_name) in enumerate(test_loader):
                #             print(i_batch)
                #             # print(f'Shape image batch: {img_batch.shape}')
                #             # print(f'Shape masks batch: {masks_batch.shape}')

                #             destination = directory + '/' + str(i_batch) + '.png'

                #             if mode == 'baseline':
                #                 img_PIL = transform2(img_batch.squeeze(0))
                #             else:
                #                 if mode == 'white':
                #                     adv_patch = torch.ones((1,3,640,640))
                #                 elif mode == 'black':
                #                     adv_patch = torch.zeros((1,3,640,640))
                #                 else:
                #                     adv_patch = torch.rand((1,3,640,640))
                #                 # Apply the attack
                #                 attacked_img_batch = self.patch_applier(img_batch, masks_batch, adv_patch)
                #                 # Transform the image tensor in image PIL 
                #                 img_PIL = transform2(attacked_img_batch.squeeze(0))

                #             # Get the output for each image
                #             output = self.model_test(img_PIL)

                #             # save the BB image 
                #             array = np.squeeze(output.render())
                #             PIL_image = Image.fromarray(np.uint8(array)).convert('RGB')
                #             PIL_image.save(destination)

                #             array = output.xywhn[0].cpu().numpy()
                #             # Select only the object that are people
                #             array = array[array[:,-1] == 0]

                #             # Define the path were the output has to be saved in 
                #             final_path = images_path + img_name[0][:-3] + 'txt'

                #             with open(final_path, "w") as txt_file:
                #                 for line in array:
                #                     txt_file.write(" ".join(str(v) for v in line) + "\n") # works with any number of elements in a line

                #         # Count the people comparing the ground truth and the prediction of the attack
                #         number_people = count_people(IoU_thresh, Confidence_thresh)
                #         line = mode + ','  + str(number_people) + '\n'
                #         f.write(line)

                        
                
            #     # #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

            #         path_attacks = '/home/andread98/yolov3/MyWork/classifier_attack'
            #         attacks = [f for f in os.listdir(path_attacks) if f.endswith('.png')]
            #         attacks = sorted(attacks)

            #         for attack in attacks:
            #             dimensions = [40, 64, 80, 128, 160, 320]

            #             for dimension in dimensions:
            #                 attack_path = path_attacks + '/' + attack
            #                 multiplier = int(640/dimension)
                            
            #                 # Obtain the adv_patch
            #                 adv_patch =classifier_attack(attack_path, dimension, multiplier)

            #                 for i_batch, (img_batch, masks_batch, img_name) in enumerate(test_loader):
            #                     print(i_batch)
                                
            #                     # Apply the attack
            #                     attacked_img_batch = self.patch_applier(img_batch, masks_batch, adv_patch)
            #                     # Transform the image tensor in image PIL 
            #                     img_PIL = transform2(attacked_img_batch.squeeze(0))
            #                     # img_PIL.show()

            #                     # Get the output for each image
            #                     output = self.model_test(img_PIL)
            #                     array = output.xywhn[0].cpu().numpy()
            #                     # Select only the object that are people
            #                     array = array[array[:,-1] == 0]

            #                     # Define the path were the output has to be saved in 
            #                     final_path = images_path + img_name[0][:-3] + 'txt'

            #                     with open(final_path, "w") as txt_file:
            #                         for line in array:
            #                             txt_file.write(" ".join(str(v) for v in line) + "\n") # works with any number of elements in a line

            #                 # Count the people comparing the ground truth and the prediction of the attack
            #                 number_people = count_people(IoU_thresh, Confidence_thresh)
            #                 line = 'attack_' + attack[:-4] + '_' + str(dimension) + ',' + str(number_people) + '\n'
            #                 f.write(line)
                        
            #         self.flag = False

                ####################################################################################################
                
                # modes = ['standard', 'perlin_noise', 'perlin_noise_inverted', 'ghost']
                modes = ['ghost']

                for mode in modes:
                    # Select the confing file 
                    self.config = patch_config_mask_test.patch_configs[mode]()  # select the mode for the patch

                    print(self.config.BackgroundStyle)
                    if use_cuda:
                        self.patch_applier = PatchApplierMask(self.config.BackgroundStyle).to(self.device)

                    else:
                        self.patch_applier = PatchApplierMask(self.config.BackgroundStyle)

                    starting_path = '/home/andread98/yolov3/MyWork/params_results/' + date + '/' + mode 
                    tensors = [f for f in os.listdir(starting_path) if f.endswith('.pt')]
                    tensors = sorted(tensors)
                
                    for tensor in tensors:
                        print(mode)
                        directory = BB_path + '/' + tensor
                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        path = starting_path + '/' + tensor
                        params = torch.load(path)
                        print(params)
                        tile = int(tensor[:-self.config.number_for_name])
                        print('Tile: ', tile)
                        
                        # Which type of attack do we want to do?
                        #------------------------------------------------------------------------------------------------
                        if date == '22-09-2022':
                            # Classes for all the possible types of tiles
                            self.tile_class = self.config.list_classes_tile[tile]()
                        else: 
                            # One class fits all
                            self.tile_class = Tile_Creator(configurations[tile])
                        #------------------------------------------------------------------------------------------------

                        if use_cuda:
                            # self.params_function = self.config.list_function_params[tile]
                            self.gen_function = Fractal_Patch_Generator(self.config.dim_tile, self.config.dim_patch, self.config.mul_fact,self.tile_class,self.config.rotation_mode,self.config.BackgroundStyle,self.config.mask_function).to(self.device)
                        else:
                            # self.params_function = self.config.list_function_params[tile]
                            self.gen_function = Fractal_Patch_Generator(self.config.dim_tile, self.config.dim_patch, self.config.mul_fact,self.tile_class,self.config.rotation_mode,self.config.BackgroundStyle,self.config.mask_function)

                        print(f'rotation: {self.config.rotation_mode}')
                        print(f'self.config.BackgroundStyle:  {self.config.BackgroundStyle}')

                        for i_batch, (img_batch, masks_batch, img_name) in enumerate(test_loader):

                            destination = directory + '/' + str(i_batch) + '.png'

                            self.gen_function.populate(params)
                            adv_patch, mask_attack = self.gen_function.application()

                            torch.save(mask_attack, 'mask.pt')

                            if use_cuda:
                                img_batch = img_batch.to(self.device)
                                masks_batch = masks_batch.to(self.device)
                                adv_patch = adv_patch.to(self.device)
                                mask_attack = mask_attack.to(self.device)
                            else:
                                img_batch = img_batch
                                masks_batch = masks_batch
                                adv_patch = adv_patch

                            attacked_img_batch = self.patch_applier(img_batch, masks_batch, adv_patch, mask_attack)
                            attacked_img_batch = attacked_img_batch.type(torch.cuda.FloatTensor)
                            # attacked_img_batch = self.patch_applier(img_batch, masks_batch, adv_patch)
                            print(i_batch)
                            img_PIL = transform2(attacked_img_batch.squeeze(0))                        

                            output = self.model_test(img_PIL)  

                            # save the BB image 
                            array = np.squeeze(output.render())
                            PIL_image = Image.fromarray(np.uint8(array)).convert('RGB')
                            PIL_image.save(destination)

                            array = output.xywhn[0].cpu().numpy()
                            # Select only the object that are people
                            array = array[array[:,-1] == 0]

                            # Define the path were the output has to be saved in 
                            final_path = images_path + img_name[0][:-3] + 'txt'

                            with open(final_path, "w") as txt_file:
                                for line in array:
                                    txt_file.write(" ".join(str(v) for v in line) + "\n") # works with any number of elements in a line

                        # Count the people comparing the ground truth and the prediction of the attack
                        number_people = count_people(IoU_thresh, Confidence_thresh)
                        line = tensor + ',' + str(number_people) + '\n'
                        f.write(line)
                


    use_cuda = 1
    
    # Tile options
    # 0: Circle 
    # 1: Ellipse
    # 2: Square 
    # 3: Rectangle
    # 4: Triangle 
    # 5: Trapezoid
    # 6: Double Circle 
    # 7: Double Ellipse
    # 8: Double Square 
    # 9: Double Rectangle
    # 10: Double Triangle 
    # 11: Double Trapezoid

    # Which experiment do you want to evaluate?
    dates = ['20-11-2022']

    flag = True

    for date in dates:
        
        for i in range(5):
            IoU_thresh = 0.5
            Confidence_thresh = 0.4
            tester = PatchTester(flag)
            tester.test(date, IoU_thresh, Confidence_thresh)
            # print(img_PIL)
            # img_tensor = transform1(img_PIL)
            # torch.save(img_tensor, 'img_tensor.pt')
            # cv2.imshow('Color image', img_tensor.cpu().detach().numpy()*255)
            # cv2.waitKey(0)
            

 

    