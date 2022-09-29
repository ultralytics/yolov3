"""
Training code for Adversarial patch training
"""

from tqdm import tqdm
import matplotlib.pyplot as plt

from patch_functions import *
from loss_functions import *
from dataset_functions import *
from tile_functions import Tile_Creator
import torch.optim as optim
import patch_config_mask as patch_config_mask
import math

from torch import autograd
from torchvision import transforms
import time
import os

# Transforming from PIL to Tensor
transform1 = transforms.ToTensor()

# Transforming from Tensor to PIL
transform2 = transforms.ToPILImage()

if __name__ == '__main__':

    class PatchTrainer(object):

        def __init__(self, mode, tile = None):

            # Select the confing file 
            self.config = patch_config_mask.patch_configs[mode]()  # select the mode for the patch

            # Backgroun mode
            #   -Used for the Fractal Creator
            #   -Used for the patch applier 

            # Device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Model
            self.model = torch.hub.load('ultralytics/yolov3', 'yolov3', autoshape = False)  # or yolov3-spp, yolov3-tiny, custom

            if use_cuda:
                self.model = self.model.eval().to(self.device)  
                self.patch_applier = PatchApplierMask(self.config.BackgroundStyle).to(self.device)
                # self.patch_transformer = PatchTransformer().to(self.device)
                self.loss_function = self.config.loss_function.to(self.device)

                # Which type of attack do we want to do?
                #------------------------------------------------------------------------------------------------
                # # Classes for all the possible types of tiles
                # self.tile_class = self.config.list_classes_tile[tile]()

                # One class fits all
                self.tile_class = Tile_Creator(self.config.list_of_shape)
                #------------------------------------------------------------------------------------------------

                # self.params_function = self.config.list_function_params[tile]
                self.gen_function = Fractal_Patch_Generator(self.config.dim_tile, self.config.dim_patch, self.config.mul_fact,self.tile_class,self.config.rotation_mode,self.config.BackgroundStyle,self.config.mask_function).to(self.device)

            else:
                self.model = self.model.eval()  # TODO: Why eval?
                self.patch_applier = PatchApplierMask(self.config.BackgroundStyle)
                # self.patch_transformer = PatchTransformer()
                self.loss_function = self.config.loss_function

                # Which type of attack do we want to do?
                #------------------------------------------------------------------------------------------------
                # # Classes for all the possible types of tiles
                # self.tile_class = self.config.list_classes_tile[tile]()

                # One class fits all
                self.tile_class = Tile_Creator(self.config.list_of_shape)
                #------------------------------------------------------------------------------------------------

                # self.params_function = self.config.list_function_params[tile]
                self.gen_function = Fractal_Patch_Generator(self.config.dim_tile, self.config.dim_patch, self.config.mul_fact,self.tile_class,self.config.rotation_mode,self.config.BackgroundStyle,self.config.mask_function)

        def train(self, name):

            """
            Optimize a patch to generate an adversarial example.
            :return: Nothing
            """

            # name = str(tile) + '_' + mode 
            name = name + '_' + mode 
            destination_path = "./yolov3/MyWork/txt_results/25-09-2022/"

            destination_name = name + '_iteration.txt'
            destination_name2 = name + '_batch.txt'
            destination = os.path.join(destination_path, destination_name)
            destination2 = os.path.join(destination_path, destination_name2)

            textfile = open(destination, 'w+')
            textfile2 = open(destination2, 'w+')

            params = self.tile_class.Params_Creator()
            print(params)

            train_loader = torch.utils.data.DataLoader(
                VOCmask(self.config.img_dir, self.config.mask_dir, self.config.img_size,
                             shuffle=True),
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=10)
            self.iteration_length = len(train_loader)
            print(f'One iteration is {len(train_loader)}')

            learning_rate = 0.03 # Mettere questo nel configuration file

            optimizer = optim.SGD(params, learning_rate)  # starting lr = 0.03
            scheduler = self.config.scheduler_factory(optimizer)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50) # write it directly

            n_iterations = self.config.n_iterations

            et0 = time.time()  # iteration start
            for iteration in range(n_iterations):
                ep_loss = 0
                bt0 = time.time()  # batch start
                for i_batch, (img_batch, masks_batch) in tqdm(enumerate(train_loader), desc=f'Running iteration {iteration}',
                                                            total=self.iteration_length):
                    self.gen_function.populate(params)
                    adv_patch, mask_attack = self.gen_function.application()

                    if use_cuda:
                        img_batch = img_batch.to(self.device)
                        masks_batch = masks_batch.to(self.device)
                        adv_patch = adv_patch.to(self.device)
                        mask_attack = mask_attack.to(self.device)
                    else:
                        img_batch = img_batch
                        masks_batch = masks_batch
                        adv_patch = adv_patch
                        mask_attack = mask_attack

                    # adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    # p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    # p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))
                    attacked_img_batch = self.patch_applier(img_batch, masks_batch, adv_patch, mask_attack)
                    attacked_img_batch = attacked_img_batch.type(torch.cuda.FloatTensor)

                    output = self.model(attacked_img_batch)  # TODO apply YOLOv2 to all (patched) images in the batch (6)
                    loss = torch.mean(self.loss_function(output))
                    # print(loss)
                    ep_loss += loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    params = self.tile_class.Params_Clamp(params)

                    bt1 = time.time()  # batch end
                    if i_batch % 1 == 0:
                        print('  BATCH NR: ', i_batch)
                        print('BATCH LOSS: ', loss)
                        print('PARAMETERS: ', params)
                        print('BATCH TIME: ', bt1 - bt0)

                        # textfile2.write(f'{i_batch} {loss} {det_loss} {nps_loss} {tv_loss}\n')
                        textfile2.write(f'{i_batch} {loss} \n')

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del output, attacked_img_batch, loss, adv_patch, mask_attack

                        if use_cuda:
                            torch.cuda.empty_cache()

                    bt0 = time.time()

                et1 = time.time()  # iteration end

                ep_loss = ep_loss / len(train_loader)

                scheduler.step(ep_loss)

                if True:
                    print('  iteration NR: ', iteration),
                    print('iteration LOSS: ', ep_loss)
                    print('iteration TIME: ', et1 - et0)

                    textfile.write(f'\ni_iteration: {iteration}\ne_total_loss:{ep_loss}\n\n')

                    # Save a sample image from the last iteration
                    # if iteration%(n_iterations-1) == 0:
                    if iteration%10 == 0:
                    #if iteration == 0:
                        att_image = attacked_img_batch[0].squeeze(0)
                        att_image_PIL = transform2(att_image)
                        att_image_PIL.save("./yolov3/MyWork/SampleImages/25-09-2022/" + name + '_iteration_' + str(iteration) + '.png')


                    torch.save(params, './yolov3/MyWork/params_results/25-09-2022/' + name + '.pt')

                    del output, attacked_img_batch, loss

                    if use_cuda:
                        torch.cuda.empty_cache()

                et0 = time.time()

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

    # for mode in modes:
    #     for tile in range(1,6):
    #         trainer = PatchTrainer(mode,tile)
    #         trainer.train()

    # # In this way we are training all the possible combination of loss and tile 
    
    # mode = 'max_prob_class'
    # tile = 1
    # trainer = PatchTrainer(mode,tile)
    # trainer.train()

    modes = ['ghost']

    for mode in modes:
        trainer = PatchTrainer(mode)
        name = '3_circles'
        trainer.train(name)