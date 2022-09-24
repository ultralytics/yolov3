"""
Training code for Adversarial patch training
"""

from tqdm import tqdm
import matplotlib.pyplot as plt

from patch_functions import *
from loss_functions import *
from dataset_functions import *
import torch.optim as optim
import patch_config as patch_config
import math

from torch import autograd
from torchvision import transforms
import time
import os

if __name__ == '__main__':

    class PatchTrainer(object):

        def __init__(self, mode, tile):

            # Select the confing file 
            self.config = patch_config.patch_configs[mode]()  # select the mode for the patch

            # Device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Model
            self.model = torch.hub.load('ultralytics/yolov3', 'yolov3', autoshape = False)  # or yolov3-spp, yolov3-tiny, custom

            if use_cuda:
                self.model = self.model.eval().to(self.device)  
                self.patch_applier = PatchApplier().to(self.device)
                self.patch_transformer = PatchTransformer().to(self.device)
                self.loss_function = self.config.loss_function.to(self.device)
                self.tile_function = self.config.list_function_tile[tile]
                self.params_function = self.config.list_function_params[tile]
                self.gen_function = Fractal_Patch_Generator(self.config.dim_tile, self.config.dim_patch, self.config.mul_fact,self.tile_function,self.config.mask_function,self.config.rotation_mode).to(self.device)

            else:
                self.model = self.model  # TODO: Why eval?
                self.patch_applier = PatchApplier()
                self.patch_transformer = PatchTransformer()
                self.loss_function = self.config.loss_function
                self.tile_function = self.config.list_function_tile[tile]
                self.params_function = self.config.list_function_params[tile]
                self.gen_function = Fractal_Patch_Generator(self.config.dim_tile, self.config.dim_patch, self.config.mul_fact,self.tile_function,self.config.mask_function,self.config.rotation_mode)

        def train(self):

            """
            Optimize a patch to generate an adversarial example.
            :return: Nothing
            """

            img_size = self.config.img_size  # 640 for this yolov3
            factor = math.sqrt(2)/2

            name = mode + '_' + str(tile)
            destination_path = "./yolov3/txt_results/"
            destination_name = name + '_batch.txt'
            destination_name2 = name + '_epoch.txt'
            # destination_name3 = 'loss_tracking_compatc_epochs_yv3_ultra_obj_noepoch.txt'
            destination = os.path.join(destination_path, destination_name)
            destination2 = os.path.join(destination_path, destination_name2)
            # destination3 = os.path.join(destination_path, destination_name3)
            textfile = open(destination, 'w+')
            textfile2 = open(destination2, 'w+')
            # textfile3 = open(destination3, 'w+')

            params = self.params_function()

            train_loader = torch.utils.data.DataLoader(
                InriaDataset(self.config.img_dir, self.config.lab_dir, self.config.max_lab, self.config.img_size,
                             shuffle=True),
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=10)
            self.epoch_length = len(train_loader)
            print(f'One epoch is {len(train_loader)}')

            learning_rate = 0.03 # Mettere questo nel configuration file

            optimizer = optim.SGD(params, learning_rate)  # starting lr = 0.03
            scheduler = self.config.scheduler_factory(optimizer)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50) # write it directly

            n_epochs = self.config.n_epochs

            et0 = time.time()  # epoch start
            for epoch in range(n_epochs):
                ep_loss = 0
                bt0 = time.time()  # batch start
                for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                            total=self.epoch_length):
                    self.gen_function.populate(params)
                    adv_patch = self.gen_function.application()

                    if use_cuda:
                        img_batch = img_batch.to(self.device)
                        lab_batch = lab_batch.to(self.device)
                        adv_patch = adv_patch.to(self.device)
                    else:
                        img_batch = img_batch
                        lab_batch = lab_batch
                        adv_patch = adv_patch

                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))
                    
                    output = self.model(p_img_batch)  # TODO apply YOLOv2 to all (patched) images in the batch (6)
                    loss = torch.mean(self.loss_function(output))
                    print(loss)
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    
                    if len(params)==3:
                        params[0].data.clamp_(0, factor)
                        params[1].data.clamp_(0, 1)
                        params[2].data.clamp_(0, 1)  # keep patch in image range
                        print(params)
                    else:
                        params[0].data.clamp_(0, factor)
                        params[1].data.clamp_(0, factor)
                        params[2].data.clamp_(0, 1)
                        params[3].data.clamp_(0, 1)  # keep patch in image range
                        print(params)

                    bt1 = time.time()  # batch end
                    if i_batch % 1 == 0:
                        print('  BATCH NR: ', i_batch)
                        print('BATCH LOSS: ', loss)
                        print('BATCH TIME: ', bt1 - bt0)

                        # textfile2.write(f'{i_batch} {loss} {det_loss} {nps_loss} {tv_loss}\n')
                        textfile2.write(f'{i_batch} {loss} \n')

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, p_img_batch, loss

                        if use_cuda:
                            torch.cuda.empty_cache()

                    bt0 = time.time()

                et1 = time.time()  # epoch end

                ep_loss = ep_loss / len(train_loader)

                scheduler.step(ep_loss)

                if True:
                    print('  EPOCH NR: ', epoch),
                    print('EPOCH LOSS: ', ep_loss)
                    print('EPOCH TIME: ', et1 - et0)

                    textfile.write(f'\ni_epoch: {epoch}\ne_total_loss:{ep_loss}\n\n')
                    # textfile3.write(f'{ep_loss}\n')

                    # Plot the final adv_patch (learned) and save it
                    # im = transforms.ToPILImage('RGB')(adv_patch)
                    # plt.imshow(im)
                    # plt.show()
                    # im.save("./yolov3_ultralytics_obj.png")

                    torch.save(params, './yolov3/params_results/' + name + '.pt')

                    del adv_batch_t, output, p_img_batch, loss

                    if use_cuda:
                        torch.cuda.empty_cache()

                et0 = time.time()

    use_cuda = 1
    modes = ["max_prob_class","max_prob_class2","max_prob_obj","new_loss_tprob"]
    # Tile options
    # 0: Circle 
    # 1: Ellipse
    # 2: Square 
    # 3: Rectangle
    # 4: Traingle 
    # 5: Trapezoid

    for mode in modes:
        for tile in range(6):
            trainer = PatchTrainer(mode,tile)
            trainer.train()

    # In this way we are training all the possible combination of loss and tile 


