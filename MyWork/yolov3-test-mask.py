
# Caricare il modello test, senza autoshape = False

# Usare 20 immagini come test set
# Creare il dataloader con batch = 1 
# il modello test non prende come input dei tensori

# Caricare ogni tensore trovato dopo l'ottimizzazione
# Per ognuno calcolare la media del massimo di ogni foto

# Create a txt and evry line is '2_max_prob_class.pt accuracy'

"""
Testing code for Adversarial patch training
"""

from tqdm import tqdm
import matplotlib.pyplot as plt

from patch_functions import *
from loss_functions import *
from dataset_functions import *
import torch.optim as optim
import patch_config_mask_test as patch_config_mask_test
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

    class PatchTester(object):

        def __init__(self, mode):

            # Select the confing file 
            self.config = patch_config_mask_test.patch_configs[mode]()  # select the mode for the patch

            # Device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Model
            self.model_test = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or yolov3-spp, yolov3-tiny, custom

            if use_cuda:
                self.model_test = self.model_test.eval().to(self.device)  
                self.patch_applier = PatchApplierMask().to(self.device)
                # self.loss_function = self.config.loss_function.to(self.device)

            else:
                self.model_test = self.model_test.eval()  # TODO: Why eval?
                self.patch_applier = PatchApplierMask()
                # self.loss_function = self.config.loss_function
            
        def test(self):
            """
            Optimize a patch to generate an adversarial example.
            :return: Nothing
            """
            with open('/home/andread98/yolov3/MyWork/test_results.txt', 'a') as f:
                test_loader = torch.utils.data.DataLoader(
                            VOCmask(self.config.img_dir_test, self.config.mask_dir_test, self.config.img_size,
                                        shuffle=True),
                            batch_size=self.config.batch_size_test,
                            shuffle=True,
                            num_workers=10)
                self.epoch_length = len(test_loader)

                baseline_loss = 0
                for i_batch, (img_batch, masks_batch) in enumerate(test_loader):
                            print(i_batch)
                            img_PIL = transform2(img_batch.squeeze(0))

                            output = self.model_test(img_PIL)  # TODO apply YOLOv2 to all (patched) images in the batch (6)
                            array = output.xywhn[0].cpu().numpy()
                            # Select only the object that are people
                            array = array[array[:,-1] == 0]
                            loss = max(array[:,-2], default=0)
                            # print(loss)
                            baseline_loss += loss
                
                baseline_loss = baseline_loss/self.epoch_length
                line = 'baseline_loss   ' + str(baseline_loss) + '\n'
                f.write(line)

                white_loss = 0
                for i_batch, (img_batch, masks_batch) in enumerate(test_loader):
                            adv_patch = torch.ones((1,3,640,640))
                            attacked_img_batch = self.patch_applier(img_batch, masks_batch, adv_patch)
                            print(i_batch)
                            img_PIL = transform2(attacked_img_batch.squeeze(0))

                            output = self.model_test(img_PIL)  # TODO apply YOLOv2 to all (patched) images in the batch (6)
                            array = output.xywhn[0].cpu().numpy()
                            # Select only the object that are people
                            array = array[array[:,-1] == 0]
                            loss = max(array[:,-2], default=0)
                            # print(loss)
                            white_loss += loss
                
                white_loss = white_loss/self.epoch_length
                line = 'white_loss   ' + str(white_loss) + '\n'
                f.write(line)

                black_loss = 0
                for i_batch, (img_batch, masks_batch) in enumerate(test_loader):
                            adv_patch = torch.zeros((1,3,640,640))
                            attacked_img_batch = self.patch_applier(img_batch, masks_batch, adv_patch)
                            print(i_batch)
                            img_PIL = transform2(attacked_img_batch.squeeze(0))


                            output = self.model_test(img_PIL)  # TODO apply YOLOv2 to all (patched) images in the batch (6)
                            array = output.xywhn[0].cpu().numpy()
                            # Select only the object that are people
                            array = array[array[:,-1] == 0]
                            loss = max(array[:,-2], default=0)
                            # print(loss)
                            black_loss += loss
                
                black_loss = black_loss/self.epoch_length
                line = 'black_loss   ' + str(black_loss) + '\n'
                f.write(line)

                random_loss = 0
                for i_batch, (img_batch, masks_batch) in enumerate(test_loader):
                            adv_patch = torch.rand((1,3,640,640))
                            attacked_img_batch = self.patch_applier(img_batch, masks_batch, adv_patch)
                            print(i_batch)
                            img_PIL = transform2(attacked_img_batch.squeeze(0))


                            output = self.model_test(img_PIL)  # TODO apply YOLOv2 to all (patched) images in the batch (6)
                            array = output.xywhn[0].cpu().numpy()
                            # Select only the object that are people
                            array = array[array[:,-1] == 0]
                            loss = max(array[:,-2], default=0)
                            # print(loss)
                            random_loss += loss
                
                random_loss = random_loss/self.epoch_length
                line = 'random_loss   ' + str(random_loss) + '\n'
                f.write(line)
                
                tensors = [f for f in os.listdir("/home/andread98/yolov3/MyWork/params_results") if f.endswith('.pt')]

            
                for tensor in tensors:
                    tensor_loss = 0
                    path = '/home/andread98/yolov3/MyWork/params_results/' + tensor
                    params = torch.load(path)
                    print(params)
                    tile = int(tensor[:-18])
                    print(tile)
                    
                    if use_cuda:
                        self.tile_class = self.config.list_classes_tile[tile]()
                        self.gen_function = Fractal_Patch_Generator(self.config.dim_tile, self.config.dim_patch, self.config.mul_fact,self.tile_class,self.config.mask_function,self.config.rotation_mode).to(self.device)

                    else:
                        self.tile_class = self.config.list_classes_tile[tile]()
                        self.gen_function = Fractal_Patch_Generator(self.config.dim_tile, self.config.dim_patch, self.config.mul_fact,self.tile_class,self.config.mask_function,self.config.rotation_mode)

                    for i_batch, (img_batch, masks_batch) in enumerate(test_loader):
                        self.gen_function.populate(params)
                        adv_patch = self.gen_function.application()

                        if use_cuda:
                            img_batch = img_batch.to(self.device)
                            masks_batch = masks_batch.to(self.device)
                            adv_patch = adv_patch.to(self.device)
                        else:
                            img_batch = img_batch
                            masks_batch = masks_batch
                            adv_patch = adv_patch

                        attacked_img_batch = self.patch_applier(img_batch, masks_batch, adv_patch)
                        print(i_batch)
                        img_PIL = transform2(attacked_img_batch.squeeze(0))

                        output = self.model_test(img_PIL)  # TODO apply YOLOv2 to all (patched) images in the batch (6)
                        array = output.xywhn[0].cpu().numpy()
                        # Select only the object that are people
                        array = array[array[:,-1] == 0]
                        loss = max(array[:,-2], default=0)
                        # print(loss)
                        tensor_loss += loss
                    
                    tensor_loss = tensor_loss/self.epoch_length
                    line = tensor[:-3] + '   ' + str(tensor_loss) + '\n'
                    f.write(line)
                


    use_cuda = 1
    modes = ["max_prob_class","max_prob_class2","max_prob_obj","new_loss_tprob"]
    # Tile options
    # 0: Circle 
    # 1: Ellipse
    # 2: Square 
    # 3: Rectangle
    # 4: Traingle 
    # 5: Trapezoid
    # 6: Double Circle 
    # 7: Double Ellipse
    # 8: Double Square 
    # 9: Double Rectangle
    # 10: Double Traingle 
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

    mode = 'test'
    tester = PatchTester(mode)
    tester.test()