import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import csv
import tqdm
import pickle
from sklearn.model_selection import train_test_split
import itertools
import glob
import random
import os
random.seed(1)

class XviewDataset():
    def __init__(self,grouped_classes_,labels_,coords_,chips_,classes_):
        self.grouped_classes = grouped_classes_
        self.labels = labels_
        self.chips = chips_        
        self.coords = coords_
        self.classes = classes_

        
    def getLabelCounts(self):
        """ 
        given a chips and classes, returns dataframe of 
        new label groupings counts
        """
        chip_names = np.unique(self.chips) 
        results = np.zeros((len(chip_names),len(self.grouped_classes)))
        chip_strs = []
        for c_idx, c in tqdm.tqdm(enumerate(chip_names)):
            chip_strs.append(c)
            classes_chip = self.classes[self.chips==c]
            idx_filter = np.isin(classes_chip,self.grouped_classes[0])
            # initialize to all false
            for i,gc in (enumerate(self.grouped_classes)):
                is_in_idxs = np.isin(classes_chip,gc)
                classes_chip[is_in_idxs] = i
                idx_filter = np.logical_or(idx_filter,is_in_idxs)
            classes_chip = classes_chip[idx_filter]
            labels, counts = np.unique(classes_chip,return_counts=True)
            for label_idx,label in enumerate(labels):
                results[int(c_idx),int(label)] = counts[label_idx]
                pass
        chip_strs_col = np.array(chip_strs).reshape(-1,1)
        return (np.hstack((chip_strs_col,results)))

    def indToTifName(self,data, inds):
        res = []
        for ind in inds:
            res.append(data[ind][0])
        return res

    def getDistribution(self,data, selected_indexes):
        res = []
        total = 0
        class_num = len(data[0])
        for i in range(class_num):
            for index in selected_indexes:
                total += float(data[index][i])
        for i in range(class_num):
            total_of_this_class = 0
            for index in selected_indexes:
                total_of_this_class += float(data[index][i])
            res.append(float(total_of_this_class)/total)
        return res

    def checkThreshold(self,distr1, distr2, thres):
        if (len(distr1) != len(distr2)):
            print("columns' numbers don't fit.")
            return -1
        for i in range(len(distr1)):
            diff = abs(distr1[i] - distr2[i])
            if diff > thres:
                return False
        return True

    def findBalance(self,data, train_percent, thres):
        tifs = len(data)
        class_num = len(data[0])
        for i in range(1000000):
            tr_set, te_set = train_test_split(np.array(list(range(len(data)))),\
                                              test_size=1-train_percent)
            tr_d = self.getDistribution(data, tr_set)
            te_d = self.getDistribution(data, te_set)
            check = self.checkThreshold(tr_d, te_d, thres)
            if (check == -1):
                return -1
            elif (check == True):
                return tr_set, te_set
        return [], []

    def splitTrainValidTest(self,all_chips,all_classes):
        # make the table where filename is rowwise, columns are class
        s_chips = self.chips
        s_classes = self.classes
        # Count number of classes
        results = self.getLabelCounts()
        # Split into train and test        
        train_ind, test_ind = self.findBalance(results[:,1:], 0.8, 0.5)
        train_tifs = self.indToTifName(results,train_ind)
        test_tifs = self.indToTifName(results, test_ind)

        # Split train into train and validation
        train_tifs = self.indToTifName(results, train_ind)
        train_mask = np.isin(s_chips,train_tifs)
        # Find split for training and validation
        train_chips = s_chips[train_mask]
        train_classes = s_classes[train_mask]
        train_results = self.getLabelCounts()
        train_ind, valid_ind = self.findBalance(train_results[:,1:], 0.7, 0.02)
        # export new train and valid tif labels
        train_tifs = self.indToTifName(train_results, train_ind)
        valid_tifs = self.indToTifName(train_results, valid_ind)
        return (train_tifs,valid_tifs, test_tifs)


    def showClassExample(self,
                         image_path = "/data/zjc4/train_images/",
                         chip_name=None,
                         class_idx=1):
        s_class=np.array(self.grouped_classes[class_idx])
        mask = (np.isin(self.classes,s_class))
        schips = self.chips[mask]
        if chip_name == None:
            chip_name = (np.random.choice(schips))

        arr = wv.get_image(image_path+chip_name)
        coords = self.coords[(self.chips==chip_name) & (mask)]
        classes = self.classes[(self.chips==chip_name) & (mask)].astype(np.int64)
        c_img, c_box, c_cls = wv.chip_image(img = arr, coords= coords,
                                            classes=classes, shape=(800,800))
        images = []
        for ind in range(c_img.shape[0]):
            if len(c_cls[ind]) > 2:
                labelled = aug.draw_bboxes(c_img[ind],c_box[ind])
                plt.figure(figsize=(10,10))
                plt.axis('off')
                plt.imshow(labelled)
                images.append(labelled)
                pass
            pass
        return images
    
class DarkNetFormatter():
    def __init__(self,output_dir_,input_dir_,coords_,chips_,classes_,grouped_classes_):
        self.output_dir = output_dir_
        self.input_dir = input_dir_
        self.chips = chips_        
        self.coords = coords_
        self.classes = classes_
        self.grouped_classes = grouped_classes_
        pass
    
    def filterClasses(self,chip_coords,chip_classes,grouped_classes):
        filtered_classes = list(itertools.chain.from_iterable(self.grouped_classes))
        mask = (np.isin(chip_classes,filtered_classes))
        chip_coords, chip_classes = chip_coords[mask], chip_classes[mask]

        for idx, g_cls in enumerate(self.grouped_classes):
            mask = (np.isin(chip_classes,g_cls))
            chip_classes[mask] = idx
        return chip_coords,chip_classes
        pass

    def plotDarknetFmt(self,c_img,x_center,y_center,ws,hs,c_cls,szx,szy):
        fig,ax = plt.subplots(1,figsize=(10,10))
        ax.imshow(c_img)
        for didx in range(c_cls.shape[0]):
            x,y = x_center[didx]*szx,y_center[didx]*szy
            w,h = ws[didx]*szx,hs[didx]*szy
            x1,y1 = x-(w/2), y-(h/2)
            w1,h1 = w,h
            rect = patches.Rectangle((x1,y1),w1,h1,\
                                     linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            pass
        plt.show()
        pass

    def toDarknetFmt(self,c_box,c_cls,c_img,debug=False):
        szx,szy,_ = c_img.shape
        c_box[:,0],c_box[:,2] = c_box[:,0]/szx,c_box[:,2]/szx
        c_box[:,1],c_box[:,3] = c_box[:,1]/szy,c_box[:,3]/szy
        xmin,ymin,xmax,ymax = c_box[:,0],c_box[:,1],c_box[:,2],c_box[:,3]
        ws,hs = (xmax-xmin), (ymax-ymin)
        x_center, y_center = xmin+(ws/2),ymin+(hs/2)
        # Visualize using mpl
        if debug:
            plotDarknetFmt(c_img,x_center,y_center,ws,hs,c_cls,szx,szy)
        result = np.vstack((c_cls,x_center,y_center,ws,hs))
        return result.T

    def checkDir(self,filepath):
        """ passed a filepath string, checks if it dne
        if it does not exists makes directory"""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        

    def parseChip(self,c_img, c_box, c_cls,img_num,c_dir):
        # Parses chips, saves chip image, and also saves corresponding labels
        fnames = []
        
        outputImgDir = "{}labels/".format(c_dir)
        outputLabelDir = "{}images/".format(c_dir)
        self.checkDir(outputImgDir)
        self.checkDir(outputLabelDir)
        
        for c_idx in range(c_img.shape[0]):
            c_name = "{:06}_{:02}".format(int(img_num), c_idx)
            sbox,scls,simg = \
                c_box[c_idx],c_cls[c_idx],c_img[c_idx]
            # Change chip into darknet format, and save
            result = self.toDarknetFmt(sbox,scls,simg)
            ff_l = "{}labels/{}.txt".format(c_dir,c_name)
            np.savetxt(ff_l, result, fmt='%i %1.6f %1.6f %1.6f %1.6f')
            # Save image to specified dir
            ff_i = "{}images/{}.jpg".format(c_dir,c_name)

            Image.fromarray(simg).save(ff_i)
            # Append file name to list
            fnames.append("{}images/{}.jpg".format(c_dir,c_name))
            pass

        return fnames

    def exportChipImages(self,image_paths,c_dir,set_str):
        fnames = []
        print(set_str)
        for img_pth in image_paths:
            try:
                img_pth = self.input_dir+'train_images/'+img_pth
                img_name = img_pth.split("/")[-1]
                img_num = img_name.split(".")[0]
                arr = wv.get_image(img_pth)
                chip_coords = self.coords[self.chips==img_name]
                chip_classes = self.classes[self.chips==img_name].astype(np.int64)
                chip_coords,chip_classes = \
                    self.filterClasses(chip_coords,chip_classes,self.grouped_classes)

                c_img, c_box, c_cls = wv.chip_image(img=arr, coords=chip_coords, 
                                                    classes=chip_classes, shape=(600,600))

                c_fnames = self.parseChip(c_img, c_box, c_cls, img_num, c_dir)
                fnames.extend(c_fnames)
            except FileNotFoundError as e:
                print(e)
                pass
            pass

        lines = sorted(fnames)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            pass

        outputTxtPath = self.output_dir+"xview_img_{}.txt".format(set_str)

        if os.path.exists(outputTxtPath):
            os.remove(outputTxtPath)
            pass
        print(outputTxtPath)
        with open(outputTxtPath, mode='w', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))
        pass

    def transformDatum(self,datum):
        """
        takes in as input list of tuples corresponding to
        first string of subset, and training file indexes
        ("train", [training file idxs])
        """
        for (data_str, data_files) in datum:
            self.exportChipImages(data_files,self.output_dir,data_str)            
            pass
        pass
#if __name__ == "main":
# Load dataset
coords1, chips1, classes1 = wv.get_labels('/data/zjc4//xView_train.geojson')
# Input desired classes
grouped_classes = [[77,73],[11,12],[13],[17,18,20,21],
       [19,23,24,25,28,29,60,61,65,26],[41,42,50,40,44,45,47,49]]
labels = ["building and facility" ,"small aircraft", 
          "large aircraft","vehicles","bus","boat"]
#print(labels[0])
#showClassExample(grouped_classes[0],chip_name = "1694.tif")
print(len(chips1))
idxs = range(0,300000)
xdataset = XviewDataset(grouped_classes,labels, coords1[idxs],chips1[idxs],classes1[idxs])
data_sets = xdataset.splitTrainValidTest(chips1[idxs],classes1[idxs])
string_sets = ["train","valid","test"]
dnf = DarkNetFormatter(output_dir_ = "/data/zjc4/chipped/data/",
                       input_dir_="/data/zjc4/",
                       coords_ = coords1[idxs],
                       chips_ = chips1[idxs],
                       classes_ = classes1[idxs],
                       grouped_classes_=grouped_classes)
datum = list(zip(string_sets,data_sets))
dnf.transformDatum(datum)
