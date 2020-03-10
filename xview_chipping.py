"""
This class converts images from xView into chipped images
for input into Darknet format
"""
import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import tqdm

fdir = '/data/zjc4/'
#Load an image
chip_name = fdir+'train_images/104.tif'
cdir = "/data/zjc4/chipped/train/"
arr = wv.get_image(chip_name)

#Loading our labels
coords1, chips1, classes1 = wv.get_labels(fdir+'xView_train.geojson')
import glob
all_images = glob.glob(fdir+'train_images/*.tif')


#Load the class number -> class string label map
labels = {}
with open('xview_class_labels.txt') as f:
    for row in csv.reader(f):
        labels[int(row[0].split(":")[0])] = row[0].split(":")[1]
        pass
    pass

i = 0
for chip_name in tqdm.tqdm(all_images[:300]):
    chip_name = (chip_name.split("/")[-1])
    img_name = chip_name.split(".")[0]
    
    coords = coords1[chips1==chip_name]
    classes = classes1[chips1==chip_name].astype(np.int64)
    
    c_img, c_box, c_cls = wv.chip_image(img = arr, coords= coords, 
                                        classes=classes, shape=(600,600))
    # xmin,ymin,xmax,ymax = c_box
    #   0    1   2    3
    for c_idx in (range(c_img.shape[0])):
        # Save the chipped image
        c_name = "{:06}_{:02}".format(int(img_name), c_idx)
        # Save the chipped label
        widths = c_box[c_idx][:,2]-c_box[c_idx][:,0]
        x = c_box[c_idx][:,0]+(widths/2)
        
        heights = c_box[c_idx][:,3]-c_box[c_idx][:,1]
        y = c_box[c_idx][:,1]+(heights/2)
        
        szx = c_img[c_idx].shape[0]
        szy = c_img[c_idx].shape[1]
        nwidths,nheights = widths/szx, heights/szy
        nx,ny = x/szx, y/szy

        nheights = np.round(nheights,6)
        nwidths = np.round(nwidths,6)
        nx,ny = np.round(nx,6),np.round(ny,6)
        
        h_cls = c_cls[c_idx]
        
        classes = [[11,12],[13],[17,18,20,21],\
                   [19,23,24,25,28,29,60,61,65,26],[41,42,50,40,44,45,47,49]]
        matching_cls_idxs = np.copy(np.isin(h_cls,classes[0])) # init class idx
        for idx,c in enumerate(classes):
            idx_filter = np.isin(h_cls,c)
            h_cls[idx_filter] = idx
            matching_cls_idxs = np.copy(np.logical_or(matching_cls_idxs,idx_filter))

        data_labels = np.vstack((h_cls,nx,ny,nwidths,nheights)).T
        # Select only valid labels
        y_valid = np.logical_and(nx-nwidths > 0.1,nx+nwidths < 0.9)
        x_valid = np.logical_and(ny-nheights > 0.1,ny+nheights < 0.9)
        
        valid = np.logical_and(x_valid,np.logical_and(y_valid,matching_cls_idxs))
        
        data_labels = data_labels[valid,:]
        
        # Break if there are no valid labels(o.w. empty label == error)
        if (np.count_nonzero(valid==True) < 1 ): break
        ff_l = "{}labels/{}.txt".format(cdir,c_name)
        np.savetxt(ff_l, data_labels, fmt='%i %1.6f %1.6f %1.6f %1.6f')
        
        img = Image.fromarray(c_img[c_idx])
        ff_i = "{}images/{}.jpg".format(cdir,c_name)
        img.save(ff_i)

lines = glob.glob(cdir+"images/*")
with open(fdir+"chipped/xview_img.txt", mode='w', encoding='utf-8') as myfile:
    myfile.write('\n'.join(lines))
