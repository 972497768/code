import h5py
import os
from skimage.morphology import dilation
import numpy as np
from skimage import measure
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle
import random
import shutil
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from util_raw.cmap import cmaps, ClassColor, create_custom_cmap
###########################
#    Defining Functions   #
###########################
def proswarm(label, dilate=[(5,5)]):# ,(7,7)
    """
    Preprocessing labels
    """
    label_dilated = label
    for d in dilate:
        label_dilated = dilation(label_dilated, np.ones(d))
    return label_dilated

def crop_pic(image, label, bb, size=256):
    """Cropping images"""
    minr, minc, maxr, maxc = bb  # (Min Height, Min Width, Max Height, Max Width)
    br, bc = int((maxr + minr) / 2), int((maxc + minc) / 2)
    height, width = Sv_label.shape
    c_left = bc - int((bc - minc) / size) * size
    c_right = int((maxc - bc) / size) * size + bc
    c_top = br - int((br - minr) / size) * size
    c_bottom = int((maxr - br) / size) * size + br
    images, labels = [], []
    for i in range(c_top, c_bottom+1, size):
        for j in range(c_left, c_right+1, size):
            left = j - int(size / 2)
            right = j + int(size / 2)
            top = i - int(size / 2)
            bottom = i + int(size / 2)

            if top < 0:
                top = 0
                bottom = top + size
            if left < 0:
                left = 0
                right = left + size
            if bottom > height:
                bottom = height
                top = bottom - size
            if right > width:
                right = width
                left = right - size

            image_patch = image[:, top:bottom, left:right]
            label_patch = label[top:bottom, left:right]
            images.append(image_patch)
            labels.append(label_patch)
            

    return image_patch, label_patch

def find_files_with_suffix(folder_path, suffix):
    # Get the paths of all files in a folder using the os module
    all_files = os.listdir(folder_path)
    # Filter files ending with a specified suffix
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files

##############################
#     define variable       #
##############################
set_trainday = 21   # (Modifiable!) Former set_trainday days for training data
choose_everyday = 5
size = 256
stride = 180
choosedepth = [0, 300]
path = r'dataset_0_300' # (Modifiable!) Path to the h5 dataset obtained from the first processing
out_path = fr'crop_dataset_{choosedepth[0]}_{choosedepth[1]}_select_{size}_{stride}_random{choose_everyday}' # (Modifiable!) Output the path to the new cropped dataset

try:
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
except:
    print("Output folder path error")

png_path = os.path.join(out_path,'PNG')
if not os.path.isdir(png_path):
    os.mkdir(png_path)

# txt_path = os.path.join(out_path, 'make_crop_dataset_report.txt')  # txt in the output path
# txt_file = open(txt_path, 'w')  # Open the txt file and write

n = 0  # (Modifiable!) Processing from the nth file
file_lists = find_files_with_suffix(path, '.h5')
set_nums = choose_everyday*set_trainday # (Modifiable!) Randomly select a certain number of files before set_day

txt_path = os.path.join(out_path, 'select_data_record.txt')
txt_file = open(txt_path, 'w')

file = []
for data_name in file_lists:
    day = int(data_name[7:9]) 
    if  day <= set_trainday:
        file.append(data_name)  # of folders all less than 21 days old

nums = len(file)  # Get the total length
l = np.linspace(0, nums-1, nums) # Create sequences
select_file = []
for j in range(set_nums):
    name = file[int(l[j])]
    select_file.append(name)
    # print(name)
    txt_file.write(name+'\n')
txt_file.close()
# file_lists.sort()
try:
    outfile_lists = find_files_with_suffix(out_path, '.h5')

except:
    print('Process New Data')

#################################
#   commencement of processing  #
#################################
for ii, file_name in enumerate(select_file):
    # print(file_name)
    # file_name = file_lists[20]
    count = 0  # recount
    day = file_name[7:9]
    if int(day) <= set_trainday:
    # if n <= 100:
        if ii >= n :
            
            print(ii, file_name)
            h5_path = os.path.join(path, file_name)
            #-------------------------
            # Read h5 file data
            h5 = h5py.File(os.path.join(h5_path), "r")
            Sv_image = h5['image']
            Sv_label = h5['label']
            # Select Depth
            # r = Sv_image[3, :, 0]
            
            # r_s = np.where(r >= choosedepth[0])[0][0]
            # r_e = np.where(r <= choosedepth[1])[0][-1]
            # Sv_image = Sv_image[:, r_s:r_e, :]
            # Sv_label = Sv_label[r_s:r_e, :]
            
            # Drawing Outsourcing Rectangles
            # else:
            count += 1 

            height, width = Sv_label.shape
            count_n = 0
            for i in range(0, height, stride):
                for j in range(0, width, stride):
                    left = j - int(size / 2)
                    right = j + int(size / 2)
                    top = i - int(size / 2)
                    bottom = i + int(size / 2)

                    if top < 0:
                        top = 0
                        bottom = top + size
                    if left < 0:
                        left = 0
                        right = left + size
                    if bottom > height:
                        bottom = height
                        top = bottom - size
                    if right > width:
                        right = width
                        left = right - size

                    image_patch = Sv_image[:, top:bottom, left:right]
                    label_patch = Sv_label[top:bottom, left:right]

                    if np.sum(label_patch)  == 0:
                        continue

                    count_n += 1
                    f = h5py.File(os.path.join(out_path, file_name.replace('.h5','_{}_{}.h5'.format(count, count_n))), 'w')
                    f['image'] = image_patch
                    f['label'] = label_patch
                    f.close()

                    Sv = image_patch[0,:,:].copy()
                    label_patch = np.array(label_patch)
                    # Sv[label_patch == 0 ] = -999

                    # plt.pcolormesh(label256,cmap=ClassColor1().scalar_map)
                    custom_cmap = create_custom_cmap(['white', 'red', 'blue'])
                    
                    plt.pcolormesh(label_patch,cmap=custom_cmap, vmin=0, vmax=2)
                    plt.axis('off')
                    plt.gca().invert_yaxis()
                    plt.savefig(os.path.join(png_path, file_name.replace('.h5','_label{}_{}.png'.format(count, count_n))), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.pcolormesh(image_patch[0,:,:], vmin=-80, vmax=-50, cmap=cmaps().ek500)
                    plt.axis('off')
                    plt.gca().invert_yaxis()
                    plt.savefig(os.path.join(png_path, file_name.replace('.h5','_Svlabel{}_{}.png'.format(count, count_n))), bbox_inches='tight', pad_inches=0)
                    plt.close()


            h5.close()

