import os
import random
import numpy as np
import shutil
# #######################
# # Defining Functions   #
# #######################
def find_files_with_suffix(folder_path, suffix):
    # Get the paths of all files in a folder using the os module
    all_files = os.listdir(folder_path)
    # Filter files ending with a specified suffix
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files

def cut_file(source_path, destination_path):
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
    else:
        print("The source file does not exist!", source_path)

# #######################
# # Defining the path    #
# #######################
h5_crop_path = r'crop_selectdataset_0_300_select_256_180'
path_list = find_files_with_suffix(h5_crop_path, '.h5')
h5s_train_path = os.path.join(h5_crop_path, 'train')
h5s_val_path = os.path.join(h5_crop_path, 'val')
os.makedirs(h5s_train_path, exist_ok=True)
os.makedirs(h5s_val_path, exist_ok=True)
traintxt_path = os.path.join(h5_crop_path, 'traindata_record.txt')
valtxt_path = os.path.join(h5_crop_path, 'valdata_record.txt')
traintxt_file = open(traintxt_path, 'w')  # Open the txt file and write
valtxt_file = open(valtxt_path, 'w')  # Open the txt file and write

train_ratio = 0.7  # Data sets are divided 7:3 by
nums = len(path_list)  # Get total length
l = np.linspace(0, nums-1, nums)  # Creating a Sequence
random.shuffle(l)  # randomly randomize the order of things

train_size = int(nums * train_ratio)  # Training set length
train_indexs = l[:train_size]   # Divide the training set sequence
val_indexs = l[train_size:]  # Divide the validation set sequence

train_indexs, val_indexs = np.array(train_indexs.astype(int)), np.array(val_indexs.astype(int))

for ii ,train_index in enumerate(train_indexs):
    train_filename = path_list[train_index]
    print(ii, train_index, train_filename)
    cut_file(os.path.join(h5_crop_path, train_filename), os.path.join(h5s_train_path, train_filename) )
    traintxt_file.write(str(train_index)+train_filename)
traintxt_file.close()

for ii ,val_index in enumerate(val_indexs):
    val_filename = path_list[val_index]
    print(ii, val_index, val_filename)
    cut_file(os.path.join(h5_crop_path, val_filename), os.path.join(h5s_val_path, val_filename))
    valtxt_file.write(str(val_index)+ val_filename)
valtxt_file.close()

