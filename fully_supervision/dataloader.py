from torch.utils.data import Dataset
import random
import torch
import os
import h5py
class MyDataset(Dataset):
    def __init__(
        self,
        _base_dir,
        split='train', 
        data_type='',
        seed=1307
    ):
        self._base_dir = _base_dir
        self.split = split
        self.seed = seed
        self.sample_list = []
        self.data_type = data_type
        # 读取训练集
        if self.split == "train":
            self.sample_list = os.listdir(os.path.join(self._base_dir, 'train'))
            random.shuffle(self.sample_list)
        # 读取验证集
        elif self.split == "val":
            self.sample_list = os.listdir(os.path.join(self._base_dir, 'val'))
            random.shuffle(self.sample_list)


    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        if self.split == "train":
            h5f = h5py.File(os.path.join(self._base_dir, 'train', case), "r")
        else:
            h5f = h5py.File(os.path.join(self._base_dir, 'val', case), "r")
        image = h5f["image"][:]
        
        label = h5f["label"][:]
        
        if self.data_type == '120+d':
            image = image[[0, 3], :, :]
        elif self.data_type == '70+d':
            image = image[[1, 3], :, :]
        elif self.data_type == '38+d':
            image = image[[2, 3], :, :]
        elif self.data_type == '120+70+d':
            image = image[[0, 1, 3], :, :]
        elif self.data_type == '120+38+d':
            image = image[[0, 2, 3], :, :]
        elif self.data_type == '70+38+d':
            image = image[[1, 2, 3], :, :]
        elif self.data_type == '120+70+38+d':
            image = image[[0, 1, 2, 3], :, :]
        elif self.data_type == '120':
            image = image[[0], :, :]
        elif self.data_type == '70':
            image = image[[1], :, :]    
        elif self.data_type == '38':
            image = image[[2], :, :]
        elif self.data_type == '120+70':
            image = image[[0, 1], :, :]
        elif self.data_type == '120+38':
            image = image[[0, 2], :, :]
        elif self.data_type == '70+38':
            image = image[[1, 2], :, :]
        elif self.data_type == '120+70+38':
            image = image[[0, 1, 2], :, :]
        random.seed(self.seed)
        # Convert data to tensor format
        img = torch.from_numpy(image).nan_to_num(0.0).float()
        random.seed(self.seed)
        label = torch.from_numpy(label).unsqueeze(0).nan_to_num(0.0).float()
        return img, label
    
    def __len__(self):
        return len(self.sample_list)

