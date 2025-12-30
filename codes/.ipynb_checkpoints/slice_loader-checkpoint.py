import torch
import os
from torch.utils.data import Dataset, DataLoader
import glob
import gc
import numpy as np
from utils import *
    
class SliceLoader(Dataset):
    def __init__(self, data_root_dir=None, images_dir="images", labels_dir="labels" ):
        assert data_root_dir is not None, "Please pass the correct data_root_dir"

        images_list = sorted(glob.glob(os.path.join(data_root_dir, images_dir,'*.mrc')))
            

        labels_list = sorted(glob.glob(os.path.join(data_root_dir, labels_dir,'*.mrc')))
            
        assert len(labels_list) == len(images_list), "Lengths of labels and images should be the same."

        self.images_list = images_list
        self.labels_list = labels_list
        gc.collect()

    def __getitem__(self, item):
        image = read_mrc(os.path.join(self.images_list[item]))
        image = np.expand_dims(image, axis=0).astype(np.float32)

        label = read_mrc(os.path.join(self.labels_list[item]))
        
        label_nucleus = (label == 2.0).astype(np.float32)
        label_mito = (label == 5.0).astype(np.float32)
        label_combined = np.stack((label_nucleus,label_mito),0)
        image_tensor = torch.as_tensor(image)
        label_tensor = torch.as_tensor(label_combined)

        return image_tensor, label_tensor
    
    def __len__(self):
        return len(self.images_list)
    
class SliceLoader_MRC(Dataset):
    def __init__(self, data_root_dir=None, images_dir="images"):
        assert data_root_dir is not None, "Please pass the correct data_root_dir"

        images_list = sorted(glob.glob(os.path.join(data_root_dir, images_dir,'*.mrc')))
        self.images_list = images_list
        gc.collect()

    def __getitem__(self, item):
        image = read_mrc(os.path.join(self.images_list[item]))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        image_tensor = torch.as_tensor(image)
        return image_tensor
    
    def __len__(self):
        return len(self.images_list)
