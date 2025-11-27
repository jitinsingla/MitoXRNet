import torch
import os
from torch.utils.data import Dataset, DataLoader
import glob
from utils import create_patches, create_patches_list
from read_write_mrc import read_mrc, write_mrc
import cupy as np
import gc


def label_smoothing(labels, smoothing = 0.1):
    labels = labels * (1 - smoothing) + smoothing * 0.5
    return labels
    
class SliceLoader(Dataset):
    def __init__(self, data_root_dir=None, images_dir="images", labels_dir="labels" , smoothing = 0.1):
        assert data_root_dir is not None, "Please pass the correct data_root_dir"

        images_list = sorted(glob.glob(os.path.join(data_root_dir, images_dir,'*.mrc')))
            

        labels_list = sorted(glob.glob(os.path.join(data_root_dir, labels_dir,'*.mrc')))
            
        assert len(labels_list) == len(images_list), "Lengths of labels and images should be the same."

        self.images_list = images_list
        self.labels_list = labels_list
        self.label_smoothing = smoothing
        gc.collect()

    def __getitem__(self, item):
        image = read_mrc(os.path.join(self.images_list[item]))
        image = np.stack((image,) * 1, axis=0)

        label = read_mrc(os.path.join(self.labels_list[item]))

        label1 = (label == 2.0).astype(np.float32)
        label3 = (label == 5.0).astype(np.float32)
        label_combined = np.stack((label1,label3),0)
        image_tensor = torch.as_tensor(image)
        label_tensor = torch.as_tensor(label_combined)

        return image_tensor, label_tensor
    
    def __len__(self):
        return len(self.images_list)
