import numpy as np
import os, shutil, random, skimage
from read_write_mrc import read_mrc, write_mrc
from patchify import patchify, unpatchify
from scipy import ndimage

def normalisation(img):
    norm_img = (img-img.min())/(img.max()-img.min())
    return norm_img

def center_padding(img, target_size):
    pad_widths = [(max((target_size[i] - img.shape[i]) // 2, 0), max((target_size[i] - img.shape[i] + 1) // 2, 0)) for i in range(3)]
    pad_img = np.pad(img, pad_widths)
    return pad_img

def anti_padding(img, target_size):
    if any(img.shape[i] < target_size[i] for i in range(3)):
        raise ValueError("Target size must be smaller than or equal to the input image size for anti-padding.")

    crop_slices = [slice((img.shape[i] - target_size[i]) // 2, (img.shape[i] - target_size[i]) // 2 + target_size[i]) for i in range(3)]
    cropped_img = img[crop_slices[0], crop_slices[1], crop_slices[2]]
    
    return cropped_img

def mask_crop(image, labels):
    cropped_mrc = np.where(labels>0, image, 0)
    return cropped_mrc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def create_and_save_slices(img, img_name, output_folder):
    patches_z = patchify(img, (64,704,704), step=(32,32,32))
    patches_z = patches_z.squeeze((1,2))
    tmp = np.rot90(img, axes=(0,1))
    patches_y = np.flip(tmp, axis = 0)
    patches_y = patchify(patches_y, (64,704,704), step = 32)
    patches_y = patches_y.squeeze((1,2))                 
    tmp = np.rot90(tmp, axes=(2,0))
    patches_x = patchify(tmp, (64 ,704 ,704), step = 32)
    patches_x = patches_x.squeeze((1,2))
    
    # Iterate through axes
    for axis_name, patches in zip(['z', 'y', 'x'], [patches_z, patches_y, patches_x]):
        for i, patch in enumerate(patches):
            patch_filename = f'{img_name}_{axis_name}_patch{i}.mrc'
            patch_path = os.path.join(output_folder, patch_filename)
            write_mrc(patch_path, patch)
    print(f'Slices created for {img_name} at {output_folder}')
         
def split_dataset(mrc_folder, mask_folder, output_folder, split_ratio=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    mrc_files = sorted([f for f in os.listdir(mrc_folder) if os.path.isfile(os.path.join(mrc_folder, f))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))])
    assert len(mrc_files) == len(mask_files), "Number of MRC files and mask files must be the same!"
    combined = list(zip(mrc_files, mask_files))
    random.shuffle(combined)
    mrc_files, mask_files = zip(*combined)
    total = len(mrc_files)
    train_end = int(total * split_ratio[0])
    test_end = train_end + int(total * split_ratio[1])
    splits = {
        "train": (0, train_end),
        "test": (train_end, test_end),
        "val": (test_end, total)
    }
    for split_name, (start, end) in splits.items():
        mrc_out = os.path.join(output_folder, f"mrc_{split_name}")
        mask_out = os.path.join(output_folder, f"mask_{split_name}")
        os.makedirs(mrc_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        for mrc_file, mask_file in zip(mrc_files[start:end], mask_files[start:end]):
            shutil.copy2(os.path.join(mrc_folder, mrc_file), os.path.join(mrc_out, mrc_file))
            shutil.copy2(os.path.join(mask_folder, mask_file), os.path.join(mask_out, mask_file))
            
    shutil.rmtree(mrc_folder)
    shutil.rmtree(mask_folder)
    
    print(f"Original source folders deleted: {mrc_folder}, {mask_folder}")
    print(f"Dataset split completed. Output folder: {output_folder}")
    print(f"Train: {train_end}, Test: {test_end - train_end}, Val: {total - test_end}")
    
def IOU_and_DICE(prediction , Ground_Truth):
    intersection = np.logical_and(prediction,Ground_Truth)
    union = np.logical_or(prediction,Ground_Truth)
    IOU = np.sum(intersection)/np.sum(union)
    DICE =  (2.0 * np.sum(intersection)) / (np.sum(prediction) + np.sum(Ground_Truth))
    return IOU , DICE  
             
def Precision_recall_f1_score(prediction, Ground_Truth):
    true_positive = np.sum(np.logical_and(prediction, Ground_Truth))
    false_positive = np.sum(np.logical_and(prediction, np.logical_not(Ground_Truth)))
    false_negative = np.sum(np.logical_and(np.logical_not(prediction), Ground_Truth))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score 
                

def hpf_sobel(image):
    gradient_x = ndimage.sobel(image, axis=1, mode='constant')
    gradient_y = ndimage.sobel(image, axis=0, mode='constant')
    gradient_z = ndimage.sobel(image, axis=2, mode='constant')
    # Combine the gradients to get the magnitude of the gradient
    sobel_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
    return sobel_magnitude

def ves_mito_separator(img,sigma=2):
    out = img + ndimage.gaussian_filter(img, sigma)
    hpf_out = hpf_sobel(out)
    out = img + 0.75*hpf_out + 0.25*hpf_out*hpf_out
    out = normalisation(out)*normalisation(img)
    out = normalisation(out)
    return out
