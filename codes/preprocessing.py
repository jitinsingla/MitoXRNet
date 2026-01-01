import numpy as np
import os,gc,glob, sys ,time
import argparse
from patchify import patchify
from utils import *


def preprocessing(split_ratio = 0.8):
    
    ##--- Use absolute paths if these paths not working----##
    print()
    print("-----Splitting Dataset into train and Validation Sets-----")    
    print()
    mrc_folder = '../Data/Training/MRCs'
    mask_folder = '../Data/Training/Labels'
    output_folder = "../Data/Training/"
    split_dataset(mrc_folder, mask_folder, output_folder,split_ratio = split_ratio)
    print()
    folders = {
        d.name for d in os.scandir(output_folder)
        if d.is_dir() and d.name.endswith("_preprocessed")
    }

    pairs = []

    for name in folders:
        if name.startswith("mrc_"):
            counterpart = name.replace("mrc_", "mask_", 1)
            if counterpart in folders:
                pairs.append((name, counterpart))
    
    target_size = (704,704,704)
    count=0
    skipped = 0
    for mrc, mask in pairs:

        assert os.path.isdir(os.path.join(output_folder, mrc))
        assert os.path.isdir(os.path.join(output_folder, mask))

        img_name_list = sorted(glob.glob(os.path.join(output_folder, mrc, "*.mrc")))
        mask_name_list =  sorted(glob.glob(os.path.join(output_folder, mask, "*.mrc")))

        
        for img_name,mask_name in zip(img_name_list, mask_name_list):

            img = read_mrc(img_name)
            mask = read_mrc(mask_name)
            assert img.shape == mask.shape, "Mask and image don't match."

            if any(dim > 704 for dim in img.shape):
                print(f"Skipping {img_name} (shape: {img.shape}) — exceeds 704 in one or more dimensions.")
                skipped += 1
                continue

            imgpp2 = ves_mito_separator(img)
            imgpp2 = mask_crop(imgpp2,mask)
            padded_imgpp2 = center_padding(imgpp2,target_size)
            padded_mask = center_padding(mask,target_size)

            assert padded_imgpp2.shape == (704,704,704), f"Something error after padding in padded_imgpp2 {padded_imgpp2.shape}"
            assert padded_mask.shape == (704,704,704), f"Something error after padding in padded_mask"
            gc.collect()
            write_mrc(img_name, padded_imgpp2.astype('float32'))
            write_mrc(mask_name, padded_mask.astype('int8'))
            count+=1
            print(f'-----Preprocessing Done for {os.path.basename(img_name)} and corresponding Mask-----')
    print()                             
    print(f'-----Preprocessing Completed , total skipped images {skipped}-----', flush=True)
    print()
    time.sleep(2)
    print('---------Slicing Started---------', flush=True)
    print()
    for i in folders:    
        path_to_subfolder = os.path.join('../Data/Training' , i)
        output_folder = os.path.join('../Data/Slices/' , i.split("_preprocessed")[0]+'_Slices')
        os.makedirs(output_folder,exist_ok=True)

        for filename in os.listdir(path_to_subfolder):
            if filename.endswith('.mrc'):  
                image_path = os.path.join(path_to_subfolder, filename)

                image = read_mrc(image_path)#.get()

                img_name, _ = os.path.splitext(filename)

                create_and_save_slices(image, img_name,output_folder)
                del img_name, image, image_path
                gc.collect()
    print()                             
    print('---------Slicing Completed---------')
                                 
                                 
def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing for MitoXRNet (train/val split, padding, slicing)"
    )

    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Train/Validation split ratio (default = 0.8)"
    )

    args = parser.parse_args()

    preprocessing(split_ratio=args.split_ratio)
    print()
    print('--------------- Preprocessing Pipeline Completed Successfully ---------------')


if __name__ == "__main__":
    main()
