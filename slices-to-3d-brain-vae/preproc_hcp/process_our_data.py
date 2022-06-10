import os
import numpy as np
import logging
import gc
from skimage import transform
import glob
import zipfile, re
import utils
from PIL import Image


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_custom_file_paths(folder, name):
    image_file_paths = []
    for root, _, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            if filename.endswith(name):
                file_path = os.path.join(root, filename)
                image_file_paths.append(file_path)
    return image_file_paths


def process_data(input_foler, processed_folder, size, target_resolution):
    t1_image_paths = natural_sort(get_custom_file_paths(input_foler, "t1.nii.gz"))
    for path in t1_image_paths:
        image, _, image_hdr = utils.load_nii(path)
        # print(image.shape)
        image = utils.crop_or_pad_volume_to_size_along_x(image, 256)
        image = utils.crop_or_pad_volume_to_size_along_y(image, 256)
        image = utils.crop_or_pad_volume_to_size_along_z(image, 256)

        # ==================
        # normalize the image
        # ==================
        image_normalized = utils.normalise_image(image, norm_type='div_by_max')
        # ======================================================
        # rescale, crop / pad to make all images of the required size and resolution
        # ======================================================
        scale_vector = [image_hdr.get_zooms()[0] / target_resolution[0],
                        image_hdr.get_zooms()[1] / target_resolution[1],
                        image_hdr.get_zooms()[2] / target_resolution[2]]
      
        image_rescaled = image_normalized
        image_name = path.split("/")[-1]
        folder_name = path.split("/")[-2]
        volume_dir = os.path.join(processed_folder, folder_name)
        os.makedirs(volume_dir, exist_ok=True)
       
        
        for i in range(size[1]):
            slice_path = os.path.join(volume_dir, "slice_{:06d}.jpeg".format(i))
            slice = image_rescaled[:, i, :] * 255
            image = Image.fromarray(slice.astype(np.uint8))
            image.save(slice_path)


def load_and_process_data(input_folder,
                          processed_folder
                          , size,
                          target_resolution):
    utils.makefolder(processed_folder)

    logging.info("Processing Now!")
    process_data(input_folder, processed_folder, size, target_resolution)
    logging.info("Processing finished!")


input_folder = "your_path"
process_foler = "your_path"
load_and_process_data(input_folder, process_foler,size =(256, 256, 256), target_resolution=(0.7, 0.7, 0.7))
