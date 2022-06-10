"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision
import nibabel as nib
import matplotlib.pyplot as plt
import re
import matplotlib as mpl

def colorFader(mix: float, c1='k',c2='r'): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    """
    Given a float number in the range [0,1], returns a interpolated gradient rgb color of the color c1 and c2
    https://stackoverflow.com/a/50784012
    """
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return (1-mix)*c1 + mix*c2

def colorFaderTensor(mix: torch.Tensor, c1='k',c2='r'): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    """
    Given a float number in the range [0,1], returns a interpolated gradient rgb color of the color c1 and c2
    https://stackoverflow.com/a/50784012
    """
    c1=torch.tensor(mpl.colors.to_rgb(c1))
    c2=torch.tensor(mpl.colors.to_rgb(c2))
    mix = torch.stack([(1-mix)*c1[i] + mix*c2[i] for i in range(3)], dim=-1)
    return mix

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255  # post-processing: tranpose and scaling
        input_image = torch.clamp(input_image.type(torch.float32), 0, 255)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_nifti_image(image_tensor: torch.Tensor, image_path: str, affine: np.ndarray, axis_code: str):
    """
    Save a MRI numpy image to the disk. Resize the image by a scaling factor and enforce an aspect ratio of 1
    """
    image_tensor = image_tensor.detach().cpu().numpy()[0,0].astype(np.uint8)
    if len(image_tensor.shape)==4:
        shape_3d = image_tensor.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        image_tensor = image_tensor.copy().view(dtype=rgb_dtype).reshape(shape_3d)  # copy used
    new_img = nib.Nifti1Image(image_tensor, np.eye(4))
    orig_ornt = nib.io_orientation(affine)
    transform = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(axis_code), orig_ornt)
    new_img =  new_img.as_reoriented(transform)
    new_img = nib.Nifti1Image(new_img.get_fdata(), affine)
    nib.save(new_img, image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)

def load_val_log(path: str):
    val = []
    legend = None
    with open(path) as f:
        lines = [line.rstrip() for line in f]
    for line in lines:
        if line.startswith('='):
            val = []
            continue
        line = line.split(') ')[-1]
        ls = line.split(', ')
        val.append([float(l.split(': ')[-1]) for l in ls])
        if legend is None:
            legend = [l.split(': ')[0] for l in ls]
    return val, legend

def val_log_2_png(path: str):
    folder_path = os.path.dirname(path)
    name = os.path.basename(folder_path)
    val = load_val_log(path)
    plt.figure()
    plt.plot(val)
    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel('L1 Loss')
    plt.legend(['validation loss'])
    
    out_path = os.path.join(folder_path, 'val_loss.png')
    plt.savefig(out_path, format='png', bbox_inches='tight')
    plt.cla()
    print('Saved plot at', out_path)

def load_loss_log(path: str, dataset_size=0):
    """
    Loads the given loss file, extracts all losses and returns them in a struct
    """
    legend = []
    y = []
    x = []
    has_legend = False
    with open(path) as f:
        lines = [line.rstrip() for line in f]
    for line in lines:
        if line.startswith('='):
            legend = []
            y = []
            x = []
            has_legend=False
            continue
        meta_data = re.sub('\(|\)|\:|\,', '', re.search('\(.*\)', line).group(0)).split()
        x_i = (int(meta_data[1]) + int(meta_data[3])/dataset_size)-1
        line =  re.sub('\(.*\)', '', line)
        y_i = []
        for t in line.split():
            try:
                y_i.append(float(t))
            except ValueError:
                if not has_legend and len(t)>1:
                    legend.append(t.replace(':', ''))
        has_legend=True
        y.append(y_i)
        x.append(x_i)
    return x, y, legend

def loss_log_2_png(path: str, dataset_size=234):
    """
    Loads the given loss file, extracts all losses and returns them in a struct
    """
    x, y, legend = load_loss_log(path, dataset_size)

    folder_path = os.path.dirname(path)
    name = os.path.basename(folder_path)
    plt.figure()
    plt.plot(x,y)
    plt.legend(legend)
    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    out_path = os.path.join(folder_path, 'train_loss.png')
    plt.savefig(out_path, format='png', bbox_inches='tight')