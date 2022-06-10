import nibabel as nib
import numpy as np
import os
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import re


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            fname: str
            if fname.endswith('.nii.gz'):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

distribution = Counter({i: 0. for i in range(256)})

img_folder = '/home/kreitnerl/Datasets/3D_brain_mri/trainT1'
dataset = natural_sort(make_dataset(img_folder))
# for img_path in dataset:
#     if int(img_path.split('.')[0].split('/')[-1]) < 774:
#         move(img_path, '/media/data_4T/william/CT_2_MRI/ct/test/')
for img_path in tqdm(dataset):

    a = np.array(nib.load(img_path).get_fdata())
    a = a[48:240,80:240,36:260]
    a_min = np.amin(a)
    a_max = np.amax(a) - a_min
    a = (a - a_min) / a_max * 255.
    a = a.astype(np.uint8)
    unique, counts = np.unique(a, return_counts=True)
    d = Counter(dict(zip(unique, counts)))
    distribution.update(d)

n_pixel = sum(distribution.values())
s = 0
for i,c in distribution.items():
    s += i*c
mean = s/n_pixel
std = 0
for i,c in distribution.items():
    std += c*((i-mean)**2)
std = np.sqrt(std/n_pixel)
print('mean: ', mean, 'std: ', std)
plt.bar(list(distribution.keys()), list(distribution.values()))
plt.title(img_folder.split('/')[-1])
plt.xlabel('Grey scale intensity')
plt.ylabel('Number of voxels')
plt.savefig('distribution_%s.png'%img_folder.split('/')[-1], bbox_inches='tight')