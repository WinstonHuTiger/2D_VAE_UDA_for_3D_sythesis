import os
import math
from shutil import copyfile
from data.image_folder import get_custom_file_paths
from tqdm import tqdm


CT_set = sorted(get_custom_file_paths('/home/kreitnerl/Datasets/registered_checked_wopathfx_SS', 'ct.nii.gz'))
MRI_set = sorted(get_custom_file_paths('/home/kreitnerl/Datasets/registered_checked_wopathfx_SS', 'T1.nii.gz'))
CT_set = [p for p in CT_set if 'sorted' in p]

assert len(CT_set) == len(MRI_set)

test_split = 0.1
train_set_size = math.ceil(len(CT_set) * (1-test_split))
test_set_size = len(CT_set) - train_set_size
print('Train set size: %d, Test set size: %d' % (train_set_size, test_set_size))

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# dataroot = '/media/data_4T/linus/3D_brain_mri/'
dataroot = '/home/kreitnerl/Datasets/CMR2MRI_wopath/'
mkdir(dataroot)
mkdir(os.path.join(dataroot, 'ct/'))
mkdir(os.path.join(dataroot, 'mri/'))
mkdir(os.path.join(dataroot, 'ct/', 'train'))
mkdir(os.path.join(dataroot, 'ct/', 'test'))
mkdir(os.path.join(dataroot, 'mri/', 'train'))
mkdir(os.path.join(dataroot, 'mri/', 'test'))


print('Creating train set...')
for i in tqdm(range(train_set_size)):
    copyfile(CT_set[i], os.path.join(dataroot, 'ct/train/%d.nii.gz'%i))
    copyfile(MRI_set[i], os.path.join(dataroot, 'mri/train/%d.nii.gz'%i))

print('Creating test set...')
for i in tqdm(range(train_set_size, len(CT_set))):
    copyfile(CT_set[i], os.path.join(dataroot, 'ct/test/%d.nii.gz'%i))
    copyfile(MRI_set[i], os.path.join(dataroot, 'mri/test/%d.nii.gz'%i))