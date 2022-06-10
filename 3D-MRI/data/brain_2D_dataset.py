from data.image_folder import get_custom_file_paths, natural_sort
from data.base_dataset import BaseDataset
from PIL import Image
import random
from torchvision import transforms
import os
import torch
import numpy as np
from data.data_augmentation_3D import PadIfNecessary, SpatialFlip, SpatialRotation, ColorJitterSphere3D, toGrayScale
from models.networks import setDimensions

class Brain2DDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        setDimensions(2)
        self.A1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 't1', opt.phase), '.png'))
        self.A2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'flair', opt.phase), '.png'))
        self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'dir', opt.phase), '.png'))
        self.A_size = len(self.A1_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        opt.input_nc = 2
        opt.output_nc = 1

        self.transformations = [
            transforms.Lambda(lambda x: toGrayScale(x)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.type(torch.float16 if opt.amp else torch.float32)),
            # SpatialRotation([(1,2)]),
            PadIfNecessary(3),
        ]

        if(opt.phase == 'train'):
            self.updateTransformations += [
                SpatialRotation([(1,2)], [*[0]*3,1,2,3], auto_update=False),
                SpatialFlip(dims=(1,2), auto_update=False),
            ]
            self.transformations += self.updateTransformations
        self.transform = transforms.Compose(self.transformations)
        self.colorJitter = ColorJitterSphere3D((0.3, 1.5), (0.3,1.5), sigma=0.5, dims=2)

    def __getitem__(self, index):
        A1_path = self.A1_paths[index % self.A_size]  # make sure index is within then range
        A2_path = self.A2_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.paired:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A1_img = np.array(Image.open(A1_path), dtype=np.float32)
        A2_img = np.array(Image.open(A2_path), dtype=np.float32)
        B_img = np.array(Image.open(B_path), dtype=np.float32)

        A1 = self.transform(A1_img)
        A2 = self.transform(A2_img)
        B = self.transform(B_img)
        if self.opt.phase == 'train' and self.opt.direction=='AtoB':
            A1 = self.colorJitter(A1)
            A2 = self.colorJitter(A2, no_update=True)
        A = torch.concat((A1, A2), dim=0)
        return {'A': A, 'B': B, 'A_paths': A1_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
        
        
