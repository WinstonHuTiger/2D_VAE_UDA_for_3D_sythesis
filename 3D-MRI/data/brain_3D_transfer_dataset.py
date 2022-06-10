from data.base_dataset import BaseDataset
from data.image_folder import get_custom_file_paths, natural_sort
import nibabel as nib
import random
from torchvision import transforms
import os
import numpy as np

import torch
from models.networks import setDimensions
import skimage.transform as sk_trans
from data.data_augmentation_3D import ColorJitter3D, PadIfNecessary, SpatialRotation, SpatialFlip, getBetterOrientation, toGrayScale

class brain3DTransferDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        
        parser.add_argument('--dataset_name_1', type=str, default="domain1_train", 
                            help='the first domain path')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        parser.add_argument("--dataset_name_2", type = str, default = "domain2_val", 
                            help = "the name of the second data set")
        
        parser.add_argument("--train_number", type=int, default=10000000,
                            help="the training number of the dataset")
        return parser
    
    def __init__(self, opt):
        super().__init__(opt)
        # self.A1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 't1', opt.phase), 't1.nii.gz'))
        # self.A2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'flair', opt.phase), 'flair.nii.gz'))
        # self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'dir', opt.phase), 'dir.nii.gz'))
        self.A1_1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
                                                                        opt.dataset_name_1), 't2.nii.gz'))
        self.A2_1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
                                                                        opt.dataset_name_1), 'flair.nii.gz'))
        self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
                                                                       opt.dataset_name_1), 't1.nii.gz'))
       
        self.A1_2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
                                                                       opt.dataset_name_2), 't2.nii.gz'))
        self.A2_2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
                                                                       opt.dataset_name_2), 'flair.nii.gz'))
        
        
        self.A1_size = len(self.A1_1_paths)  # get the size of dataset A
        self.A2_size = len(self.A1_2_paths)
        self.B_size = len(self.B_paths)  # get the size of dataset B
        setDimensions(3)
        opt.input_nc = 2
        opt.output_nc = 1

        transformations = [
            transforms.Lambda(lambda x: getBetterOrientation(x, "IPL")),
            transforms.Lambda(lambda x: np.array(x.get_fdata())[np.newaxis, ...]),
            # transforms.Lambda(lambda x: sk_trans.resize(x, (256, 256, 160), order = 1, preserve_range=True)),
            # image size [1, 160, 240, 240]
            transforms.Lambda(lambda x: x[:,8:152,24:216,24:216]),
            # transforms.Lambda(lambda x: resize(x, (x.shape[0],), order=1, anti_aliasing=True)),
            transforms.Lambda(lambda x: toGrayScale(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float16 if opt.amp else torch.float32)),
            # transforms.Resize((256, 256)),
            PadIfNecessary(3),
        ]

        if(opt.phase == 'train'):
            self.updateTransformations += [
                SpatialRotation([(1,2), (1,3), (2,3)], [*[0]*12,1,2,3], auto_update=False), # With a probability of approx. 51% no rotation is performed
                SpatialFlip(dims=(1,2,3), auto_update=False)
            ]
        transformations += self.updateTransformations
        self.transform = transforms.Compose(transformations)
        self.colorJitter = ColorJitter3D((0.3,1.5), (0.3,1.5))
        
    
    def __getitem__(self, index):
        A1_1_path = self.A1_1_paths[index % self.A1_size]  # make sure index is within then range
        A1_1_img: nib.Nifti1Image = nib.load(A1_1_path)
        affine1 = A1_1_img.affine

        A2_1_path = self.A2_1_paths[index % self.A1_size]  # make sure index is within then range
        A2_1_img = nib.load(A2_1_path)
        
        A1_2_path = self.A1_2_paths[index % self.A2_size]  # make sure index is within then range
        A1_2_img: nib.Nifti1Image = nib.load(A1_2_path)
        affine2 = A1_2_img.affine

        A2_2_path = self.A2_2_paths[index % self.A2_size]  # make sure index is within then range
        A2_2_img = nib.load(A2_2_path)
        if self.opt.paired:   # make sure index is within then range
            index_B = index % self.B_size
            B_path = self.B_paths[index_B]
            B_img = nib.load(B_path)
        else:
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            B_img = nib.load(B_path)
        
        A1_1 = self.transform(A1_1_img)
        A2_1 = self.transform(A2_1_img)
        
        A1_2 = self.transform(A1_2_img)
        A2_2 = self.transform(A2_2_img)
        
        if self.opt.phase == 'train':
            A1_1 = self.colorJitter(A1_1)
            A2_1 = self.colorJitter(A2_1, no_update=True)
            
            A1_2 = self.colorJitter(A1_2)
            A2_2 = self.colorJitter(A2_2, no_update=True)
        
        A_1 = torch.concat((A1_1, A2_1), dim=0)
        A_2 = torch.concat((A1_2, A2_2), dim=0)
        B = self.transform(B_img)
        
        return {'A': A_1, 'B': B, 'affine': affine1, 'axis_code': "IPL", 'A_paths': A1_1_path, 'B_paths': B_path}, \
    {'A': A_2, 'B': B, 'affine': affine2, 'axis_code': "IPL", 'A_paths': A1_2_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        
        return min(self.opt.train_number, max(self.A1_size, self.B_size, self.A2_size) )