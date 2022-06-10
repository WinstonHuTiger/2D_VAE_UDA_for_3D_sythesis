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

class brain3DDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--dataset_name', type=str, default="center_1_2", help='new dataset option')
        parser.add_argument("--train_number", type=int, default=10000000, help="the training number of the dataset")
        parser.add_argument("--k_fold", type = int, default = 0, help = "the fold parameter, K")
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser
    
    def __init__(self, opt):
        super().__init__(opt)
        # self.A1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 't1', opt.phase), 't1.nii.gz'))
        # self.A2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'flair', opt.phase), 'flair.nii.gz'))
        # self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'dir', opt.phase), 'dir.nii.gz'))
        suffix_str = ""
        if opt.k_fold > 0:
            
            suffix_str = "_" + str(opt.k_fold) + "_fold"
        self.A1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
                                                                        opt.dataset_name+ "_" + str(opt.phase) + suffix_str), 't2.nii.gz'))
        self.A2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
                                                                        opt.dataset_name+ "_" + str(opt.phase)+ suffix_str), 'flair.nii.gz'))
        self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
                                                                       opt.dataset_name + "_" + str(opt.phase)+ suffix_str), 't1.nii.gz'))
        
        self.A_size = len(self.A1_paths)  # get the size of dataset A
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
        A1_path = self.A1_paths[index % self.A_size]  # make sure index is within then range
        A1_img: nib.Nifti1Image = nib.load(A1_path)
        affine = A1_img.affine

        A2_path = self.A2_paths[index % self.A_size]  # make sure index is within then range
        A2_img = nib.load(A2_path)
        if self.opt.paired:   # make sure index is within then range
            index_B = index % self.B_size
            B_path = self.B_paths[index_B]
            B_img = nib.load(B_path)
        else:
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            B_img = nib.load(B_path)
        A1 = self.transform(A1_img)
        A2 = self.transform(A2_img)
        if self.opt.phase == 'train':
            A1 = self.colorJitter(A1)
            A2 = self.colorJitter(A2, no_update=True)
        A = torch.concat((A1, A2), dim=0)
        B = self.transform(B_img)
        return {'A': A, 'B': B, 'affine': affine, 'axis_code': "IPL", 'A_paths': A1_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return min(self.opt.train_number, max(self.A_size, self.B_size) )