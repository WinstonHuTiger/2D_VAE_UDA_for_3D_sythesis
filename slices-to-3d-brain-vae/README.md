# Readme
This is a repository containing code for the paper "Modelling the Distribution of 3D Brain MRI using a 2D Slice VAE", published in MICCAI 2020.
Paper link: https://arxiv.org/abs/2007.04780

This repository contains (a) code to train slice VAE's and (b) to fit Gaussian models in the latent space of those VAEs, for 64^3, 128^3 and 256^3 brain volumes.
This code uses pytorch and python 3. Full requirements of the anaconda package are given in requirements.txt

Additionally, the paths are all written in a way that files should be run from the root directory of the repository.
Eg to train the VAE, I would run
`python experiments/MICCAI-release-version/main_experiment_128.py --some --arguments --here`
rather than running `python main_experiment_128.py --some --arguments --here` from the `experiments/MICCAI-release-version/` directory.


## Training VAES
main_experiment_64.py
The training code for the VAE expects the slices to be pre-computed, i.e. it works with images, not volumes.  Additionally, it expects the training slices to be arranged in the following directory structure.

``` 
├── train
│   ├── volume_000000
│   │   ├── slice_000000.jpeg
│   │   ├── slice_000001.jpeg
│   │   ├── slice_000002.jpeg
│   │   ├── slice_000003.jpeg
│   │   ├── slice_000004.jpeg
...
```
The mri_data_dir variable should be set to the correct data directory.  The preprocessing script is also included.

## Fitting Gaussian Models
`Gaussian_latent_model_64.ipynb`

The notebook contains code to sample volumes from the Gaussian model, and to save the volumes to file.

In this script the batch size is used as the slice dimension, for cubic volumes eg 64^3, the image size is 64x64, and the batch size needs to be 64 also.
I recommend running the notebooks on GPU, as it is a lot faster.

Code to compute MMD and MS-SSIM in pytorch is availble at https://github.com/cyclomon/3dbraingen.

## Data Preprocessing
data_hcp.py
This is a script to preprocess the HCP (http://www.humanconnectomeproject.org) volumes into slices.

The following variables need to be set to point to your own directories:
input_folder = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/HCP/3T_Structurals_Preprocessed/'
preprocessing_folder = '/scratch_net/biwinator02/voanna/gpflow-data/hcp_preproc/MRI_slices_64_isotropic'

When calling the main function, make sure to pick the right setting for the right volume size:
### 64^3
```
size=(64, 64, 64),
target_resolution=(2.8, 2.8, 2.8)
```
### 128^3
```
size=(128, 128, 128),
target_resolution=(1.4, 1.4, 1.4)
```
### 256^3
```
size=(256, 256, 256),
target_resolution=(0.7, 0.7, 0.7)
```
for use in the following function call:
```
data_hcp = load_and_maybe_process_data(input_folder,
                                       preprocessing_folder,
                                       idx_start=0,
                                       idx_end=1040,
                                       protocol='T1',
                                       size=(64, 64, 64),
                                       target_resolution=(2.8, 2.8, 2.8),
                                       force_overwrite=False)
```
The actual files used are called something like `Originals/HCP/3T_Structurals_Preprocessed/878877/T1w/T1w_acpc_dc_restore_brain.nii.gz`

I used volumes 0000 - 0959 for training, 0960 - 0999 for validation, and 1000-1039 for testing. The splitting into
train/test/val was done manually


## Trained Models
I am including trained models for the three resolutions, in the directory `experiments/MICCAI-release-version/trained_models`. These are used in the jupyter notebooks.



## Misc.
I include a `driver_64.py` script that can be used to automatically generate shell scripts with different hyperparameter combinations in a way that can be easily submitted to the cluster.




## Citations
To cite, please use:
```
@inproceedings{volokitin2020modelling,
  title={Modelling the Distribution of 3D Brain MRI using a 2D Slice VAE},
  author={Volokitin, Anna and Erdil, Ertunc and Karani, Neerav and Tezcan, Kerem Can and Chen, Xiaoran and Van Gool, Luc and Konukoglu, Ender},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={657--666},
  year={2020},
  organization={Springer}
}
```


