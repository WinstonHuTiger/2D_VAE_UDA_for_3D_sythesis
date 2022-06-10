

# Efficient 3D Image Translation Method for the Spine

This repository is part of a project that focuses on (un)paired image style translation between CT and MRI images of the spine. The generated images are used for subsequent segmentation tasks.

The project further investigates the effectiveness of different state-of-the-art deep learning methods such as
 - Pix2Pix / CycleGAN / CUT
- 2D / 3D input data
 - Deterministic / Bayesian


## Usage
In the following we assume you work on a linux machine with a CUDA enabled CUDA-enabled GPU (https://developer.nvidia.com/cuda-gpus).

Clone this repository to your local workspace:
```sh
# Clone repo from github
git clone https://github.com/Linus4world/3D-MRI-style-transfer.git
# Navigate into the project folder
cd ./3D-MRI-style-transfer
```

To use this code you have two options: Docker or Manual

### Docker
You can simply run an inference using our pretrained model using docker. For this, please install docker and docker-compose. Additionally, make sure to have at least 6GB of diskspace available. Note that with this method, the program will not utilize the GPU but the CPU. 

Before you can start, you first have to configure the `docker-compose.yaml` file. Replace the following three placeholders:
- `DATASET_DIR`: The absolute path to your dataset
- `RESULTS_DIR`: The absolute path to the directory the results should be saved in
- `CHECKPOINT_PATH`: The absolute path to the checkpoint file that should be used by the generator.

Then, you can run:
```sh
# Build and run docker image
docker-compose up
```

After the container finished its execution you will find the results in the configured `RESULTS_DIR`.

### Manual
We will show the manual installation for an ubuntu:20.04 system and a Nvidia GPU using CUDA version 10.2.

First, install Miniconda (or Anaconda) from the [offical website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Create a virtual environment with python 3.8.10:
```sh
# Create virtual environment with name 'env' and python version 3.8.10
conda env create --name env python=3.8.10
# Auto-enable this environment on bash startup
conda init bash
echo "source /opt/conda/bin/activate && conda activate env" >> ~/.bashrc
# Activate environment
conda activate env
```

Install PyTorch and torchvision. Make sure to install a [version compatible with your GPU's CUDA version](https://pytorch.org/get-started/previous-versions/). E.g. for a conda environment and CUDA 10.2 run:
```sh
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Install all remaining dependencies:
```sh
pip install -r requirements.txt
```

You can perform training / testing using
```sh
# Train
python train.py

# Test
python test.py
```

### Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/pdf/2007.15651).
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

If you use the original [pix2pix](https://phillipi.github.io/pix2pix/) and [CycleGAN](https://junyanz.github.io/CycleGAN/) model included in this repo, please cite the following papers
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```

Bayesian Networks
```
@misc{esposito2020blitzbdl,
    author = {Piero Esposito},
    title = {BLiTZ - Bayesian Layers in Torch Zoo (a Bayesian Deep Learing library for Torch)},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/piEsposito/blitz-bayesian-deep-learning/}},
}
```
