import random
import torch.nn.functional as F
from collections.abc import Sequence
import torch
import nibabel as nib
import numpy as np

class SpatialRotation():
    def __init__(self, dimensions: Sequence, k: Sequence = [3], auto_update=True):
        self.dimensions = dimensions
        self.k = k
        self.args = None
        self.auto_update = auto_update
        self.update()

    def update(self):
        self.args = [random.choice(self.k) for dim in self.dimensions]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_update:
            self.update()
        for k, dim in zip(self.args, self.dimensions):
            x = torch.rot90(x, k, dim)
        return x

class SpatialFlip():
    def __init__(self, dims: Sequence, auto_update=True) -> None:
        self.dims = dims
        self.args = None
        self.auto_update = auto_update
        self.update()

    def update(self):
        self.args = tuple(random.sample(self.dims, random.choice(range(len(self.dims)))))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_update:
            self.update()
        x = torch.flip(x, self.args)
        return x

class PadIfNecessary():
    def __init__(self, n_downsampling: int):
        self.mod = 2**n_downsampling

    def __call__(self, x: torch.Tensor):
        padding = []
        for dim in reversed(x.shape[1:]):
            padding.extend([0, (self.mod - dim%self.mod)%self.mod])
        x = F.pad(x, padding)
        return x

    def pad(x, n_downsampling: int = 1):
        mod = 2**n_downsampling
        padding = []
        for dim in reversed(x.shape[1:]):
            padding.extend([0, (mod - dim%mod)%mod])
        x = F.pad(x, padding)
        return x

class ColorJitter3D():
    """
    Randomly change the brightness and contrast an image.
    A grayscale tensor with values between 0-1 and shape BxCxHxWxD is expected.
    Args:
        brightness (float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """
    def __init__(self, brightness_min_max: tuple=None, contrast_min_max: tuple=None) -> None:
        self.brightness_min_max = brightness_min_max
        self.contrast_min_max = contrast_min_max
        self.update()

    def update(self):
        if self.brightness_min_max:
            self.brightness = float(torch.empty(1).uniform_(self.brightness_min_max[0], self.brightness_min_max[1]))
        if self.contrast_min_max:
            self.contrast = float(torch.empty(1).uniform_(self.contrast_min_max[0], self.contrast_min_max[1]))

    def __call__(self, x: torch.Tensor, no_update=False) -> torch.Tensor:
        if not no_update:
            self.update()
        if self.brightness_min_max:
            x = (self.brightness * x).float().clamp(0, 1.).to(x.dtype)
        if self.contrast_min_max:
            mean = torch.mean(x.float(), dim=(-4, -3, -2, -1), keepdim=True)
            x = (self.contrast * x + (1.0 - self.contrast) * mean).float().clamp(0, 1.).to(x.dtype)
        return x

class ColorJitterSphere3D():
    """
    Randomly change the brightness and contrast an image.
    A grayscale tensor with values between 0-1 and shape BxCxHxWxD is expected.
    Args:
        brightness (float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """
    def __init__(self, brightness_min_max: tuple=None, contrast_min_max: tuple=None, sigma: float=1., dims: int=3) -> None:
        self.brightness_min_max = brightness_min_max
        self.contrast_min_max = contrast_min_max
        self.sigma = sigma
        self.dims = dims
        self.update()

    def update(self):
        if self.brightness_min_max:
            self.brightness = float(torch.empty(1).uniform_(self.brightness_min_max[0], self.brightness_min_max[1]))
        if self.contrast_min_max:
            self.contrast = float(torch.empty(1).uniform_(self.contrast_min_max[0], self.contrast_min_max[1]))
        self.ranges = []
        for _ in range(self.dims):
            r = torch.rand(2) * 10 - 5
            self.ranges.append((r.min().item(), r.max().item()))

    def __call__(self, x: torch.Tensor, no_update=False) -> torch.Tensor:
        if not no_update:
            self.update()

        jitterSphere = torch.zeros(1)
        for i,r in enumerate(self.ranges):
            jitterSphere_i = torch.linspace(*r, steps=x.shape[i + 1])
            jitterSphere_i = (1 / (self.sigma * 2.51)) * 2.71**(-0.5 * (jitterSphere_i/self.sigma) ** 2) # Random section of a normal distribution between (-5,5)
            jitterSphere = jitterSphere.unsqueeze(-1) + jitterSphere_i.view(1, *[1]*i, -1)
        jitterSphere /= torch.max(jitterSphere) # Random 3D section of a normal distribution sphere
        
        if self.brightness_min_max:
            brightness = (self.brightness - 1) * jitterSphere + 1
            x = (brightness * x).float().clamp(0, 1.).to(x.dtype)
        if self.contrast_min_max:
            contrast = (self.contrast - 1) * jitterSphere + 1
            mean =x.float().mean()
            x = (contrast * x + (1.0 - self.contrast) * mean).float().clamp(0, 1.).to(x.dtype)
        return x

def getBetterOrientation(nifti: nib.Nifti1Image, axisCode="IPL"):
    orig_ornt = nib.io_orientation(nifti.affine)
    targ_ornt = nib.orientations.axcodes2ornt(axisCode)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    nifti = nifti.as_reoriented(transform)
    return nifti

def toGrayScale(x):
    x_min = np.amin(x)
    x_max = np.amax(x) - x_min
    x = (x - x_min) / x_max
    return x

def center(x, mean, std):
    return (x - mean) / std
