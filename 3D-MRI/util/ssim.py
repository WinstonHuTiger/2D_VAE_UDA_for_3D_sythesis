from typing import Sequence
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size: int, shape: Sequence = [1,1,1]):
    channel = shape[1]
    dim = len(shape)-2
    _1D_window = gaussian(window_size, 1.5)
    window = _1D_window
    _1D_window = _1D_window.unsqueeze(1).t()
    for _ in range(dim-1):
        window = torch.matmul(window.unsqueeze(-1), _1D_window).float()
    return Variable(window.view(1,1,*dim*[window_size]).expand(channel, 1, *dim*[window_size]).contiguous())

def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: Variable, window_size: int, channel: int, size_average = True):
    conv = F.conv1d if img1.dim()==3 else F.conv2d if img1.dim()==4 else F.conv3d
    mu1 = conv(img1, window, padding = window_size//2, groups = channel)
    mu2 = conv(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = conv(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = conv(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = conv(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        channel = img1.shape[1]

        if self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, img1.shape)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    channel = img1.shape[1]
    window = create_window(window_size, img1.shape)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
