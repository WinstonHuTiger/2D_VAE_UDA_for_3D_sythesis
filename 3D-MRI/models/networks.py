import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm3d
from torch.nn.modules.padding import ConstantPad3d
from torch.nn.modules.pooling import AdaptiveMaxPool2d
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import numpy as np
from .stylegan_networks import StyleGAN2Discriminator, StyleGAN2Generator
from models.bayesian import BayesianConv2d, BayesianConv3d, BayesianConvTranspose2d, BayesianConvTranspose3d
from .vae import Encoder 
from .vae_3d import Encoder as Encoder_3d

dimensions = 2
conv = nn.Conv2d
convTranspose = nn.ConvTranspose2d
batchNorm = nn.BatchNorm2d
avgPool = nn.AvgPool2d
adaptiveAvgPool = nn.AdaptiveAvgPool2d
maxPool = nn.MaxPool2d
adaptiveMaxPool = AdaptiveMaxPool2d

convOptions = {
    2: nn.Conv2d,
    3: nn.Conv3d
}
convTransposeOptions = {
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d
}
bayesianConvOptions = {
    2: BayesianConv2d,
    3: BayesianConv3d
}
bayesianConvTransposeOptions = {
    2: BayesianConvTranspose2d,
    3: BayesianConvTranspose3d
}

avgPoolOptions = {
    2: nn.AvgPool2d,
    3: nn.AvgPool3d
}

adaptiveAvgPoolOptions = {
    2: nn.AdaptiveAvgPool2d,
    3: nn.AdaptiveAvgPool3d
}

maxPoolOptions = {
    2: nn.MaxPool2d,
    3: nn.MaxPool3d
}

adaptiveMaxPoolOptions = {
    2: nn.AdaptiveMaxPool2d,
    3: nn.AdaptiveMaxPool3d
}

batchNormOptions = {
    2: BatchNorm2d,
    3: BatchNorm3d
}

def setDimensions(dim: int=-1, bayesian: bool = False):
    global dimensions, conv, convTranspose, batchNorm, avgPool, adaptiveAvgPool, maxPool, adaptiveMaxPool
    if dim == -1:
        dim = dimensions
    else:
        dimensions = dim
    if bayesian:
        conv = bayesianConvOptions[dimensions]
        convTranspose = bayesianConvTransposeOptions[dimensions]
    else:
        conv = convOptions[dimensions]
        convTranspose = convTransposeOptions[dimensions]
    batchNorm = batchNormOptions[dimensions]
    avgPool = avgPoolOptions[dimensions]
    adaptiveAvgPool = adaptiveAvgPoolOptions[dimensions]
    maxPool = maxPoolOptions[dimensions]
    adaptiveMaxPool = adaptiveMaxPoolOptions[dimensions]




###############################################################################
# Helper Functions
###############################################################################


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    if dimensions==2:
        filt = torch.Tensor(a[:, None] * a[None, :])
    else:
        filt = torch.Tensor(a[:,None, None] * a[None,:,None] * a[None,None,:])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))] * dimensions
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        if dimensions == 2:
            self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        else:
            self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                if dimensions == 2:
                    return inp[:, :, ::self.stride, ::self.stride]
                else:
                    return inp[:, :, ::self.stride, ::self.stride, ::self.stride]
            else:
                if dimensions == 2:
                    return self.pad(inp)[:, :, ::self.stride, ::self.stride]
                else:
                    return self.pad(inp)[:, :, ::self.stride, ::self.stride, ::self.stride]
        else:
            if dimensions == 2:
                return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
            else:
                return F.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        if dimensions == 2:
            self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        else:
            self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1]*dimensions)

    def forward(self, inp):
        if dimensions == 2:
            ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        else:
            ret_val = F.conv_transpose3d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            if dimensions == 2:
                return ret_val[:, :, :-1, :-1]
            else:
                return ret_val[:, :, :-1, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        if dimensions == 2:
            PadLayer = nn.ReflectionPad2d
        elif hasattr(nn, 'ReflectionPad3d'):
            PadLayer = nn.ReflectionPad3d
        else:
            PadLayer = ZeroPad3d
    elif(pad_type in ['repl', 'replicate']):
        if dimensions == 2:
            PadLayer = nn.ReplicationPad2d
        else:
            PadLayer = nn.ReplicationPad3d
    elif(pad_type == 'zero'):
        if dimensions == 2:
            PadLayer = nn.ZeroPad2d
        else:
            PadLayer = ZeroPad3d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class ZeroPad3d(ConstantPad3d):
    def __init__(self, padding: tuple) -> None:
        super(ZeroPad3d, self).__init__(padding, 0.)


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        if dimensions == 2:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        else:
            norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        if dimensions == 2:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        else:
            norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, debug=False, nonlinearity='leaky_relu'):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity=nonlinearity)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1: # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True, nonlinearity='leaky_relu'):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug, nonlinearity=nonlinearity)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    setDimensions(bayesian=opt.bayesian)
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=opt.ngl, opt=opt)
    elif netG == 'obelisk':
        net = ObeliskHybridGenerator(output_nc)
    elif netG == 'sit':
        net = SIT((input_nc,opt.ngf,output_nc), norm_layer=norm_layer)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'stylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, opt=opt)
    elif netG == 'smallstylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=2, opt=opt)
    elif netG == 'resnet_cat':
        n_blocks = 8
        net = G_Resnet(input_nc, output_nc, opt.nz, num_downs=2, n_res=n_blocks - 4, ngf=ngf, norm='inst', nl_layer='relu')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    setDimensions(bayesian=False)
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG), nonlinearity='relu')


def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids, nonlinearity='relu')


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leaky RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias,)
    elif netD =='obelisk':
        net = ObeliskDiscriminator(input_nc)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias,)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD =="encoder":  # default Encoder from slice 3D paper
        net = Encoder(n_channels=1, gf_dim = 4)
    elif netD == "encoder_3d":
        net = Encoder_3d(n_channels= 1, gf_dim = 4)
    elif 'stylegan2' in netD:
        net = StyleGAN2Discriminator(input_nc, ndf, n_layers_D, no_antialias=no_antialias, opt=opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids,
                    initialize_weights=('stylegan2' not in netD))


##############################################################################
# Classes
##############################################################################

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=adaptiveMaxPool(1)
        self.avgpool=adaptiveAvgPool(1)
        self.se=nn.Sequential(
            conv(channel,channel//reduction,1,bias=False),
            nn.Hardswish(True),
            conv(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=conv(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, conv):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, batchNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual
    

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, dtype=torch.float32):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label, dtype=dtype))
        self.register_buffer('fake_label', torch.tensor(target_fake_label, dtype=dtype))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power).type(torch.float16)
        out: torch.Tensor = x.div(norm + 1e-7)
        return out


class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)


class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(conv(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(conv(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def forward(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = F.instance_norm(x)
        return self.l2_norm(x)


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            if feat.dim() == 5:
                B, H, W, D = feat.shape[0], feat.shape[2], feat.shape[3], feat.shape[4]
                feat_reshape = feat.permute(0, 2, 3, 4, 1).flatten(1, 3)
            else:
                B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            with torch.cuda.amp.autocast(enabled=False):
                t = x_sample.dtype
                x_sample = x_sample.type(torch.float32)
                if self.use_mlp:
                    mlp = getattr(self, 'mlp_%d' % feat_id)
                    x_sample = mlp(x_sample)
                return_ids.append(patch_id)
                x_sample = self.l2norm(x_sample)
                x_sample = x_sample.type(t)

            if num_patches == 0:
                if feat.dim() == 5:
                    x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W, D])
                else:
                    x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class G_Resnet(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, n_res, ngf=64,
                 norm=None, nl_layer=None):
        super(G_Resnet, self).__init__()
        n_downsample = num_downs
        pad_type = 'reflect'
        self.enc_content = ContentEncoder(n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type)
        if nz == 0:
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)
        else:
            self.dec = Decoder_all(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)

    def decode(self, content, style=None):
        return self.dec(content, style)

    def forward(self, image, style=None, nce_layers=[], encode_only=False):
        content, feats = self.enc_content(image, nce_layers=nce_layers, encode_only=encode_only)
        if encode_only:
            return feats
        else:
            images_recon = self.decode(content, style)
            if len(nce_layers) > 0:
                return images_recon, feats
            else:
                return images_recon

##################################################################################
# Encoder and Decoders
##################################################################################


class E_adaIN(nn.Module):
    def __init__(self, input_nc, output_nc=1, nef=64, n_layers=4,
                 norm=None, nl_layer=None, vae=False):
        # style encoder
        super(E_adaIN, self).__init__()
        self.enc_style = StyleEncoder(n_layers, input_nc, nef, output_nc, norm='none', activ='relu', vae=vae)

    def forward(self, image):
        style = self.enc_style(image)
        return style


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        if self.vae:
            self.fc_mean = nn.Linear(dim, style_dim)  # , 1, 1, 0)
            self.fc_var = nn.Linear(dim, style_dim)  # , 1, 1, 0)
        else:
            self.model += [conv(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        if self.vae:
            output = self.model(x)
            output = output.view(x.size(0), -1)
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            return self.model(x).view(x.size(0), -1)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x, nce_layers=[], encode_only=False):
        if len(nce_layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in nce_layers:
                    feats.append(feat)
                if layer_id == nce_layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            return self.model(x), None

        for layer_id, layer in enumerate(self.model):
            print(layer_id, layer)


class Decoder_all(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder_all, self).__init__()
        # AdaIN residual blocks
        self.resnet_block = ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
        self.n_blocks = 0
        # upsampling blocks
        for i in range(n_upsample):
            block = [Upsample2(scale_factor=2), Conv2dBlock(dim + nz, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            setattr(self, 'block_{:d}'.format(self.n_blocks), nn.Sequential(*block))
            self.n_blocks += 1
            dim //= 2
        # use reflection padding in the last conv layer
        setattr(self, 'block_{:d}'.format(self.n_blocks), Conv2dBlock(dim + nz, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect'))
        self.n_blocks += 1

    def forward(self, x, y=None):
        if y is not None:
            output = self.resnet_block(cat_feature(x, y))
            for n in range(self.n_blocks):
                block = getattr(self, 'block_{:d}'.format(n))
                if n > 0:
                    output = block(cat_feature(output, y))
                else:
                    output = block(output)
            return output


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
        # upsampling blocks
        for i in range(n_upsample):
            if i == 0:
                input_dim = dim + nz
            else:
                input_dim = dim
            self.model += [Upsample2(scale_factor=2), Conv2dBlock(input_dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, y=None):
        if y is not None:
            return self.model(cat_feature(x, y))
        else:
            return self.model(x)

##################################################################################
# Sequential Models
##################################################################################


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat


class ResBlock(nn.Module):
    def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = get_pad_layer('refl')(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = conv(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d or norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d or norm_layer == nn.InstanceNorm3d)

        model = [get_pad_layer('refl')(3),
                 conv(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = opt.n_downsampling
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [conv(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [conv(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [convTranspose(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          conv(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [get_pad_layer('refl')(3)]
        model += [conv(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=None, encode_only=False):
        if layers is not None:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            if encode_only:
                return feats
            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class ResnetDecoder(nn.Module):
    """Resnet-based decoder that consists of a few Resnet blocks + a few upsampling operations.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based decoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if(no_antialias):
                model += [convTranspose(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          conv(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [get_pad_layer('refl')(3)]
        model += [conv(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetEncoder(nn.Module):
    """Resnet-based encoder that consists of a few downsampling + several Resnet blocks
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based encoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [get_pad_layer('refl')(3),
                 conv(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [conv(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [conv(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [get_pad_layer('refl')(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [conv(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [get_pad_layer('refl')(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = conv(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = convTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = convTranspose(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = convTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), norm_layer(ndf), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [conv(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), norm_layer(ndf), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            conv(ndf * nf_mult_prev, ndf * nf_mult_prev, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult_prev),
            nn.LeakyReLU(0.2, True),
            conv(ndf * nf_mult_prev, ndf * nf_mult_prev, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult_prev),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [conv(ndf * nf_mult_prev, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input, layers=None, encode_only=False):
        """Standard forward."""
        if not layers:
            layers = []
        feat = input
        feats = []
        for layer_id, layer in enumerate(self.model):
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            if encode_only and layer_id == layers[-1]:
                return feats
        if len(layers) > 0:
            return feat, feats
        else:
            return feat


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            conv(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            conv(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            conv(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator(NLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        size = 16
        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().forward(input)


class GroupedChannelNorm(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        shape = list(x.shape)
        new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
        x = x.view(*new_shape)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x_norm = (x - mean) / (std + 1e-7)
        return x_norm.view(*shape)


class ObeliskLayer(nn.Module):
    """
    Taken from https://github.com/mattiaspaul/OBELISK
    Defines the OBELISK layer that performs deformable convolution with the given number of trainable spatial offsets and a 5 layer 1x1 Dense-Net. The tensor after this layer will be interpolated the input tensor shape using trilinear interpolation
    """
    def __init__(self, C: tuple, down_scale_factor: int=1, K: int = 128, denseNetLayers: int = 4, upscale=True, activation=nn.ReLU):
        """
        Creates an OBELISK layer that performs deformable convolution with the given number of trainable spatial offsets and a 5 layer 1x1 Dense-Net. The tensor after this layer will be interpolated the input tensor shape using trilinear interpolation

        Parameters:
        ----------
            - C_out (int): Number of Output channels
            - full_res: list
        """
        super().__init__()
        C_in, C_mid, C_out = C
        self.down_scale_factor = down_scale_factor
        self.grid_initialized_with = 0
        self.sample_grid = None
        self.upscale = upscale
        self.denseNetLayers = denseNetLayers
        norm=get_norm_layer('instance')
        
        # Obelisk N=1 variant offsets: 1x #offsets x1xNx3
        self.offset = nn.Parameter(torch.randn(1,K,*[1]*(dimensions-1),dimensions)*0.05)

        # Dense-Net with 1x1x1 kernels
        self.conv1 = nn.Sequential(conv(C_in*K,C_mid,1,groups=4,bias=False), activation(True))
        # self.conv2 = nn.Sequential(batchNorm(128), conv(128,32,1,bias=False), nn.ReLU(True))
        self.denseNet = nn.ModuleList([])
        C = C_mid
        for i in range(denseNetLayers):
            self.denseNet.append(nn.Sequential(norm(C), conv(C,32,1,bias=False), activation(True)))
            C+=32
        self.conv3 = nn.Sequential(norm(C), conv(C,C_out,1,bias=False), activation(True))

    def create_grid(self, quarter_res, device):
        grid_base = [2,3] if dimensions==2 else [3,4]
        # Obelisk sample_grid: 1 x 1 x #samples x 1 x 3
        self.sample_grid = F.affine_grid(torch.eye(*grid_base, device=device).unsqueeze(0), torch.Size((1,1,*quarter_res)), align_corners=False).view(1,1,-1,*[1]*(dimensions-2),dimensions).detach()
        self.sample_grid.requires_grad = False
        self.grid_initialized_with = quarter_res

    def forward(self, x: torch.Tensor):
        half_res = list(map(lambda x: int(x/(self.down_scale_factor*2)), x.shape[2:]))
        quarter_res = list(map(lambda x: int(x/(self.down_scale_factor*4)), x.shape[2:]))

        if self.grid_initialized_with != quarter_res:
            self.create_grid(quarter_res, x.device)
        B = x.size()[0]

        # Obelisk Layer
        x = F.grid_sample(x, (self.sample_grid.repeat(B,1,*[1]*dimensions) + self.offset), align_corners=False).view(B,-1,*quarter_res)
        x = self.conv1(x)

        # Dense-Net with 1x1x1 kernels
        for i in range(self.denseNetLayers):
            x = torch.cat([x, self.denseNet[i](x)], dim=1)
        x = self.conv3(x)
        if self.upscale:
            x = F.interpolate(x, size=half_res, mode='trilinear' if dimensions==3 else 'bilinear', align_corners=False)
        return x

class SIT(nn.Module):
    """
    The Scout-Identify-Transform network for modality translation.
    """
    def __init__(self, C: tuple, factor=4, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        C_in, C_mid, C_out = C
        self.factor = factor
        self.layers = nn.ModuleList([])
        self.layers_in = nn.Sequential()
        self.weightedSum = nn.ModuleList([])
        for i in range(1, factor):
            self.layers_in.add_module(str(i-1), nn.Sequential(
                conv(C_in if i==1 else C_mid , C_mid, kernel_size=4, stride=2, padding=1),
                norm_layer(C_mid),
                nn.ReLU(True)
            ))
            if i < factor-1:
                combine_module = [conv(2*C_mid,C_mid, kernel_size=1)]
            else:
                combine_module = []
            if i==1:
                self.layers.append(
                    nn.Sequential(
                        *combine_module,
                        DenseShortut(C_mid, norm_layer=norm_layer),
                        norm_layer(C_mid),
                        nn.ReLU(True),
                        DenseShortut(C_mid, norm_layer=norm_layer),
                        norm_layer(C_mid),
                        nn.ReLU(True),
                        convTranspose(C_mid, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(8),
                        nn.ReLU(True),
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        *combine_module,
                        DenseShortut(C_mid, norm_layer=norm_layer),
                        norm_layer(C_mid),
                        nn.ReLU(True),
                        convTranspose(C_mid, C_mid, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(C_mid),
                        nn.ReLU(True),
                    )
                )
    
        self.out = nn.Sequential(
            norm_layer(8),
            nn.ReLU(True),
            conv(8, C_out, 3, padding=1),
            nn.Sigmoid())

    def forward(self, input, layers=None, encode_only=False):
        if not layers:
            layers = []
        feats = []
        layer_id = 0

        # Downsample
        r_in = []
        for l_i_in in self.layers_in:
            input = l_i_in[:-2](input)
            if layer_id in layers:
                feats.append(input)
                if encode_only and layer_id == layers[-1]:
                    return feats
            layer_id += 1
            input = l_i_in[-2:](input)
            r_in.append(input)

        # Upsample
        r_out = None
        for i in range(self.factor-1, 0, -1):
            l_i = self.layers[i-1]
            r = r_in.pop()
            if r_out is not None:
                r = torch.concat([r, r_out], dim=1)
            r = l_i[:-2](r)
            if layer_id in layers:
                feats.append(r)
                if encode_only and layer_id == layers[-1]:
                    return feats
            layer_id += 1
            r = l_i[-2:](r)
            r_out = r
            r = None
        r_out = self.out(r_out)

        if len(layers)==0:
            return r_out
        return r_out, feats
    
        

#         Parameters:
#         ----------
#             - C_out (int): Number of Output channels
#             - full_res: list
#         """
#         super().__init__()
#         self.C_in, self.C_out = C
#         self.group_size = 4
#         self.grid_initialized_with = 0
#         self.sample_grid = None
#         self.stride = 2**(stride-1)
        
#         # Obelisk N=1 variant offsets: 1x #offsets x1xNx3
#         self.offset = nn.Parameter(torch.randn(1,K,*[1]*(dimensions-1),dimensions)*0.05)

#         self.conv1 = nn.ModuleList(
#             conv(K, self.C_out, kernel_size=1) for _ in range(self.C_in // self.group_size)
#         )

#     def create_grid(self, quarter_res, device):
#         grid_base = [2,3] if dimensions==2 else [3,4]
#         # Obelisk sample_grid: 1 x 1 x #samples x 1 x 3
#         self.sample_grid = F.affine_grid(torch.eye(*grid_base, device=device).unsqueeze(0), torch.Size((1,1,*quarter_res))).view(1,1,-1,*[1]*(dimensions-2),dimensions).detach()
#         self.sample_grid.requires_grad = False
#         self.grid_initialized_with = quarter_res

#     def forward(self, x: torch.Tensor):
#         S = list(torch.tensor(x.shape[2:]) // self.stride)
#         if self.grid_initialized_with != S:
#             self.create_grid(S, x.device)
#         B = x.size()[0]

#         # Obelisk Layer
#         y = 0
#         for i in range(self.C_in):
#             feature_map = F.grid_sample(x[:,i:i+1], (self.sample_grid.repeat(B,1,*[1]*dimensions) + self.offset), align_corners=True).view(B,-1,*S)
#             y = y + self.conv1[i//self.group_size](feature_map)
#         return y

class DSNetBlock(nn.Module):
    def __init__(self, block: nn.Module, norm: nn.Module, C: int, skip = True):
        super().__init__()
        self.block = block
        self.norm = norm
        self.weighted_sum = conv(C+1, 1, kernel_size=1)
        self.C = C
        self.skip = skip

    def forward(self, x: torch.Tensor, skip_connections = None):
        if skip_connections is None:
            skip_connections = [x]
        y = self.block(x)
        y = self.weighted_sum(
            torch.concat([y,*skip_connections], 1).view(x.shape[0], self.C+1,*[1]*(dimensions-1), -1)
            ).view(x.shape)
        skip_connections.append(self.norm(y))
        if self.skip:
            return y, skip_connections
        return y

class DenseShortut(nn.Module):
    def __init__(self, C: int, block1=None, block2=None, norm_layer=nn.InstanceNorm2d) -> None:
        super().__init__()
        self.block1 = block1 or conv(C, C, kernel_size=3, padding=1)
        self.norm1 =  nn.Sequential(norm_layer(C),nn.ReLU(True))
        self.weightedSum1 = Parameter(torch.normal(mean=0, std=1, size=(1,C,*[1]*dimensions)))
        self.block2 = block2 or conv(C, C, kernel_size=3, padding=1)
        self.norm2 =  nn.Sequential(norm_layer(C),nn.ReLU(True))
        self.weightedSum2_1 = Parameter(torch.normal(mean=0, std=1, size=(1,C,*[1]*dimensions)))
        self.weightedSum2_2 = Parameter(torch.normal(mean=0, std=1, size=(1,C,*[1]*dimensions)))
        self.block3 = conv(C, C, kernel_size=3, padding=1)
        self.norm3 =  nn.Sequential(norm_layer(C),nn.ReLU(True))
        self.weightedSum3_1 = Parameter(torch.normal(mean=0, std=1, size=(1,C,*[1]*dimensions)))
        self.weightedSum3_2 = Parameter(torch.normal(mean=0, std=1, size=(1,C,*[1]*dimensions)))
        self.weightedSum3_3 = Parameter(torch.normal(mean=0, std=1, size=(1,C,*[1]*dimensions)))
    
    def forward(self, x):
        x0 = self.norm1(x)
        x1 = self.block1(x0) + x0 * self.weightedSum1
        x1 = self.norm2(x1)
        x2 = self.block2(x1) + x0 * self.weightedSum2_1 + x1 * self.weightedSum2_2
        x2 = self.norm3(x2)
        x = self.block3(x2) + x0 * self.weightedSum3_1 + x1 * self.weightedSum3_2 + x2 * self.weightedSum3_3
        return x

class DUNe(nn.Module):
    def __init__(self, C: tuple, ngl: int = 3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        C_in, C_mid, C_out = C
        self.ngl = ngl
        model = [
            conv(C_in, 16, kernel_size=3, stride=2, padding=1),
            norm_layer(16),
            nn.ReLU(True),
            conv(16+1, 16, kernel_size=3, padding=1),
            norm_layer(16),
            nn.ReLU(True),
            conv(16, 32, kernel_size=3, stride=2, padding=1),
            norm_layer(32),
            nn.ReLU(True),
        ]
        num_skip = 1+16
        for i in range(1,ngl+1):       # add DSNet blocks
            model += [
                conv(32*i + num_skip, 32*(i+1), kernel_size=3, padding=1),
                norm_layer(C_mid),
                nn.ReLU(True),
            ]
            num_skip += 32*i

        model += [
            conv(32*(ngl+1) + num_skip, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(True),
            convTranspose(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(32),
            nn.ReLU(True),
            conv(32+1+16+64, 32, kernel_size=3, padding=1),
            norm_layer(32),
            nn.ReLU(True),
            convTranspose(32, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(8),
            nn.ReLU(True),
            conv(8+1+32, C_out, kernel_size=3, padding=1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input, layers=None, encode_only=False):
        if not layers:
            layers = []
        feat = input
        feats = []
        d_1 = F.interpolate(input, scale_factor=0.5)
        d_2 = F.interpolate(input, scale_factor=0.25)
        d_0_skip = [input]
        d_1_skip = [d_1]
        d_2_skip = [d_2]
        for layer_id, layer in enumerate(self.model):
            if (layer_id == 3):
                feat = torch.concat([feat, d_1], dim=1)
            elif (layer_id==6):
                d_1_skip.append(feat)
                d_2_skip.append(F.interpolate(feat, scale_factor=0.5))
            elif (layer_id >=9 and layer_id <=9+self.ngl*3 and layer_id%3==0):
                d_2_skip.append(feat)
                feat = torch.concat([*d_2_skip], dim=1)
            elif (layer_id==12+self.ngl*3):
                d_1_skip.append(F.interpolate(feat, scale_factor=2))
            elif (layer_id==15+self.ngl*3):
                feat = torch.concat([feat, *d_1_skip], dim=1)
            elif (layer_id==18+self.ngl*3):
                d_0_skip.append(F.interpolate(feat, scale_factor=2))
            elif (layer_id==21+self.ngl*3):
                feat = torch.concat([feat, *d_0_skip], dim=1)
            feat = layer(feat)
            if layer_id in layers:
                # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                feats.append(feat)
            else:
                # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                pass
            if encode_only and layer_id == layers[-1]:
                # print('encoder only return features')
                return feats  # return intermediate features alone; stop in the last layers
        if len(layers)==0:
            return feat
        return feat, feats

class ObeliskDiscriminator(nn.Module):
    def __init__(self, C_in):
        super().__init__()
        norm = get_norm_layer('instance')
        self.model = nn.Sequential(
            conv(C_in, 64, kernel_size=3, stride=2, padding=1),
            norm(64),
            nn.LeakyReLU(0.05),
            ObeliskLayer((64,128,128), down_scale_factor=1, K=128,
                        upscale=False, activation=nn.LeakyReLU),
            norm(128),
            nn.LeakyReLU(0.05),
            conv(128, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x

class ObeliskHybridGenerator(nn.Module):
    """
    Taken from https://github.com/mattiaspaul/OBELISK
    Hybrid OBELISK CNN model that contains two obelisk layers combined with traditional CNNs the layers have 512 and 128 trainable offsets and 230k trainable weights in total
    """
    def __init__(self, C_out: int, cbam=False):
        super().__init__()

        norm = get_norm_layer('instance')
        activation = lambda: nn.ReLU(True)
        self.activation = activation()
        self.cbam = cbam
        
        def get_cbam(*x):
            if self.cbam:
                return [CBAMBlock(x)]
            else:
                return []

        #U-Net Encoder
        leakage = 0.05
        self.conv2 = conv(16, 32, 3, stride=2, padding=1)
        self.batch2 = norm(32)

        # self.conv8 = conv(32, C_out, 1)
        self.conv8 = convTranspose(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        obelisk1 = ObeliskLayer((1,128,32), down_scale_factor=2, K=512)
        obelisk2 = ObeliskLayer((4,128,32), down_scale_factor=1, K=128)
        self.model0 = nn.Sequential(avgPool(3, padding=1, stride=1), obelisk1)
        self.norm_act0 = nn.Sequential(norm(32), activation(), *get_cbam(32, 2))
        self.model1 = nn.Sequential(conv(1, 4, 3, padding=1))
        self.norm_act1 = nn.Sequential(norm(4), activation())
        self.model10 = nn.Sequential(obelisk2, activation())
        self.norm_act10 = nn.Sequential(norm(32), activation(), *get_cbam(32, 2))
        self.model11 = nn.Sequential(
            conv(4, 16, 3, stride=2, padding=1),
            norm(16),
            activation(),
            *get_cbam(16, 2),
            conv(16, 16, 3, padding=1)
        )
        self.norm_act11 = nn.Sequential(norm(16), activation(),*get_cbam(16, 2))
        self.model110 = nn.Sequential(conv(16, 32, 3, stride=2, padding=1))
        self.norm_act110 = nn.Sequential(norm(32), activation(),*get_cbam(32, 2))
        self.output1 = nn.Sequential(
            conv(64, 32, 3, padding=1),
            norm(32),
            activation(),
            *get_cbam(32, 2)
        )
        self.output2 = nn.Sequential(
            conv(64+16, 32, 3, padding=1),
            norm(32),
            activation(),
            *get_cbam(32, 2)
        )

    def forward(self, x: torch.Tensor, layers=None, encode_only=False):
        size = x.size()
        half_res = list(map(lambda x: int(x/2), size[2:]))
        all_feats=[]

        x0 = self.model0(x)
        all_feats.append(x0)
        x0 = self.norm_act0(x0)
        x1 = self.model1(x)
        all_feats.append(x1)
        x1 = self.norm_act1(x1)
        x10 = self.model10(x1)
        all_feats.append(x10)
        x10 = self.norm_act10(x10)
        x11 = self.model11(x1)
        all_feats.append(x11)
        x11 = self.norm_act11(x11)
        x110 = self.model110(x11)
        all_feats.append(x110)
        x110 = self.norm_act110(x110)

        if layers is not None:
            feats = []
            for i,feat in enumerate(all_feats):
                if i in layers:
                    feats.append(feat)
            if encode_only:
                return feats

        #unet-decoder
        x = self.output1(torch.cat((x0,x110),1))
        x = F.interpolate(x, size=half_res, mode='trilinear' if dimensions==3 else 'bilinear', align_corners=False)
        x = self.output2(torch.cat((x,x10,x11),1))
        x = self.conv8(x)
        x = torch.sigmoid(x)

        if layers is not None:
            return x, feats
        return x

#Hybrid OBELISK CNN model that contains two obelisk layers combined with traditional CNNs
#the layers have 512 and 128 trainable offsets and 230k trainable weights in total
#trained with pytorch v1.0 for VISCERAL data
class obeliskhybrid_visceral(nn.Module):
    def __init__(self, num_labels: int):
        super(obeliskhybrid_visceral, self).__init__()
        self.num_labels = num_labels
        self.initialized = False

        #U-Net Encoder
        self.conv0 = conv(1, 4, 3, padding=1)
        self.batch0 = batchNorm(4)
        self.conv1 = conv(4, 16, 3, stride=2, padding=1)
        self.batch1 = batchNorm(16)
        self.conv11 = conv(16, 16, 3, padding=1)
        self.batch11 = batchNorm(16)
        self.conv2 = conv(16, 32, 3, stride=2, padding=1)
        self.batch2 = batchNorm(32)
        
        # Obelisk Encoder (for simplicity using regular sampling grid)
        # the first obelisk layer has 128 the second 512 trainable offsets
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3
        
        self.offset1 = nn.Parameter(torch.randn(1,128,1,1,3)*0.05)
        self.linear1a = conv(4*128,128,1,groups=4,bias=False)
        self.batch1a = batchNorm(128)
        self.linear1b = conv(128,32,1,bias=False)
        self.batch1b = batchNorm(128+32)
        self.linear1c = conv(128+32,32,1,bias=False)
        self.batch1c = batchNorm(128+64)
        self.linear1d = conv(128+64,32,1,bias=False)
        self.batch1d = batchNorm(128+96)
        self.linear1e = conv(128+96,32,1,bias=False)
        
        self.offset2 = nn.Parameter(torch.randn(1,512,1,1,3)*0.05)
        self.linear2a = conv(512,128,1,groups=4,bias=False)
        self.batch2a = batchNorm(128)
        self.linear2b = conv(128,32,1,bias=False)
        self.batch2b = batchNorm(128+32)
        self.linear2c = conv(128+32,32,1,bias=False)
        self.batch2c = batchNorm(128+64)
        self.linear2d = conv(128+64,32,1,bias=False)
        self.batch2d = batchNorm(128+96)
        self.linear2e = conv(128+96,32,1,bias=False)
        
        #U-Net Decoder 
        self.conv6bU = conv(64, 32, 3, padding=1)
        self.batch6bU = batchNorm(32)
        self.conv6U = conv(64+16, 32, 3, padding=1)
        self.batch6U = batchNorm(32)
        self.conv8 = conv(32, num_labels, 1)
        
    def forward(self, inputImg: torch.Tensor, layers=[], encode_only=False):
        half_res = list(map(lambda x: int(x/2), inputImg.shape[2:]))
        quarter_res = list(map(lambda x: int(x/4), inputImg.shape[2:]))
        eighth_res = list(map(lambda x: int(x/8), inputImg.shape[2:]))

        # if not self.initialized:
        self.sample_grid1 = F.affine_grid(torch.eye(3,4, device=inputImg.device).unsqueeze(0),torch.Size((1,1,*quarter_res))).view(1,1,-1,1,3).detach()
        self.sample_grid1.requires_grad = False
        self.sample_grid2 = F.affine_grid(torch.eye(3,4, device=inputImg.device).unsqueeze(0),torch.Size((1,1,*eighth_res))).view(1,1,-1,1,3).detach()
        self.sample_grid2.requires_grad = False
            # self.initialized = True
    
        B,C,D,H,W = inputImg.size()
        leakage = 0.05 #leaky ReLU used for conventional CNNs
        
        #unet-encoder
        x00 = F.avg_pool3d(inputImg,3,padding=1,stride=1)
        
        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), leakage)
        x = F.leaky_relu(self.batch1(self.conv1(x1)),leakage)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)),leakage)
        x = F.leaky_relu(self.batch2(self.conv2(x2)),leakage)
        
        #in this model two obelisk layers with fewer spatial offsets are used
        #obelisk layer 1
        x_o1 = F.grid_sample(x1, (self.sample_grid1.repeat(B,1,1,1,1) + self.offset1)).view(B,-1,*quarter_res)
        #1x1 kernel dense-net
        x_o1 = F.relu(self.linear1a(x_o1))
        x_o1a = torch.cat((x_o1,F.relu(self.linear1b(self.batch1a(x_o1)))),dim=1)
        x_o1b = torch.cat((x_o1a,F.relu(self.linear1c(self.batch1b(x_o1a)))),dim=1)
        x_o1c = torch.cat((x_o1b,F.relu(self.linear1d(self.batch1c(x_o1b)))),dim=1)
        x_o1d = F.relu(self.linear1e(self.batch1d(x_o1c)))
        x_o1 = F.interpolate(x_o1d, size=[*half_res], mode='trilinear', align_corners=False)
        
        #obelisk layer 2
        x_o2 = F.grid_sample(x00, (self.sample_grid2.repeat(B,1,1,1,1) + self.offset2)).view(B,-1,*eighth_res)
        x_o2 = F.relu(self.linear2a(x_o2))
        #1x1 kernel dense-net
        x_o2a = torch.cat((x_o2,F.relu(self.linear2b(self.batch2a(x_o2)))),dim=1)
        x_o2b = torch.cat((x_o2a,F.relu(self.linear2c(self.batch2b(x_o2a)))),dim=1)
        x_o2c = torch.cat((x_o2b,F.relu(self.linear2d(self.batch2c(x_o2b)))),dim=1)
        x_o2d = F.relu(self.linear2e(self.batch2d(x_o2c)))
        x_o2 = F.interpolate(x_o2d, size=[*quarter_res], mode='trilinear', align_corners=False)

        if encode_only:
            all_feats = [x_o2, x1, x_o1, x2, x]
            feats = []
            for i,feat in enumerate(all_feats):
                if i in layers:
                    feats.append(feat)
            return feats

        #unet-decoder
        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x,x_o2),1))),leakage)
        x = F.interpolate(x, size=[*half_res], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x,x_o1,x2),1))),leakage)
        x = F.interpolate(self.conv8(x), size=[D,H,W], mode='trilinear', align_corners=False)
        x = torch.sigmoid(x)
        return x