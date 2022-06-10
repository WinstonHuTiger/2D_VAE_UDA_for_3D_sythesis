import torch
from .base_model import BaseModel
from . import networks
import numpy as np
from util.crop_and_pad_volume import crop_or_pad_volume_to_size_along_x, \
crop_or_pad_volume_to_size_along_y, crop_or_pad_volume_to_size_along_z, normalise_image, percentile
from torch.distributions import normal, kl

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan', paired=True)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument("--netVae_path", type = str, default = None, help = "the path of self-defined disriminator")
            parser.add_argument("--ele_percent", type = int, default = 90, help = "the percentile of elements' kl div")
            parser.add_argument("--use_3d_vae", action='store_true', help="whether or not using the 3d vae for for kl div")
            parser.add_argument("--freeze_sit_layers_in", action='store_true', help="freeze the sit's layers_in parameters during domain apdaption")
        parser.add_argument('--lambda_perceptual', type=float, default=1.5, help="weight for perceptual loss")

        return parser

    def __init__(self, opt):
        def freeze_sit_layers_in(model):
            for para in model.layers_in.parameters():
                para.requires_grad = False
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        if opt.lambda_perceptual != 0:
            self.loss_names.append('G_perceptual')
        
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            if self.opt.netVae_path is not None:
                self.loss_names.append("kl")
            else:
                if opt.dataset_mode == "brain_3D_transfer":
                    self.loss_names.append("fine_tuned")
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.networks = [self.netG]
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            
            if self.opt.netVae_path is not None:
                encoder_name = None
                if not self.opt.use_3d_vae:
                    encoder_name = "encoder"
                else:
                    encoder_name = "encoder_3d"
                
                self.netVae = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, encoder_name,
                                          opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, 
                                          opt.no_antialias, self.gpu_ids, opt)
            
                self.load_netVae()
                self.networks.extend([self.netVae])
            
            self.networks.extend([self.netD])
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, dtype=torch.float16 if opt.amp else torch.float32).to(self.device)
            l1 = torch.nn.L1Loss()
            self.perceptual_loss = torch.nn.L1Loss()

            self.criterionL1 = lambda i1,i2: l1(i1,i2)
            if self.opt.freeze_sit_layers_in:
                freeze_sit_layers_in(self.netG)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.glr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.dlr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        if self.registration_artifacts_idx is not None:
            self.fake_B = self.fake_B * self.registration_artifacts_idx.to(self.fake_B.device)
    
    def load_netVae(self):
        netVae = self.netVae
        if isinstance(netVae, torch.nn.DataParallel):
            netVae = netVae.module
        if self.opt.use_3d_vae:
            print('loading 3d vae model from %s' % self.opt.netVae_path)
        else:
            print('loading vae model from %s' % self.opt.netVae_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(self.opt.netVae_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        netVae.load_state_dict(state_dict["encoder"])
    
    def convert_image_to_256_size(self, x):
        # x, [1, 1, 144, 192, 192]
        
        #batch_size = 1
        #channels = 1
        shape = x.shape
        x = x.view((shape[0], shape[1], shape[3], shape[4], shape[2]))
        x = x.squeeze()
        x = crop_or_pad_volume_to_size_along_x(x, 256)
        x = crop_or_pad_volume_to_size_along_y(x, 256)
        x = crop_or_pad_volume_to_size_along_z(x, 256)
        return torch.unsqueeze(x, 1)
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            del fake_AB, pred_fake, real_AB, pred_real
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.scaler.scale(self.loss_D).backward()

    def backward_G_wo_l1(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake, feats_fake = self.netD(fake_AB, layers=[0, 3, 6, 9])
            self.loss_fine_tuned = self.criterionGAN(pred_fake, True)
 
        self.scaler.scale(self.loss_fine_tuned).backward()
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake, feats_fake = self.netD(fake_AB, layers=[0, 3, 6, 9])
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            self.loss_G_perceptual = torch.tensor(0, device=self.opt.gpu_ids[0], dtype=torch.float16)
            if self.opt.lambda_perceptual!=0:
                feats_real = self.netD(torch.cat((self.real_A, self.real_B), 1).detach(), layers=[0, 3, 6, 9], encode_only=True)
                for i, λ_i in enumerate([5, 1.5, 1.5, 1]):
                    self.loss_G_perceptual += λ_i * self.perceptual_loss(feats_fake[i], feats_real[i])
                self.loss_G_perceptual*=self.opt.lambda_perceptual
            # combine loss and calculate gradients
            del fake_AB, pred_fake, feats_fake, feats_real
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual
        self.scaler.scale(self.loss_G).backward()

    def optimize_parameters(self):
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.scaler.step(self.optimizer_D)  # udpate D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.scaler.step(self.optimizer_G)  # udpate G's weights
        self.scaler.update()                # Updates the scale for next iteration 
        
    def optimize_parameters_wo_l1(self):
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.scaler.step(self.optimizer_D)  # udpate D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G_wo_l1()             # calculate graidents for G without l1 loss
        self.scaler.step(self.optimizer_G)  # udpate G's weights
        self.scaler.update()                # Updates the scale for next iteration 
    
    def backward_kl(self):
        def unflatten_conv_map(flattened):
            return flattened.reshape([flattened.shape[0], flattened.shape[1], 
                                      self.latent_dimension, self.conv_h, self.conv_w])
        def convert_image_to_256_3d(x):
            # x, [1, 1, 144, 192, 192]
            
            #batch_size = 1
            #channels = 1
            shape = x.shape
            x = x.squeeze()
            x = crop_or_pad_volume_to_size_along_x(x, 256)
            x = crop_or_pad_volume_to_size_along_y(x, 256)
            x = crop_or_pad_volume_to_size_along_z(x, 256)
            return torch.unsqueeze(torch.unsqueeze(x, 0), 0) # x, [1,1, 256, 256, 256]
        
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        with torch.cuda.amp.autocast(enabled=self.opt.amp):

            # print(self.netVae)
            # print(self.fake_B.device)
            # print(next(self.netVae.parameters()).device)
            real_, fake_ = None, None
            if self.opt.use_3d_vae:
                fake_ = normalise_image( convert_image_to_256_3d(self.fake_B) )
                real_ = normalise_image( convert_image_to_256_3d(self.real_B.detach() ) )
                
            else:
                fake_ = normalise_image( self.convert_image_to_256_size(self.fake_B) )
                real_ = normalise_image( self.convert_image_to_256_size(self.real_B.detach()) )
            
            latent_fake, mu_fake, std_fake = self.netVae(fake_)
            
            # print(temp.device)
            latent_real, mu_real, std_real = self.netVae(real_)
            
            normal_fake = normal.Normal(mu_fake, torch.exp(std_fake))
            normal_real = normal.Normal(mu_real, torch.exp(std_real))
            
            del real_, fake_, latent_fake, latent_real # delete the useless references
            
            kl_ =  kl.kl_divergence(normal_real, normal_fake)
            p = percentile(kl_,100 - self.opt.ele_percent)
         
            kl_ = kl_[kl_ > p] 
            
            self.loss_kl = torch.mean(kl_)
            # print(self.loss_kl)
           
        self.scaler.scale(self.loss_kl).backward()

    def optimize_parameters_kl(self):
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.forward()                   # compute fake images: G(A)
     
        # update G
        self.set_requires_grad(self.netVae, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_kl()                   # calculate graidents for G
        self.scaler.step(self.optimizer_G)  # udpate G's weights
        self.scaler.update()                # Updates the scale for next iteration 
