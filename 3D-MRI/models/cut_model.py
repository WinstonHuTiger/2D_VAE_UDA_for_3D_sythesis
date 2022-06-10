import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0, paired=False)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G']
        if opt.lambda_NCE > 0:
            self.loss_names.append('NCE')
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            if opt.lambda_NCE > 0:
                self.loss_names.append('NCE_Y')
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.networks = [self.netG]
        if self.isTrain:
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.networks.extend([self.netD, self.netF])
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, dtype=torch.float16 if opt.amp else torch.float32).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.set_input(data)
            bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
            self.real_A = self.real_A[:bs_per_gpu]
            self.real_B = self.real_B[:bs_per_gpu]
            with torch.no_grad():
                self.forward()                     # compute fake images: G(A)
                if self.opt.isTrain:
                    self.compute_D_loss()                # calculate gradients for D
                    self.compute_G_loss()                   # calculate gradients for G
                    if self.opt.lambda_NCE > 0.0:
                        self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                        self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        # Casts operations to mixed precision
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.forward()
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
        self.scaler.scale(self.loss_D).backward()
        self.scaler.step(self.optimizer_D)
        # Updates the scale for next iteration
        torch.cuda.empty_cache()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE>0:
            self.optimizer_F.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.loss_G = self.compute_G_loss()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)
        # Updates the scale for next iteration
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE>0:
            self.scaler.step(self.optimizer_F)
            # Updates the scale for next iteration
        self.scaler.update()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        real_A_in = self.real_A
        if self.opt.nce_idt and self.opt.isTrain:
            real_B_in = self.real_B.expand([*self.real_A.shape[:2], *self.real_B.shape[2:]])
        if self.opt.lambda_NCE and self.opt.phase == 'train':
            self.fake_B, self.real_A_feats = self.netG(real_A_in, self.nce_layers)
        else:
            self.fake_B = self.netG(real_A_in)
        if self.registration_artifacts_idx is not None:
            self.fake_B = self.fake_B * self.registration_artifacts_idx.to(self.fake_B.device)
        if self.opt.lambda_NCE and self.opt.nce_idt and self.opt.isTrain:
            self.idt_B, self.real_B_feats = self.netG(real_B_in, self.nce_layers)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A_feats, self.fake_B.expand(self.real_A.shape))
            self.real_A_feats = None
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B_feats, self.idt_B.expand([*self.real_A.shape[:2], *self.real_B.shape[2:]]))
            self.real_B_feats = None
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, feat_k, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        # feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

def log_gradients(model):
		all_grads = {}
		all_weights = {}
		for n, p in model.named_parameters():
			if p.grad is not None:
				all_grads[n] = np.mean(p.grad.cpu().numpy())
				all_weights[n] = p 
		return all_grads, all_weights
