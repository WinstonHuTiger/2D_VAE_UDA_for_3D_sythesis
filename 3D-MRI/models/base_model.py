import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from util.visualizer import Visualizer
from . import networks
from util.util import colorFaderTensor, load_loss_log, load_val_log


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.networks = []
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.opt.amp, init_scale=1)

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.registration_artifacts_idx = None
        if 'registration_artifacts_idx' in input:
            self.registration_artifacts_idx = input['registration_artifacts_idx']

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setSchedulers(self, opt):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            if self.opt.isTrain and self.opt.pretrained_name is not None:
                load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
            else:
                load_dir = self.save_dir
            load_suffix = opt.epoch
            if opt.checkpoint_path is not None:
                self.load_checkpoint(opt.checkpoint_path)
            else:
                self.load_networks(load_suffix)
        if self.isTrain and opt.continue_train and int(opt.epoch_count) > 0:
            loss_data = load_loss_log(os.path.join(load_dir, 'loss_log.txt'), opt.dataset_size)
            y, legend = load_val_log(os.path.join(load_dir, 'val_loss_log.txt'))
            val_data = (list(range(len(y))), y, legend)
            v: Visualizer = opt.visualizer
            v.set_plot_data(loss_data, val_data)
            v.plot_current_losses(legend=loss_data[-1])
            v.plot_current_validation_losses()

        self.print_networks(opt.verbose)
        if self.opt.phase=="train":
            self.save_network_architecture()

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                setattr(self, 'net' + name, torch.nn.DataParallel(net, self.opt.gpu_ids))

    def data_dependent_initialize(self, data):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.opt.amp):
                self.forward()
                if self.opt.bayesian:
                    preds = [self.fake_B]
                    for i in range(10 - 1):
                        self.forward()
                        preds.append(self.fake_B)
                    preds = torch.stack(preds)
                    self.fake_B = preds.mean(axis=0)
                    self.std_map = preds.std(axis=0).detach().cpu()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self, slice=True):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                tmp = getattr(self, name).detach().cpu()
                if len(tmp[0])>1:
                    for i in range(len(tmp[0])):
                        if tmp.dim() == 5 and slice:
                            visual_ret[name+f'_{i}'] = [tmp[:,i:i+1,tmp.shape[-3]//2,:,:], tmp[:,i:i+1,:,tmp.shape[-2]//2,:], tmp[:,i:i+1,:,:,tmp.shape[-1]//2]]
                    # For 3D data, take a slice along the z-axis
                        else:
                            visual_ret[name+f'_{i}'] = [tmp[:,i:i+1]]
                else:
                    if tmp.dim() == 5 and slice:
                        visual_ret[name] = [tmp[:,:,tmp.shape[-3]//2,:,:], tmp[:,:,:,tmp.shape[-2]//2,:], tmp[:,:,:,:,tmp.shape[-1]//2]]
                    else:
                        visual_ret[name] = [tmp[:,:]]
        if self.opt.bayesian:
            std_map: torch.Tensor = self.std_map[0:1,0:1]
            if std_map.dim() == 5 and slice:
                std_maps = []
                for i in range(2,5):
                    std_map_i = std_map.select(i, std_map.shape[i]//2).clone()
                    shape = std_map_i.shape
                    std_map_i -= std_map_i.min()
                    std_map_i /= std_map_i.max()
                    std_map_i = std_map_i.float()
                    std_map_i = std_map_i.view(1, -1)
                    std_map_i = colorFaderTensor(std_map_i)
                    std_map_i = std_map_i.permute(0,2,1)
                    std_map_i = std_map_i.view(shape[0], 3, *shape[2:])
                    std_maps.append(std_map_i)
                visual_ret['confidence'] = std_maps
            else:
                std_map -= std_map.min()
                std_map /= std_map.max()
                std_map = std_map.float()
                visual_ret['confidence'] = [std_map]
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name).detach().cpu())  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_network_architecture(self):
        networks = [(name, getattr(self, 'net' + name)) for name in self.model_names]
        save_filename = 'architecture.txt'
        save_path = os.path.join(self.save_dir, save_filename)

        architecture = ''
        for name, n in networks:
            architecture += str(n) + '\n'
            num_params = 0
            for param in n.parameters():
                num_params += param.numel()
            architecture += '[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6)
        with open(save_path, 'w') as f:
            f.write(architecture)
            f.flush()
            f.close()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        states = checkpoint.pop('networks')
        for i in range(len(self.networks)):
            self.networks[i].load_state_dict(states[i])
        print('Loaded checkpoint successfully')
        return checkpoint

    def create_checkpoint(self, path, d=None):
        states = list(map(lambda x: x.cpu().state_dict(), self.networks))
        checkpoint = {
            "networks": states
        }
        if d is not None:
            checkpoint.update(d)
        torch.save(checkpoint, path)
        for x in self.networks:
            x.cuda()
