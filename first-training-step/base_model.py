import os
import glob
import datetime

import torch
import torch.nn as nn
from collections import OrderedDict
from abc import ABC, abstractmethod
from .common import networks

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
        self.is_train = opt.is_train
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if len(self.gpu_ids) > 0 else torch.device('cpu')  # get device name: CPU or GPU
        self.multi_gpu = opt.multi_gpu
        # print("{} opt.device: {}".format(__file__, opt.device))
        self.device = opt.device
        
        # self.savedir = os.path.join(opt.checkpoints_dir, opt.model)  # save all the checkpoints to savedir
        # if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        #     torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.var_names = []
        self.optimizer_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        self.savedir = opt.savedir

        # self.batch_loss = 0
        # self.batch_psnr = 0

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

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    """My abstract methods"""
    @staticmethod
    def set_savedir(opt):
        """
        This functions should set save directory to save checkpoints
        """
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.dataA:
            dataset_name = dataset_name + d

        for d in opt.dataB:
            dataset_name =  dataset_name  + '-' + d
        model_opt = dataset_name  + "-" + date + "-" + opt.model

        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        save_dir = os.path.join(opt.checkpoints_dir, model_opt)
        return save_dir
        # pass

    def _name_dataset(self, opt):
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d
        return dataset_name

    """My abstact methods"""
    @abstractmethod
    def log_loss(self):
        """Define your log of losses"""
        pass

    # """My abstact methods"""
    # @abstractmethod
    # def get_batch_loss_psnr(self):
    #     """Define getter of batch loss"""
    #     pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.is_train and opt.resume:
            opt.epoch, opt.best_psnr = self.load_networks(opt.load_epoch)
        elif not self.is_train and opt.url:
            self.download_url()
        elif not self.is_train:
            opt.epoch, opt.best_psnr = self.load_networks(opt.load_epoch)

        if self.is_train:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        self.print_networks(opt.verbose)

        if self.multi_gpu:
            for net_name in self.model_names:
                if isinstance(net_name, str):
                    net = getattr(self, net_name)
                    setattr(self, net_name, nn.DataParallel(net, device_ids=opt.gpu_ids))

    """ My method """
    def train(self):
        """Make models eval mode during test time"""
        for net_name in self.model_names:
            if isinstance(net_name, str):
                net = getattr(self, net_name)
                net.train()
                # print("trainin:", net.training)
    
    """ My method """
    def get_batch_loss_psnr(self):
        return self.loss.detach(), self.psnr.detach()

    def eval(self):
        """Make models eval mode during test time"""
        for net_name in self.model_names:
            if isinstance(net_name, str):
                net = getattr(self, net_name)
                net.eval()
                # print("traning:", net.training)

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            # self.compute_visuals()

    # def compute_visuals(self):
    #     """Calculate additional output images for visdom and HTML visualization"""
    #     pass

    # def get_image_paths(self):
    #     """ Return image paths that are used to load current data"""
    #     return self.image_paths

    def update_learning_rate(self, metric):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # def get_current_visuals(self):
    #     """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
    #     visual_ret = OrderedDict()
    #     for name in self.visual_names:
    #         if isinstance(name, str):
    #             visual_ret[name] = getattr(self, name)
    #     return visual_ret

    def get_current_losses(self):
        """Return training losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    # def save_networks(self, epoch):
    #     """Save all the networks to the disk.

    #     Parameters:
    #         epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    #     """
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             save_filename = '%s_net_%s.pth' % (epoch, name)
    #             os.makedirs(self.savedir, exist_ok=True)
    #             save_path = os.path.join(self.savedir, save_filename)
    #             net = getattr(self, 'net' + name)

    #             print("Saving to {}".format(os.path.abspath(save_path)))

    #             if len(self.gpu_ids) == 1 and torch.cuda.is_available():
    #                 torch.save(net.to("cpu").state_dict(), save_path)
    #                 net =net.to(self.gpu_ids[0])
    #             elif len(self.gpu_ids) > 1 and torch.cuda.is_available():
    #                 torch.save(net.module.to("cpu").state_dict(), save_path)
    #                 net = net.to(self.gpu_ids[0])
    #             else:
    #                 torch.save(net.to("cpu").state_dict(), save_path)

    def save_networks(self, epoch, epoch_num, loss, psnr):
        """
        Define saving networks and OPTIMIZERS to the disk

        parameters:
            epoch(int) -- current epoch
            loss(float) -- calculated current loss
            used in the file name 'epoch_{}_loss_{}.pth'.format(epoch, loss)
        """
        # os.makedirs(self.savedir, exist_ok=True)
        pth_dir = os.path.join(self.savedir, 'pth')
        os.makedirs(pth_dir, exist_ok=True)
        state = {}
        save_file_name = 'epoch_{}_n{:04d}_loss{:.8f}_psnr{:.4f}.pth'.format(epoch, epoch_num, loss, psnr)
        save_path = os.path.join(self.savedir, 'pth', save_file_name)

        state['epoch'] = epoch_num
        state['loss'] = loss
        state['psnr'] = psnr
        for net_name in self.model_names:
            if isinstance(net_name, str):
                net = getattr(self, net_name)
                if self.multi_gpu:
                    state[net_name] = net.module.state_dict()
                else:
                    state[net_name] = net.state_dict()
        for opt_name in self.optimizer_names:
            if isinstance(opt_name, str):
                optimizer = getattr(self, opt_name)
                state[opt_name] = optimizer.state_dict()
        
        print("Saving to {}".format(os.path.abspath(save_path)))
        torch.save(state, save_path)

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

        pth_dir = os.path.join(self.savedir, 'pth')
        print(pth_dir)
        print(epoch)
        print('epoch_*{}*.pth'.format(epoch))
        epoch_path = os.path.join(pth_dir, 'epoch_*{}*.pth'.format(epoch))
        epoch_search = glob.glob(epoch_path)
        epoch_path = epoch_search[0]

        print('loading the checkpoint from {}'.format(os.path.abspath(epoch_path)))
        checkpoint = torch.load(epoch_path, map_location=self.device)

        epoch = checkpoint['epoch']
        psnr = checkpoint['psnr']
        for net_name in self.model_names:
            if isinstance(net_name, str):
                net = getattr(self, net_name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model {}'.format(net_name))
                state_dict = checkpoint[net_name]
                net.load_state_dict(state_dict)
        
        if self.is_train:
            for opt_name in self.optimizer_names:
                if isinstance(opt_name, str):
                    optimizer = getattr(self, opt_name)
                    print('loading the {}'.format(opt_name))
                    state_dict = checkpoint[opt_name]
                    optimizer.load_state_dict(state_dict)
        
        print('finished loading the checkpoint from {} with epoch {}'.format(os.path.abspath(epoch_path), epoch))
        return epoch, psnr

    def remove_networks(self, epoch):
        pth_dir = os.path.join(self.savedir, 'pth')

        epoch_path = os.path.join(pth_dir, 'epoch_*{}*.pth'.format(epoch))
        epoch_search = glob.glob(epoch_path)
        
        if len(epoch_search) == 1:
            epoch_path = epoch_search[0]

            if os.path.exists(epoch_path):
                print('removing epoch {}'.format(epoch_path))
                os.remove(epoch_path)

    def download_url(self):
        print("downloading url from {}".format(self.url))
        model_dir = os.path.join(self.savedir, self.url_name)
        
        os.makedirs(model_dir, exist_ok=True)
        checkpoint = torch.hub.load_state_dict_from_url(self.url, model_dir=model_dir)

        for net_name in self.model_names:
            if isinstance(net_name, str):
                net = getattr(self, net_name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model {}'.format(net_name))
                state_dict = checkpoint[net_name]
                net.load_state_dict(state_dict)
        
        if self.is_train:
            for opt_name in self.optimizer_names:
                if isinstance(opt_name, str):
                    optimizer = getattr(self, opt_name)
                    print('loading the {}'.format(opt_name))
                    state_dict = checkpoint[opt_name]
                    optimizer.load_state_dict(state_dict)
        
        epoch = checkpoint['epoch']
        psnr = checkpoint['psnr']
        print('finished downloading the checkpoint from {} with epoch {} and psnr {:.8f}'.format(
            self.url, epoch, psnr))
        return epoch, psnr

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                print("name:", name)
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

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

    # for calculating receptif fiels
    def init_nets(self):
        for net_name in self.model_names:
            if isinstance(net_name, str):
                net = getattr(self, net_name)
                for module in net. modules():
                    try:
                        # Make all convolution weights equal.
                        # Set all biases to zero.
                        nn.init.constant_(module.weight, 0.05)
                        nn.init.zeros_(module.bias)
	                    
                        # Set BatchNorm means to zeros,
	                    # variances - to 1.
                        nn.init.zeros_(module.running_mean)
                        nn.init.ones_(module.running_var)
                    except:
                        pass

                    if isinstance(module, torch.nn.modules.BatchNorm2d):
                        module.eval()

        if self.opt.model == 'waveletdl2':
            model_names = ['swt', 'iswt']
            for net_name in model_names:
                if isinstance(net_name, str):
                    net = getattr(self, net_name)
                    for module in net. modules():
                        try:
                            # Make all convolution weights equal.
                            # Set all biases to zero.
                            nn.init.constant_(module.weight, 0.05)
                            nn.init.zeros_(module.bias)
                            
                            # Set BatchNorm means to zeros,
                            # variances - to 1.
                            nn.init.zeros_(module.running_mean)
                            nn.init.ones_(module.running_var)
                        except:
                            pass

                    for params in net.parameters():
                        params.requires_grad = True


    def print_weights(self):
        for net_name in self.model_names:
            if isinstance(net_name, str):
                net = getattr(self, net_name)
                for module in net. modules():
                    try:
                        # Make all convolution weights equal.
                        # Set all biases to zero.
                        print('weight: ', module.weight)
                        print('bias:', module.bias)
                    except:
                        pass

    def grad_backward(self):
        grad = torch.zeros_like(self.out, requires_grad=True)
        grad[:, :, 242:262, 242:262] = 1

        self.out.backward(gradient=grad)


        # if self.opt.model == 'waveletdl2':


