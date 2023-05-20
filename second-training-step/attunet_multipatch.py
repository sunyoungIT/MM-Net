import os
import datetime
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.transforms import Grayscale
from data.common import get_patch2
import itertools
from models.convs.wavelet import SWTForward, serialize_swt, SWTInverse, unserialize_swt
import PIL
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models.common.unet import create_unet
from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss
from .base_model import BaseModel


# I will update it soon. I have uploaded the code with essential components. 
# For the code related to generating masks, please refer to the algorithm described in the paper. 
# To train the second stage, you will need the pretrained .pth file from the first stage. 
# This pretrained .pth file is trained on the Mayo dataset using the multiscale attention U-Net. 
# If you require it, please feel free to contact me at any time.

def create_qenet(opt):
    
    qenet = create_mfunet3(opt)
    
    pretrained_path = '' # I will update it soon. If you need the pretrained .pth file, please feel free to contact me.
   
    checkpoint = torch.load(pretrained_path)
    
    qenet.load_state_dict(checkpoint['mfattunet3'])
    return qenet

class ATTUNET_MULTIPATCH_NEW(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # Network parameters
        parser.add_argument('--wavelet_func', type=str, default='haar', #'bior2.2',
            help='wavelet function ex: haar, bior2.2, or etc.')
        parser.add_argument('--ms_channels', type=int, default=32,
            help='number of features of output in multi-scale convolution')
        parser.add_argument('--growth_rate', type=int, default=64,
            help='growth rate of each layer in dense block')
        parser.add_argument('--n_denselayers', type=int, default=5,
            help='number of layers in dense block')
        # parser.add_argument('--bilinear', type=str, default='bilinear',
        #     help='up convolution type (bilineaer or transposed2d)')
        #parser.add_argument('--content_loss', type=str, choices=['l1', 'l2'], default='l2',
        #        help='loss function (l1, l2)')
        parser.add_argument('--n_dense_channels', type=int, default=32,
            help='number of features of output in multi-scale convolution')
        # parser.add_argument('--growth_rate', type=int, default=32,
        #     help='growth rate of each layer in dense block')
        # parser.add_argument('--n_denselayers', type=int, default=5,
        #     help='number of layers in dense block')
        parser.add_argument('--n_denseblocks', type=int, default=2,
            help='number of layers of dense blocks')
        parser.add_argument('--qenet', type=str, default='mfselfattn',
            help='specify the pretrained qenet')
        parser.add_argument('--patch_size2', type=int, default=120, # n2m
                help='n2m size of patch')
        parser.add_argument('--alpha', type=float, default=0.08, # n2m
            help='n2m size of patch')
        parser.add_argument('--patch_number', type=int, default=10, # n2m
                help='number of patch')    

        
        #parser.set_defaults(patch_size=40)
        parser.set_defaults(content_loss = 'l1')

        parser.add_argument('--backbone', type=str, default='unet',
            choices=['unet'],
            help='backbone model'
        )
        opt, _ = parser.parse_known_args()
        parser.add_argument('--perceptual_loss', type=str, default=None,
             choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
             help='specity loss_type')
         # U-Net
        if opt.backbone == 'unet':
            parser.add_argument('--bilinear', type=str, default='bilinear',
                help='up convolution type (bilineaer or transposed2d)')

        if is_train:
            parser.set_defaults(lr=1e-4)
            parser.set_defaults(b1=0.5)
            parser.set_defaults(b2=0.999)
            parser.set_defaults(valid_ratio=0.1)
            parser = parse_perceptual_loss(parser)
        return parser

    def __init__(self, opt):
            BaseModel.__init__(self, opt)

            self.model_names = ['netG']  # model_names
            self.swt_lv = 1
            self.swt = SWTForward(J=self.swt_lv, wave=opt.wavelet_func).to(self.device)
            self.iswt = SWTInverse(wave=opt.wavelet_func).to(self.device)
        
            # Create model
            self.netG = create_unet(opt).to(self.device)
            self.nc = opt.n_channels
            self.n_inputs = opt.depth
            if opt.perceptual_loss is not None:
                self.perceptual_loss = opt.perceptual_loss
            self.content_loss = opt.content_loss

            self.qenet = create_qenet(opt).to(self.device)
            self.set_requires_grad([self.qenet], False)
            # Define losses and optimizers
            if self.is_train:
                if opt.perceptual_loss is not None:
                    self.perceptual_loss = True
                    self.loss_type = opt.perceptual_loss
                else :
                    self.perceptual_loss = False
                self.model_names = ['qenet', 'netG']
                if self.content_loss == 'l1':
                    print("l1 Loss using...")
                    self.criterionL1 = nn.L1Loss()
                
                self.criterionMSE = nn.MSELoss()
                self.patch_number = opt.patch_number
                self.alpha = opt.alpha
                self.patch_size2 = opt.patch_size2 
            
                if self.perceptual_loss:
                    self.perceptual_loss_criterion = PerceptualLoss(opt)
                self.optimizer_names = ['optimizerG']
                self.optimizerG = torch.optim.Adam(
                    #itertools.chain(self.netG.parameters(), self.netG2.parameters()),
                    self.netG.parameters(),
                    lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0
                )
                self.optimizers.append(self.optimizerG)
                
            # url_name = 'n{}m{}g{}n{}'.format(opt.n_inputs, opt.ms_channels, opt.growth_rate, opt.n_denselayers)
            # if url_name in url:
            #     self.url = url[url_name]
        # else:
        #     self.url = None

    def set_input(self, input):
            x = input['x'].to(self.device)
            bs, c, d, h, w = x.shape
            
            # For QENet
            self.ix = torch.cat((x[:, :, :self.n_inputs//2], x[:, :, self.n_inputs//2+1:]), dim=2)
            self.ix = self.ix.view(bs, (d-1)*c, h, w)
            self.nx = x[:, :, self.n_inputs//2]
            
            self.real_x1 = x[:, :, self.n_inputs//2]
        
            if 'target' in input:
                target = input['target'].to(self.device)
                self.target = target[:, :, self.n_inputs//2]

    def forward(self):
        self.out = self.netG(self.real_x1)
        if self.is_train:
            self.input_x, self.input_y, self.x_mask, self.y_mask, self.xy_metric = self.patch(self.out)

        self.qe_out = self.qenet(self.ix.detach()) 

    def backward(self):
        # QENet prior image
        self.loss_zero = self.mask_patch_zero(self.input_x[0], self.input_y[0], self.x_mask[0], self.y_mask[0], self.qe_out, self.xy_metric[0])
        for i in range(1,self.patch_number-1):
                self.t_loss = self.mask_patch(self.input_x[i], self.input_y[i], self.x_mask[i], self.y_mask[i], self.qe_out, self.xy_metric[i])
                self.loss_zero = self.loss_zero + self.t_loss
                #print("self.loss_zero is ?",self.loss_zero.item())
        self.loss_gq = self.loss_zero
        self.loss_g = self.loss_gq
        #(self.netG.parameters())
        
        self.loss_g.backward()
        #print(self.netG.grad)

        # To calculate PSNR
        self.loss = self.criterionMSE(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / self.loss)
        #self.loss2 = self.criterionMSE(self.out2.detach(), self.target.detach())
        #self.psnr2 = 10 * torch.log10(1 / self.loss2)
        #self.psnr = (self.psnr1 + self.psnr2)  / 2
        self.qe_loss = self.criterionMSE(self.qe_out.detach(), self.target.detach())
        self.qe_psnr = 10 * torch.log10(1 / self.qe_loss)



    def optimize_parameters(self):
        self.optimizerG.zero_grad()
        self.forward()
        self.backward()
        self.optimizerG.step()

    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        # if self.perceptual_loss:
        #     print("Content Loss: {:.8f}, Style Loss: {:.8f}".format(
        #         self.content_loss, self.style_loss)
        #     )
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.8f}, PSNR: {:.5f}, QE PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss_gq.item(), self.psnr.item(), self.qe_psnr.item())
        )
    
    def metric_function(self,xy):
        x_a = []
        y_b = []
        for i in range(self.patch_number-1):
            x, y = self._metric(xy[i][0],xy[i][1]) # _metric :Please refer to the algorithm outlined in the paper for guidance.
            x_a.append(x)
            y_b.append(y)
      
        return x_a, y_b

    def patch(self, lr):
        x = []
        y = []
        xy = []
        ret1, metric1 = get_patch2(
            lr,
            patch_size=self.patch_size2,
            n_channels=1,
            patch_number = self.patch_number)
        
        ret2, metric2 = get_patch2(
            lr,
            patch_size=self.patch_size2,
            n_channels=1,
            patch_number=self.patch_number)
        
        patch_number = self.patch_number
        x.append(list(metric1[i] for i in range(patch_number-1)))
        y.append(list(metric2[i] for i in range(patch_number-1)))
        xy = list((x[0][i],y[0][i]) for i in range(patch_number-1))
        
        x_m, y_m = self.metric_function(xy)
    
        return ret1, ret2, x_m, y_m, xy
