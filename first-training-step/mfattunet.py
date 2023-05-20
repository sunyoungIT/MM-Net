import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel

from models.convs import common

# for first-training step : Multi-frame multi-scale attention U-Net
class MFATTUNET(BaseModel):
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
        # n_inputs is set in base options
        # n_channels is set in dataset options
        parser.add_argument('--ms_channels', type=int, default=32,
            help='number of features of output in multi-scale convolution')
        parser.add_argument('--growth_rate', type=int, default=64,
            help='growth rate of each layer in dense block')
        parser.add_argument('--n_denselayers', type=int, default=5,
            help='number of layers in dense block')
        parser.add_argument('--bilinear', type=str, default='bilinear',
            help='up convolution type (bilineaer or transposed2d)')
        if is_train:
            parser.add_argument('--content_loss', type=str, choices=['l1', 'l2'], default='l2',
                help='loss function (l1, l2)')
      
        parser.set_defaults(depth=5)

        return parser

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.dataA:
            dataset_name = dataset_name + d

        dataset_name = dataset_name + '-'
        for d in opt.dataB:
            dataset_name = dataset_name + d
        model_opt = dataset_name  + "-" + date + "-" + opt.model

        model_opt = dataset_name  + "-" + date + "-" + opt.model
        model_opt = model_opt + "-depth" + str(opt.depth)

        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        savedir = os.path.join(opt.checkpoints_dir, model_opt)
        return savedir

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # for the first-training step : MFATTUNET (Multi-scale attention U-Net)
        self.model_names = ['mfattunet'] 
        # Create model
        self.mfattunet = create_model(opt).to(self.device)
        
        #print(self.qenet)
        # Define losses and optimizers
        if self.is_train:
            self.content_loss_criterion = nn.L1Loss()
            self.optimizer_names = ['optimizerQ']
            self.optimizerQ = torch.optim.Adam(
                self.mfattunet.parameters(),
                lr=opt.lr,
                betas=(opt.b1, opt.b2),
                eps=1e-8,
                weight_decay=0
            )
            self.optimizers.append(self.optimizerQ)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()
        self.n_inputs = opt.depth

        # url_name = 'n{}m{}g{}n{}'.format(opt.n_inputs, opt.ms_channels, opt.growth_rate, opt.n_denselayers)
        # if url_name in url:
        #     self.url = url[url_name]
        # else:
        #     self.url = None

    def set_input(self, input):

        # revise your code according to the specifications of your dataset
        x = input['x'].to(self.device)
        
        bs, c, d, h, w = x.shape # batch size, channel, depth (number of the input frames), H, W
        self.ix = torch.cat((x[:, :, :self.n_inputs//2], x[:, :, self.n_inputs//2+1:]), dim=1)
        self.ix = self.ix.view(bs, (d-1)*c, h, w)
        self.nx = x[:, :, self.n_inputs//2] 
    
        if 'target' in input:
            self.target = input['target'].to(self.device)
            self.ct = self.target[:, :, self.n_inputs//2]

    def forward(self):
        self.out = self.mfattunet(self.ix)

    def backward(self):
        # if self.perceptual_loss:
        #     self.content_loss, self.style_loss = self.perceptual_loss_criterion(self.nx, self.out)
        #     self.loss = self.content_loss + self.style_loss
        # else:
        self.loss = self.content_loss_criterion(self.nx, self.out)

        self.loss.backward()

        mse_loss = self.mse_loss_criterion(self.out.detach(), self.ct.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def optimize_parameters(self):
        self.optimizerQ.zero_grad()
        self.forward()
        self.backward()
        self.optimizerQ.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        # if self.perceptual_loss:
        #     print("Content Loss: {:.8f}, Style Loss: {:.8f}".format(
        #         self.content_loss, self.style_loss)
        #     )
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.8f}, PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )

def create_model(opt):
    return MFAttunet(opt)

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=3//2)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=5//2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, padding=7//2)

    def forward(self, x):
       
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        out = torch.cat((x3, x5, x7), dim=1)

        
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0)

    def forward(self, x):
        return self.conv(x)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class ATTUNET(nn.Module):
    def __init__(self,img_ch,output_ch):
        super(ATTUNET,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)    
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)     
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
    
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class MFAttunet(nn.Module):
    def __init__(self, opt):
        super(MFAttunet, self).__init__()
        # n_inputs : number of the input frames
        self.n_inputs = opt.depth - 1  # except for center frame 
        self.nc = n_channels

        n_channels = opt.n_channels
        ms_channels = opt.ms_channels # number of channels of output multi-scale conv
        dense_in_channels = n_channels * ms_channels * 3 * self.n_inputs
        growth_rate = opt.growth_rate
       
        multiscale_conv = [MultiScaleConv(n_channels, ms_channels) for _ in range(self.n_inputs)]
        self.multiscale_conv = nn.ModuleList(multiscale_conv)
        self.attunet = ATTUNET(dense_in_channels, growth_rate)
        self.outc = OutConv(64, n_channels)
        
    def forward(self, x):
        
        x_mean = torch.zeros(x[:,:self.nc].shape, dtype=x.dtype, device=x.device)
        for i in range(self.n_inputs):
            x_mean = x_mean + x[:, i * self.nc: i * self.nc + self.nc]
        x_mean = x_mean / self.n_inputs
     
        # multi-scale network
        ms_out = []
        for i in range(self.n_inputs):
            x_in  = x[:, i*self.nc:i*self.nc+self.nc]
            ms_conv = self.multiscale_conv[i]
        
            ms_out.append(ms_conv(x_in))
        
        ms_out = torch.cat(ms_out, dim=1)
        
        out = self.attunet(ms_out) 
        out = self.outc(out) + x_mean
        
        return out
