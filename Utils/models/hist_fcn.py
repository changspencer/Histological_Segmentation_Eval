import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

#Turn off plotting
plt.ioff()


class HistogramLayer(nn.Module):
    '''
    Taken from Josh's Histogram_Layer Texture Classification network (HistNet)
    '''
    def __init__(self,in_channels,kernel_size,dim=2,num_bins=4,
                 stride=1,padding=0,normalize_count=True,normalize_bins = True,
                 count_include_pad=False,
                 ceil_mode=False):

        # inherit nn.module
        super(HistogramLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        
        #For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if self.dim == 1:
            self.bin_centers_conv = nn.Conv1d(self.in_channels,self.numBins*self.in_channels,1,
                                            groups=self.in_channels,bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv1d(self.numBins*self.in_channels,
                                             self.numBins*self.in_channels,1,
                                             groups=self.numBins*self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool1d(self.filt_dim,stride=self.stride,
                                             padding=self.padding,ceil_mode=self.ceil_mode,
                                             count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
        
        # Image Data
        elif self.dim == 2:
            self.bin_centers_conv = nn.Conv2d(self.in_channels,self.numBins*self.in_channels,1,
                                            groups=self.in_channels,bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv2d(self.numBins*self.in_channels,
                                             self.numBins*self.in_channels,1,
                                             groups=self.numBins*self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool2d(self.kernel_size,stride=self.stride,
                                             padding=self.padding,ceil_mode=self.ceil_mode,
                                             count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
        
        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            self.bin_centers_conv = nn.Conv3d(self.in_channels,self.numBins*self.in_channels,1,
                                            groups=self.in_channels,bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv3d(self.numBins*self.in_channels,
                                             self.numBins*self.in_channels,1,
                                             groups=self.numBins*self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool3d(self.filt_dim,stride=self.stride,
                                             padding=self.padding,ceil_mode=self.ceil_mode,
                                             count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
            
        else:
            raise RuntimeError('Invalid dimension for histogram layer')

        # Change initializations for the histogram mean values
        # nn.init.uniform(self.centers, a=-2, b=2)

        
    def forward(self,xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window

        #Pass through first convolution to learn bin centers
        xx = self.bin_centers_conv(xx)
        
        #Pass through second convolution to learn bin widths
        xx = self.bin_widths_conv(xx)
        
        #Pass through radial basis function
        xx = torch.exp(-(xx**2))
        
        #Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if(self.normalize_bins):
            xx = self.constrain_bins(xx)
        
        #Get localized histogram output, if normalize, average count
        if(self.normalize_count):
            xx = self.hist_pool(xx)
        else:
            xx = np.prod(np.asarray(self.hist_pool.kernel_size))*self.hist_pool(xx)

        return xx
    
    
    def constrain_bins(self,xx):
        #Enforce sum to one constraint across bins
        # Time series/ signal Data
        if self.dim == 1:
            n,c,l = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins, l).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum  
        
        # Image Data
        elif self.dim == 2:
            n,c,h,w = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum  
        
        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            n,c,d,h,w = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins,d, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum   
            
        else:
            raise RuntimeError('Invalid dimension for histogram layer')
         
        return xx


class HistFCN(nn.Module):
    """
    A fully-convolutional network with a feature extractor modeled after AlexNet and
    histogram layer pooling modeled after Josh's dissertation.

    Original 'self.features' found at 
        https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py.
    """
    def __init__(self, in_channels, num_classes, n_bins=4, norm_count=True,
                 norm_bins=True, dropout=0.1):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.bilinear = True
        self.use_attention = False
        self.analyze = False
        
        self.n_bins = n_bins
        self.norm_count = norm_count
        self.norm_bins = norm_bins

        self.main_feats = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Output (H-1)/2
        self.sub_feats = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            HistogramLayer(256 // self.n_bins,
                           kernel_size=2,
                           num_bins=self.n_bins,
                           stride=2,
                           normalize_count=self.norm_count,
                           normalize_bins=self.norm_bins,
            )
        )  # Output (H-1)/2

        # Shortcut upsampling connections
        self.shortcut = nn.Sequential(
            nn.Conv2d(384, 96, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Dropout(dropout),
            nn.Conv2d(96, num_classes, 1),
        )  # Output H x 2

        # Final Concatenation part - connects with output of self.conv4
        self.head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        # )
        # self.out_conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        x2 = self.main_feats(x)
        x3 = self.sub_feats(x2)

        # Shortcuts
        x2 = F.pad(x2, [0, 0, 0, 1])
        skip = self.shortcut(x2)

        # Consolidate shortcuts and FCN head output
        x3 = F.pad(x3, [0, 0, 0, 1])
        x5 = self.head(x3)
        x5 = x5[:, :, :skip.shape[-2], :skip.shape[-1]]  # Crop out extra pixels
        head = (x5 + skip) / 2  # Average the prediction values

        out = self.up(self.up(head))
        out = out[:, :, :x.shape[-2], :x.shape[-1]]  # Crop out extra pixels
        return out