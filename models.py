#!/usr/bin/env python3
# author: jinhee

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)

"""
for within-subject : Deep_ConvNEt, EEGNet, EEG-TCNet, CCRNN
for cross-subject : with 'sub_*'
"""
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return F.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride, self.padding, self.dilation, self.groups)

class ConstrainedLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight.clamp(min=-0.25, max=0.25), self.bias)
    
class Deep_ConvNet(nn.Module):
    def __init__(self, bias=False, num_class=2):
        super(Deep_ConvNet, self).__init__()

        self.conv_split = nn.Sequential(
            nn.Conv2d(1, 25, (1,10), 1),
            nn.Conv2d(25, 25, (32,1), 1, bias=False),
            )
        self.post_conv = nn.Sequential(
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool1 = nn.Sequential(
            nn.Conv2d(25, 50, (1,10), 1, bias=False),
            nn.BatchNorm2d(50),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool2 = nn.Sequential(
            nn.Conv2d(50, 100, (1,10), 1, bias=False),
            nn.BatchNorm2d(100),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool3 = nn.Sequential(
            nn.Conv2d(100, 200, (1,10), 1, bias=False),
            nn.BatchNorm2d(200),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_fc = nn.Sequential(
            ConstrainedLinear(200*1*1, num_class)
            )

    def forward(self, x):
        out = self.conv_split(x)
        out = self.post_conv(out)
        out = self.conv_pool1(out)
        out = self.conv_pool2(out)
        out = self.conv_pool3(out)
        out = out.view(-1, np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        return out

class EEGNet(nn.Module):
    def __init__(self, num_class=2, bias=False, drop_ratio=.5, F1=8, D=2):
        super(EEGNet, self).__init__()
        F2 = F1*D

        self.conv_temporal = nn.Sequential(
            nn.ZeroPad2d((((250-1)//2)+1, ((250-1)//2), 0, 0)), 
            nn.Conv2d(1, F1, (1,250), 1, bias=bias),
            nn.BatchNorm2d(F1),
            )

        self.conv_spatial = nn.Sequential(
            ConstrainedConv2d(F1, F1*D, (32,1), 1, bias=bias, groups=F1),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(drop_ratio)
            )

        self.conv_separable = nn.Sequential(
            nn.ZeroPad2d((((125-1)//2)+1, ((125-1)//2), 0, 0)),
            nn.Conv2d(F1*D, F2, (1,125), 1, bias=bias, groups=F1*D), #depthwise
            nn.Conv2d(F2, F2, 1, 1),  #pointwise = 1dconv
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)), #(12)
            nn.Dropout(drop_ratio)
            )

        self.conv_fc = nn.Sequential(
            ConstrainedLinear(F2*1*15, num_class)
            #nn.Linear(F2*1*15, num_class) #(16*1*10)
            )

    def forward(self, x):
        out = self.conv_temporal(x)
        out = self.conv_spatial(out)
        out = self.conv_separable(out)
        out = out.view(-1, np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        return out

class EEG_TCNet(nn.Module):
    def __init__(self, bias=False, num_class=2, drop_ratio=.5, F1=8, D=2):
        super(EEG_TCNet, self).__init__()
        F2 = F1*D

        self.conv_temporal = nn.Sequential(
            nn.ZeroPad2d((((250-1)//2)+1, ((250-1)//2), 0, 0)), 
            nn.Conv2d(1, F1, (1,250), 1, bias=bias),
            nn.BatchNorm2d(F1),
            )

        self.conv_spatial = nn.Sequential(
            ConstrainedConv2d(F1, F1*D, (32,1), 1, bias=bias, groups=F1),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(drop_ratio)
            )

        self.conv_separable = nn.Sequential(
            nn.ZeroPad2d((((125-1)//2)+1, ((125-1)//2), 0, 0)),
            nn.Conv2d(F1*D, F2, (1,125), 1, bias=bias, groups=F1*D), #depthwise
            nn.Conv2d(F2, F2, 1, 1),  #pointwise = 1dconv
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)), #(12)
            nn.Dropout(drop_ratio)
            )

        self.conv_fc = nn.Sequential(
            ConstrainedLinear(F2*1*15, num_class)
            #nn.Linear(F2*1*15, num_class) #(16*1*10)
            )

        # TCN-block
        self.tcn_block1 = nn.Sequential(
            nn.ZeroPad2d((2,1,0,0)),
            nn.Conv1d(F2, F2, 4, 1),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.ZeroPad2d((2,1,0,0)),
            nn.Conv1d(F2, F2, 4, 1),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),
            )
        self.tcn_block2 = nn.Sequential(
            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv1d(F2, F2, 4, 1, dilation=2),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv1d(F2, F2, 4, 1, dilation=2),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),
            )

    def forward(self, x):
        out = self.conv_temporal(x)
        out = self.conv_spatial(out)
        out = self.conv_separable(out)
        out = torch.squeeze(out, axis=2)
        tcn = self.tcn_block1(out)
        out = out + tcn
        out = nn.ELU()(out)
        tcn = self.tcn_block2(out)
        out = out + tcn
        out = nn.ELU()(out)
        out = out.view(-1, np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        return out

class CCRNN(nn.Module):
    def __init__(self, num_classes=2, drop_ratio=0.5, nSeg=30):
        super(CCRNN, self).__init__()
        self.nSeg = nSeg
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=(3-1)//2), 
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, padding=(3-1)//2), 
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, padding=(3-1)//2), 
            nn.ELU()
        )
        self.conv_fc = nn.Sequential(
            nn.Linear(128*7*5, 1024),
            nn.ELU(),
            nn.Dropout(drop_ratio)
        )
        self.rnn_module = nn.Sequential(
            nn.LSTM(1024, 64, 2, batch_first=True, dropout=drop_ratio)
        )
        self.rnn_fc = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ELU(),
            nn.Dropout(drop_ratio)
        )
        self.readout = nn.Sequential(
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        out = self.conv_module(x)
        out = out.reshape(out.shape[0], np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        out = out.reshape(-1, self.nSeg, out.shape[-1])
        out, (hn, cn) = self.rnn_module(out)
        out = out[:, -1]
        out = self.rnn_fc(out)
        out = self.readout(out)
        return out
    
    
class sub_Deep_ConvNet(nn.Module):
    def __init__(self, bias=False, drop_ratio=0.5, num_class=2):
        super(sub_Deep_ConvNet, self).__init__()

        self.conv_split = nn.Sequential(
            nn.Conv2d(1, 25, (1,10), 1),
            nn.Conv2d(25, 25, (32,1), 1, bias=False),
            )
        self.post_conv = nn.Sequential(
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool1 = nn.Sequential(
            nn.Conv2d(25, 50, (1,10), 1, bias=False),
            nn.BatchNorm2d(50),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool2 = nn.Sequential(
            nn.Conv2d(50, 100, (1,10), 1, bias=False),
            nn.BatchNorm2d(100),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool3 = nn.Sequential(
            nn.Conv2d(100, 200, (1,10), 1, bias=False),
            nn.BatchNorm2d(200),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_pool3 = nn.Sequential(
            nn.Conv2d(100, 200, (1,10), 1, bias=False),
            nn.BatchNorm2d(200),
            nn.MaxPool2d((1,3), 3),
            nn.Dropout(0.3)
            )
        self.conv_fc = nn.Sequential(
            ConstrainedLinear(200*1*1, 1024),
            nn.Dropout(drop_ratio),
            ConstrainedLinear(1024, 512),
            nn.Dropout(drop_ratio),
            ConstrainedLinear(512, num_class)
            )

    def forward(self, x):
        out = self.conv_split(x)
        out = self.post_conv(out)
        out = self.conv_pool1(out)
        out = self.conv_pool2(out)
        out = self.conv_pool3(out)
        out = out.view(-1, np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        return out
    
    
class sub_EEGNet(nn.Module):
    def __init__(self, drop_ratio=.5, bias=False, num_class=2, F1=8, D=2):
        super(sub_EEGNet, self).__init__()
        F2 = F1*D

        self.conv_temporal = nn.Sequential(
            nn.ZeroPad2d((((250-1)//2)+1, ((250-1)//2), 0, 0)), 
            nn.Conv2d(1, F1, (1,250), 1, bias=bias),
            nn.BatchNorm2d(F1),
            )

        self.conv_spatial = nn.Sequential(
            ConstrainedConv2d(F1, F1*D, (32,1), 1, bias=bias, groups=F1), 
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(drop_ratio)
            )

        self.conv_separable = nn.Sequential(
            nn.ZeroPad2d((((125-1)//2)+1, ((125-1)//2), 0, 0)),
            nn.Conv2d(F1*D, F2, (1,125), 1, bias=bias, groups=F1*D), #depthwise
            nn.Conv2d(F2, F2, 1, 1),  #pointwise = 1dconv
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)), #(12)
            nn.Dropout(drop_ratio)
            )
        
        self.conv_fc = nn.Sequential(
            ConstrainedLinear(F2*1*15, 1024),
            #nn.Linear(F2*1*15, 1024),
            nn.Dropout(drop_ratio),
            ConstrainedLinear(1024, 512),
            #nn.Linear(1024, 512),
            nn.Dropout(drop_ratio),
            ConstrainedLinear(512, num_class)
            #nn.Linear(512, num_class)
            )

    def forward(self, x):
        out = self.conv_temporal(x)
        out = self.conv_spatial(out)
        out = self.conv_separable(out)
        out = out.view(-1, np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        return out

class sub_EEG_TCNet(nn.Module):
    def __init__(self, bias=False, drop_ratio=.5, num_class=2, F1=8, D=2):
        super(sub_EEG_TCNet, self).__init__()
        F2 = F1*D

        self.conv_temporal = nn.Sequential(
            nn.ZeroPad2d((((250-1)//2)+1, ((250-1)//2), 0, 0)), 
            nn.Conv2d(1, F1, (1,250), 1, bias=bias),
            nn.BatchNorm2d(F1),
            )

        self.conv_spatial = nn.Sequential(
            ConstrainedConv2d(F1, F1*D, (32,1), 1, bias=bias, groups=F1),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(drop_ratio)
            )

        self.conv_separable = nn.Sequential(
            nn.ZeroPad2d((((125-1)//2)+1, ((125-1)//2), 0, 0)),
            nn.Conv2d(F1*D, F2, (1,125), 1, bias=bias, groups=F1*D), #depthwise
            nn.Conv2d(F2, F2, 1, 1),  #pointwise = 1dconv
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)), #(12)
            nn.Dropout(drop_ratio)
            )

        self.conv_fc = nn.Sequential(
            ConstrainedLinear(F2*1*15, 1024),
            nn.Dropout(drop_ratio),
            ConstrainedLinear(1024, 512),
            nn.Dropout(drop_ratio),
            ConstrainedLinear(512, num_class)
            )

        # TCN-block
        self.tcn_block1 = nn.Sequential(
            nn.ZeroPad2d((2,1,0,0)),
            nn.Conv1d(F2, F2, 4, 1),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.ZeroPad2d((2,1,0,0)),
            nn.Conv1d(F2, F2, 4, 1),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),
            )
        self.tcn_block2 = nn.Sequential(
            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv1d(F2, F2, 4, 1, dilation=2),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.ZeroPad2d((3,3,0,0)),
            nn.Conv1d(F2, F2, 4, 1, dilation=2),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(0.3),
            )

    def forward(self, x):
        out = self.conv_temporal(x)
        out = self.conv_spatial(out)
        out = self.conv_separable(out)
        out = torch.squeeze(out, axis=2)
        tcn = self.tcn_block1(out)
        out = out + tcn
        out = nn.ELU()(out)
        tcn = self.tcn_block2(out)
        out = out + tcn
        out = nn.ELU()(out)
        out = out.view(-1, np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        return out

class sub_CCRNN(nn.Module):
    def __init__(self, drop_ratio=0.5, nSeg=30, num_classes=2):
        super(sub_CCRNN, self).__init__()
        self.nSeg = nSeg
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=(3-1)//2), 
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, padding=(3-1)//2),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, padding=(3-1)//2), 
            nn.ELU()
        )
        self.conv_fc = nn.Sequential(
            nn.Linear(128*7*5, 1024),
            nn.ELU(),
            nn.Dropout(drop_ratio)
        )
        self.rnn_module = nn.Sequential(
            nn.LSTM(1024, 64, 2, batch_first=True, dropout=drop_ratio)
        )
        self.rnn_fc = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ELU(),
            nn.Dropout(drop_ratio)
        )
        self.readout = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Dropout(drop_ratio),
            nn.Linear(128, 128),
            nn.Dropout(drop_ratio),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        out = self.conv_module(x)
        out = out.reshape(out.shape[0], np.prod(out.shape[1:]))
        out = self.conv_fc(out)
        out = out.reshape(-1, self.nSeg, out.shape[-1])
        out, (hn, cn) = self.rnn_module(out)
        out = out[:, -1]
        out = self.rnn_fc(out)
        out = self.readout(out)
        return out
