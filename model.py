import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        # depthwise conv
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.01),
        )
        # pointwise conv
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.01),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop_prob = 0.01
        # Convolution Block 1 (standard convolutions only)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias = False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.BatchNorm2d(32),
            nn.ReLU(),            
            nn.Dropout2d(self.drop_prob),

            nn.Conv2d(32, 64, 3, padding=1, bias = False), # Input: 32x32x32 | Output: 32x32x64 | RF: 5x5
            nn.BatchNorm2d(64),
            nn.ReLU(),            
            nn.Dropout2d(self.drop_prob),
            # extra standard conv to increase RF cheaply
            #nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
        )
        self.sc1 = nn.Sequential(
            nn.Conv2d(64, 32,1, stride=2), # Input: 32x32x64 | Output: 16x16x32 | RF: 5x5
            nn.ReLU(),
        )

        # Convolution Block 2: depthwise separable (only DS layer in network)
        # 32x32x48 -> 32x32x64
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3,  padding=1, bias = False), # Input: 16x16x32 | Output: 16x16x32 | RF: 9x9
            nn.BatchNorm2d(32),
            nn.ReLU(),            
            nn.Dropout2d(self.drop_prob),

            ## Depthwise Seperable Convolution1
            nn.Conv2d(32,32, 3,  padding=1,groups=32 ,bias = False),  # Input: 16x16x32 | Output: 16x16x32 | RF: 13x13
            nn.Conv2d(32, 64, 1, padding=0, bias = False),   # Input: 16x16x32 | Output: 18x18x64 | RF: 13x13
            nn.BatchNorm2d(64),
            nn.ReLU(),            
            nn.Dropout2d(self.drop_prob),
        )
        self.sc2 = nn.Sequential(

            nn.Conv2d(64, 32, 1, stride=2), # Input: 18x18x32 | Output: 9x9x64 | RF: 13x13
            nn.ReLU()
        )
        

        # Convolution Block 3: single dilated convolution to ensure RF > 44
        # 32x32x64 -> 32x32x96
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3,  padding=1, bias = False,dilation=2), # Input: 9x9x64 | Output: 7x7x64 | RF: 29x29
            nn.BatchNorm2d(64),
            nn.ReLU(),            
            nn.Dropout2d(self.drop_prob),

            nn.Conv2d(64, 64, 3,  padding=1, bias = False),  # Input: 7x7x64| Output: 7x7x64 | RF: 45x45
            nn.BatchNorm2d(64),
            nn.ReLU(),            
            nn.Dropout2d(self.drop_prob),

            
        )
        self.sc3 = nn.Sequential(
            nn.Conv2d(64, 16, 1, stride=2), # Input: 7x7x64| Output: 4x4x16 | RF: 61x61
            nn.ReLU()
        )        

        # Convolution Block 4: bottleneck with stride=2 in the 3x3 (reduces params)
        # 32x32x96 -> 16x16x128
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias = False), # Input: 4x4x16 | Output: 4x4x32 | RF: 93x93
            nn.BatchNorm2d(32),
            nn.ReLU(),            
            nn.Dropout2d(self.drop_prob),

            ## Depthwise seperable Convolution2
            nn.Conv2d(32,32, 3,  padding=1,groups=32 ,bias = False),# Input: 4x4x16 | Output: 4x4x32 | RF: 125x125
            nn.Conv2d(32, 10, 1, padding=0, bias = False),          # Input: 4x4x32| Output: 6x6x10 | RF: 125x125
            # nn.ReLU(),
            # nn.BatchNorm2d(10),
        )        

        # Head: GAP then FC to 10 classes
        self.gap = nn.AdaptiveAvgPool2d(1)
        #self.dropout = nn.Dropout(p=0.1)
        #self.fc = nn.Linear(128, 10)


    def forward(self, x):

        x = self.conv_block1(x)
        x = self.sc1(x)

        x = self.conv_block2(x) 
        x = self.sc2(x) 

        x = self.conv_block3(x) 
        x = self.sc3(x)

        x = self.conv_block4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        #x = self.fc(x)
        
        return F.log_softmax(x,dim=1)