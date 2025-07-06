from mmengine.model import BaseModule
from mmseg.registry import MODELS
import torch.nn as nn


@MODELS.register_module()
class DirectReduction(BaseModule):
    def __init__(self, output_channel=1):
        super(DirectReduction, self).__init__()
        self.conv = nn.Conv2d(in_channels=480, out_channels=output_channel, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv(x)
        return x


@MODELS.register_module()
class GradualReduction(BaseModule):
    def __init__(self, output_channel=1):
        super(GradualReduction, self).__init__()
        self.conv1 = nn.Conv2d(480, 128, kernel_size=1)  # 第一步降维到128
        self.relu = nn.ReLU()  # 激活函数
        self.conv2 = nn.Conv2d(128, output_channel, kernel_size=1)  # 第二步降维到1
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


@MODELS.register_module()
class SEBlock(BaseModule):
    '''
     https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
     Squeeze-and-Excitation Networks
     '''
    def __init__(self, input_channels, output_channels=1, reduction=16):
        super(SEBlock, self).__init__()
        self.se = SELayer(input_channels, reduction)
        # 修改处：将通道数改变的卷积层移动到SE层之后
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.se(x)
        x = self.conv(x)
        return x
