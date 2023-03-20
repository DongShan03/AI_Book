import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable


class MobileNet(nn.Module):
    def __init__(self, n_class=1000) -> None:
        super().__init__()
        self.n_class = n_class
    
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
    
        self.model = nn.Sequential(     #224 * 224 * 3
            conv_bn(3, 32, 2),          #112 * 112 * 32
            conv_dw(32, 64, 1),         #112 * 112 * 64
            conv_dw(64, 128, 2),        #56 * 56 * 128
            conv_dw(128, 128, 1),       #56 * 56 * 128
            conv_dw(128, 256, 2),       #28 * 28 * 256
            conv_dw(256, 256, 1),       #28 * 28 * 256
            conv_dw(256, 512, 2),       #14 * 14 * 512
            conv_dw(512, 512, 1),       #14 * 14 * 512 
            conv_dw(512, 512, 1),       #14 * 14 * 512
            conv_dw(512, 512, 1),       #14 * 14 * 512
            conv_dw(512, 512, 1),       #14 * 14 * 512
            conv_dw(512, 512, 1),       #14 * 14 * 512
            conv_dw(512, 1024, 2),      #7 * 7 * 1024
            conv_dw(1024, 1024, 1),     #7 * 7 * 1024   
            nn.AvgPool2d(7),            #1 * 1 * 1024
        )

        self.fc = nn.Linear(1024, self.n_class)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
    


mobilenet = MobileNet().cuda()


from torchsummary import summary

summary(mobilenet, input_size=(3, 224, 224))





