import torch
import torch.nn as nn
import torch.nn.init as init
from torchsummary import summary


cfg = [32, 'M', 64, 'M', 128, 64, 128, 'M', 256, 128, 256, 'M',
       512, 256, 512, 256, 512, 'M', 1024, 512, 1024, 512, 1024]


def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    flag = True
    in_channels = in_channels
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=v,
                                    kernel_size=(1, 3)[flag], stride=1,
                                    padding=(0, 1)[flag], bias=False))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            in_channels = v
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        
        flag = not flag
    return nn.Sequential(*layers)


class Darknet19(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, batch_norm=True,
                    pretrain=False):
        super(Darknet19, self).__init__()
        self.features = make_layers(cfg, in_channels, batch_norm)
        self.classifier = nn.Sequential( 
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(output_size=(1)),
            nn.Softmax(dim=0),
        )
        if pretrain:
            pass
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)


net = Darknet19(num_classes=1000).to("cuda")


summary(net, input_size=(3, 224, 224))


