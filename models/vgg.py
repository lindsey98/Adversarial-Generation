import torch.nn as nn
import torch
from torchvision import transforms
import torchvision

# Initialize the network
class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv1 = self._make_layer(3, 64, maxpool=True)
        self.conv2 = self._make_layer(64, 128, maxpool=True)
        self.conv3 = self._make_layer(128, 256, maxpool=False)
        self.conv4 = self._make_layer(256, 256, maxpool=True)
        self.conv5 = self._make_layer(256, 512, maxpool=False)
        self.conv6 = self._make_layer(512, 512, maxpool=True)
        self.conv7 = self._make_layer(512, 512, maxpool=False)
        self.conv8 = self._make_layer(512, 512, maxpool=True)
        self.avgpool = nn.AvgPool2d(kernel_size=1)
        self.classifier = nn.Linear(512, 10)
        
    def _make_layer(self, in_dim, out_dim, maxpool=False):
        if maxpool:
            return nn.Sequential(
                        nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            return nn.Sequential(
                        nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True))
        
    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2) 
        o4 = self.conv4(o3)
        o5 = self.conv5(o4)
        o6 = self.conv6(o5)
        o7 = self.conv7(o6)
        o8 = self.conv8(o7)
        
        o9 = self.avgpool(o8)
        o9 = o9.view(-1, torch.prod(torch.tensor(o9.shape)[1:]))
        
        out = self.classifier(o9)
        return out, (o1, o2, o3, o4, o5, o6, o7, o8, o9)