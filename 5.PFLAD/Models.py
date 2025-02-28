import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #kernel1(10,1,5,5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #kernel2(20,10,5,5)
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(320,10) # weight(320,10)
     
    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        batch_size = tensor.size(0)
        tensor = self.pooling(F.relu(self.conv1(tensor)))
        tensor = self.pooling(F.relu(self.conv2(tensor)))
        tensor = tensor.view(batch_size,-1) # Flatten
        return self.fc(tensor)  


# Local Feature Extraction Model
class LinearNet1(nn.Module): 
    def __init__(self, dimen):
        super(LinearNet1,self).__init__()
        self.dimen = dimen
        self.linear1 = nn.Linear(self.dimen, self.dimen//10)
        self.linear2 = nn.Linear(self.dimen//10, self.dimen//100)
        self.linear3 = nn.Linear(self.dimen//100, 1)

        
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x**2


# Local Feature Extraction Model
class LinearNet2(nn.Module): 
    def __init__(self, dimen):
        super(LinearNet2,self).__init__()
        self.dimen = dimen
        self.linear1 = nn.Linear(self.dimen, self.dimen//10)
        self.linear2 = nn.Linear(self.dimen//10, self.dimen//100)
        self.linear3 = nn.Linear(self.dimen//100, self.dimen//1000)
        self.linear4 = nn.Linear(self.dimen//1000, 1)

        
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x**2


class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        stride = 1
        _features = out_features
        if self.in_features != self.out_features:
            if self.out_features / self.in_features == 2.0:
                stride = 2  
            else:
                raise ValueError("The number of output features is at most two times the number of input features!")
 
        self.conv1 = nn.Conv2d(in_features, _features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(_features, _features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
 
        self.downsample = None if self.in_features == self.out_features else nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
 
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
 
        # Downsampling layer is used when the number of features in the input and output is different
        if self.in_features != self.out_features:
            identity = self.downsample(x)
 
        # Summation of residuals
        out += identity
        out = self.relu(out)
        return out
 
 
class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=10, bias=True)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # output: Tensor (64,512,1,1)
        x = torch.flatten(x, 1)  # output: Tensor (64,512)
        x = self.fc(x)  # output: Tensor (64,10)
        return x
 