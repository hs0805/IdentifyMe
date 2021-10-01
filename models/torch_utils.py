import io
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as transforms 
from PIL import Image
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, res_block=None):
        super(ResidualBlock, self).__init__()

        self.layer = self._make_layer(in_planes, planes)
        self.res_block = None
        if not res_block is None:
            self.res_block = nn.Sequential(
                res_block(planes, planes)
            )
    
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.layer(x)
        if not self.res_block is None:
            x = x + self.res_block(x)
        return x


class ResModel(nn.Module):

    def __init__(self, block, res_block):
        super(ResModel, self).__init__()

        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block_layers = nn.Sequential(
            block(64, 128, res_block=res_block),
            block(128, 256),
            block(256, 512, res_block=res_block)
        )
        
        self.pool = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.prep_layer(x)
        x = self.block_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def ResidualModel():
    return ResModel(ResidualBlock, BasicBlock)

PATH = "models/model.pt"
print('loading model ... ')
model = ResidualModel()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(bytearray(image_bytes)))
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))])

    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    images = image_tensor.reshape(-1, 3, 32, 32)

    outputs = model(images)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return classes[predicted.item()]

# image = open('vishwa/cat.jpg', 'rb').read()
# tensor = transform_image(image)
# prediction = get_prediction(tensor)
# print(prediction)

