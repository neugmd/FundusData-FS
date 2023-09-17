""" ResNet with MTL. """
import torch.nn as nn
import pdb
from models.conv2d_mtl import Conv2dMtl

class ResNetMtl(nn.Module):

    def __init__(self, mtl=True):
        super(ResNetMtl, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl            
        else:
            self.Conv2d = nn.Conv2d
        # Block 1 -------------------------------------------------------------- 
        self.conv1_1 = self.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        
        self.conv1_2 = self.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        
        self.conv1_3 = self.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.relu1_3 = nn.ReLU(inplace=True)
        
        self.conv1_4 = self.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        self.bn1_4 = nn.BatchNorm2d(64)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Block 2 -------------------------------------------------------------- 
        self.conv2_1 = self.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        
        self.conv2_2 = self.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        
        self.conv2_3 = self.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(128)
        self.relu2_3 = nn.ReLU(inplace=True)
        
        self.conv2_4 = self.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.bn2_4 = nn.BatchNorm2d(128)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Block 3 -------------------------------------------------------------- 
        self.conv3_1 = self.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        
        self.conv3_2 = self.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        
        self.conv3_3 = self.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        
        self.conv3_4 = self.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.bn3_4 = nn.BatchNorm2d(256)
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        #Block 4 -------------------------------------------------------------- 
        self.conv4_1 = self.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        
        self.conv4_2 = self.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        
        self.conv4_3 = self.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        
        self.conv4_4 = self.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.bn4_4 = nn.BatchNorm2d(512)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
     
    def forward(self, x):
        # Block 1 ---------------------------------  
        #pdb.set_trace()
        # layer 1
        x1_1 = self.conv1_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu1_1(x1_1)
        # layer 2
        x1_2 = self.conv1_2(x1_1)
        x1_2 = self.bn1_2(x1_2)
        x1_2 = self.relu1_2(x1_2)
        # layer 3
        x1_3 = self.conv1_3(x1_2)
        x1_3 = self.bn1_3(x1_3)
        x1_3 = self.relu1_3(x1_3)       
        
        x1_4 = self.bn1_4(self.conv1_4(x))
        x1_4 = x1_3 + x1_4
        x1_4 = self.maxpool1(x1_4)       
        
        # Block 2 ---------------------------------  
        # layer 4
        x2_1 = self.conv2_1(x1_4)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu2_1(x2_1)
        # layer 5
        x2_2 = self.conv2_2(x2_1)
        x2_2 = self.bn2_2(x2_2)
        x2_2 = self.relu2_2(x2_2)
        # layer 6
        x2_3 = self.conv2_3(x2_2)
        x2_3 = self.bn2_3(x2_3)
        x2_3 = self.relu2_3(x2_3)       
        
        x2_4 = self.bn2_4(self.conv2_4(x1_4))
        x2_4 = x2_3 + x2_4
        x2_4 = self.maxpool1(x2_4)
        
        # Block 3 --------------------------------- 
        # layer 7       
        x3_1 = self.conv3_1(x2_4)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu3_1(x3_1)
        # layer 8
        x3_2 = self.conv3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)
        x3_2 = self.relu3_2(x3_2)
        # layer 9
        x3_3 = self.conv3_3(x3_2)
        x3_3 = self.bn3_3(x3_3)
        x3_3 = self.relu3_3(x3_3)       
        
        x3_4 = self.bn3_4(self.conv3_4(x2_4))
        x3_4 = x3_3 + x3_4
        x3_4 = self.maxpool1(x3_4)
        
        # Block 4 ---------------------------------
        # layer 10        
        x4_1 = self.conv4_1(x3_4)
        x4_1 = self.bn4_1(x4_1)
        x4_1 = self.relu4_1(x4_1)
        # layer 11
        x4_2 = self.conv4_2(x4_1)
        x4_2 = self.bn4_2(x4_2)
        x4_2 = self.relu4_2(x4_2)
        # layer 12
        x4_3 = self.conv4_3(x4_2)
        x4_3 = self.bn4_3(x4_3)
        x4_3 = self.relu4_3(x4_3)       
        
        x4_4 = self.bn4_4(self.conv4_4(x3_4))
        x4_4 = x4_3 + x4_4
        x4_4 = self.maxpool1(x4_4)
                 
        # 6*6
        x4 = self.avgpool(x4_4)
        x4 = x4.view(x4.size(0), -1)
        
        x43 = self.avgpool(x4_3)
        x43 = x43.view(x43.size(0), -1)
        
        x42 = self.avgpool(x4_2)
        x42 = x42.view(x42.size(0), -1)
        
        x41 = self.avgpool(x4_1)
        x41 = x41.view(x41.size(0), -1)
        
        # add multi-scale output        
        # 11*11
        x3 = self.avgpool(x3_4)
        x3 = x3.view(x3.size(0), -1)
        # 21*21
        x2 = self.avgpool(x2_4)
        x2 = x2.view(x2.size(0), -1)
        # 41*41
        x1 = self.avgpool(x1_4)
        x1 = x1.view(x1.size(0), -1)

        return x4, x43, x42, x41, x3, x2, x1
        
