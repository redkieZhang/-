# -*- coding: utf-8 -*-
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from statistics import mode

curDir=os.path.dirname(os.path.abspath(__file__))

# 人脸数据归一化,将像素值从0-255映射到0-1之间
def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images/255.0
    return images

class ResNet(nn.Module):
    def __init__(self, *args):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0],-1)

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

# 残差神经网络
class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

    
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)
    
class EmotionDetector:

    def __init__(self,model_path=os.path.join(curDir,'model/model_resnet.pkl')):
        resnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7 , stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
        resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
        resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
        resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
        resnet.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
        resnet.add_module("fc", nn.Sequential(ResNet(), nn.Linear(512, 7)))
        self.emotion_classifier = torch.load(model_path)
        self.frame_window = 10
        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
        self.emotion_window = []
        
    def detect(self,face):
        grayFace = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        try:
            # shape变为(48,48)
            grayFace = cv2.resize(grayFace,(48,48))
        except:
            return None
        
        # 扩充维度，shape变为(1,48,48,1)
        #将（1，48，48，1）转换成为(1,1,48,48)
        grayFace = np.expand_dims(grayFace,0)
        grayFace = np.expand_dims(grayFace,0)
        
        # 人脸数据归一化，将像素值从0-255映射到0-1之间
        grayFace = preprocess_input(grayFace)
        new_face=torch.from_numpy(grayFace)
        new_new_face = new_face.float().requires_grad_(False)
        
        # 调用我们训练好的表情识别模型，预测分类
        emotion_arg = np.argmax(self.emotion_classifier.forward(new_new_face).detach().numpy())
        emotion = self.emotion_labels[emotion_arg]
        return emotion
        #self.emotion_window.append(emotion)
        #
        #if len(self.emotion_window) >= self.frame_window:
        #    self.emotion_window.pop(0)
        #try:
        #    # 获得出现次数最多的分类
        #    emotion_mode = mode(self.emotion_window)
        #    return emotion_mode
        #except:
        #    return None