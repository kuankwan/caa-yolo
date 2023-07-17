import torch
import torch.nn as nn
from mmdet.utils import OptMultiConfig
from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from mmengine.model import BaseModule
from abc import ABCMeta, abstractmethod
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
import cv2
import torch.nn.functional as F

@MODELS.register_module()
class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)

@MODELS.register_module()
class D_layer2(BaseModule):
    def __init__(self,in_channel:int,grad_reverse_lambda=0.1,init_cfg: OptMultiConfig = None):
        super(D_layer2,self).__init__(init_cfg)
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel, int(self.in_channel), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(int(self.in_channel), int(self.in_channel), kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(self.in_channel), int(self.in_channel), kernel_size=3, stride=1, padding=1)
        self.classier = nn.Conv2d(int(self.in_channel), 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.relu = nn.SiLU(inplace=True)
        self.LabelResizeLayer = ImageLabelResizeLayer()
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,0,0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classier.bias, bias_value)

    def forward(self, x,need_backprop):
        x = grad_reverse(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.classier(x)
        if need_backprop is not None:
            label=self.LabelResizeLayer(x,need_backprop)
        else:
            label=None
        return x,label

@MODELS.register_module()
class D_layer3(BaseModule):
    def __init__(self,in_channel:int,grad_reverse_lambda=0.1,init_cfg: OptMultiConfig = None):
        super(D_layer3,self).__init__(init_cfg)
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel, int(self.in_channel), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(int(self.in_channel), int(self.in_channel), kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(self.in_channel), int(self.in_channel), kernel_size=3, stride=1, padding=1)
        self.classier = nn.Conv2d(int(self.in_channel), 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.relu = nn.SiLU(inplace=True)
        self.LabelResizeLayer = ImageLabelResizeLayer()
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,0,0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classier.bias, bias_value)

    def forward(self, x,need_backprop):
        x = grad_reverse(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.classier(x)
        if need_backprop is not None:
            label=self.LabelResizeLayer(x,need_backprop)
        else:
            label=None
        return x,label

@MODELS.register_module()
class D_layer4(BaseModule):
    def __init__(self,in_channel:int,grad_reverse_lambda=0.1,init_cfg: OptMultiConfig = None):
        super(D_layer4,self).__init__(init_cfg)
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(int(self.in_channel), int(self.in_channel), kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(self.in_channel), int(self.in_channel), kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(int(self.in_channel), int(self.in_channel), kernel_size=3, stride=1, padding=1)
        self.classier = nn.Conv2d(int(self.in_channel), 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.relu = nn.SiLU(inplace=True)
        self._init_weights()
        self.LabelResizeLayer = ImageLabelResizeLayer()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,0,0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classier.bias, bias_value)

    def forward(self, x,need_backprop):
        x = grad_reverse(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.classier(x)
        if need_backprop is not None:
            label=self.LabelResizeLayer(x,need_backprop)
        else:
            label=None
        return x,label

@MODELS.register_module()
class Local_D(BaseModule):
    def __init__(self,in_channel,grad_reverse_lambda=1.0,init_cfg: OptMultiConfig = None):
        super(Local_D,self).__init__(init_cfg)
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,stride=1,padding=1)
        self.classifer = nn.Conv2d(self.in_channel,1,kernel_size=3,padding=1)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,0,0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classifer.bias, bias_value)


    def forward(self,x):
        ins_feat = grad_reverse(x)
        mask = self.relu(self.conv1(ins_feat))
        mask = self.relu(self.conv2(mask))
        mask = self.classifer(mask)
        # mask = F.avg_pool1d(mask,(mask[2],mask[3]))
        return mask

@MODELS.register_module()
class Global_D(BaseModule):
    def __init__(self,in_channel,grad_reverse_lambda=1.0,init_cfg: OptMultiConfig = None):
        super(Global_D,self).__init__(init_cfg)
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.classifer = nn.Conv2d(self.in_channel, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,0,0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.classifer.bias, bias_value)


    def forward(self,x):
        x = grad_reverse(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.classifer(x))
        x = self.avgpool(x)

        return x
@MODELS.register_module()
class ImageLabelResizeLayer(nn.Module):
    """
    Resize label to be the same size with the samples
    """
    def __init__(self):
        super(ImageLabelResizeLayer, self).__init__()


    def forward(self,x,need_backprop):

        feats = x.detach().cpu().numpy()
        lbs = need_backprop.detach().cpu().numpy()
        gt_blob = np.zeros((lbs.shape[0], feats.shape[2], feats.shape[3], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            lb=np.array([lbs[i]])
            lbs_resize = cv2.resize(lb, (feats.shape[3] ,feats.shape[2]),  interpolation=cv2.INTER_NEAREST)
            gt_blob[i, 0:lbs_resize.shape[0], 0:lbs_resize.shape[1], 0] = lbs_resize

        channel_swap = (0, 3, 1, 2)
        gt_blob = gt_blob.transpose(channel_swap)
        y=Variable(torch.from_numpy(gt_blob)).cuda()
        # y=y.squeeze(1).long()
        return y