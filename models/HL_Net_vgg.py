import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class HLNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    # extra_conv_layers and extra_smooth_layers are in reverse way
    def __init__(self, phase, size, base, extras, extra_conv_layers, extra_smooth_layers, head_1, head_2, num_classes, C_agnostic):
        super(HLNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.C_agnostic = C_agnostic
        self.size = size

        # if size == 300:
        #     self.indicator = 3
        # elif size == 512:
        #     self.indicator = 5
        # else:
        #     print("Error: Sorry only SSD300 and SSD512 are supported!")
        #     return
        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4
        self.Conv4_3_Norm = L2Norm(512, 10)
        self.Conv5_3_Norm = L2Norm(512, 8)
        self.extras = nn.ModuleList(extras)

        self.extra_conv_layers = nn.ModuleList(extra_conv_layers)
        self.extra_smooth_layers = nn.ModuleList(extra_smooth_layers)

        self.loc_1 = nn.ModuleList(head_1[0])
        self.conf_1 = nn.ModuleList(head_1[1])
        self.loc_2 = nn.ModuleList(head_2[0])
        self.conf_2 = nn.ModuleList(head_2[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources_1 = list()
        sources_2 = list()
        loc_1 = list()
        loc_2 = list()
        conf_1 = list()
        conf_2 = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        f0 = self.Conv4_3_Norm(x)

        # apply vgg up to conv5_3 relu
        for k in range(23, 30):
            x = self.base[k](x)

        f1 = self.Conv5_3_Norm(x)

        # apply vgg up to fc7
        for k in range(30, len(self.base)):
            x = self.base[k](x)

        f2 = x

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                f3 = x

        feat3 = self.extra_conv_layers[3](f3)
        feat3 = self.extra_smooth_layers[3](feat3)

        feat2 = self.extra_conv_layers[2](f2)
        feat2 = self._upsample_add(feat3, feat2)
        feat2 = self.extra_smooth_layers[2](feat2)

        feat1 = self.extra_conv_layers[1](f1)
        feat1 = self._upsample_add(feat2, feat1)
        feat1 = self.extra_smooth_layers[1](feat1)

        feat0 = self.extra_conv_layers[0](f0)
        feat0 = self._upsample_add(feat1, feat0)
        feat0 = self.extra_smooth_layers[0](feat0)

        sources_1 = [f0, f1, f2, f3]
        sources_2 = [feat0, feat1, feat2, feat3]

        # apply multibox head to source layers
        for (x, l, c) in zip(sources_1, self.loc_1, self.conf_1):
            loc_1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_1.append(c(x).permute(0, 2, 3, 1).contiguous())
        for (x, l, c) in zip(sources_2, self.loc_2, self.conf_2):
            loc_2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_2.append(c(x).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])

        loc_1 = torch.cat([o.view(o.size(0), -1) for o in loc_1], 1)
        loc_2 = torch.cat([o.view(o.size(0), -1) for o in loc_2], 1)
        conf_1 = torch.cat([o.view(o.size(0), -1) for o in conf_1], 1)
        conf_2 = torch.cat([o.view(o.size(0), -1) for o in conf_2], 1)

        if self.phase == "test":
            output_1 = (
                loc_1.view(loc_1.size(0), -1, 4),                   # loc preds
                self.softmax(conf_1.view(-1, 2 if self.C_agnostic else self.num_classes)),  # conf preds
            )
            output_2 = (
                loc_2.view(loc_2.size(0), -1, 4),                   # loc preds
                self.softmax(conf_2.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output_1 = (
                loc_1.view(loc_1.size(0), -1, 4),
                conf_1.view(conf_1.size(0), -1, 2 if self.C_agnostic else self.num_classes),
            )
            output_2 = (
                loc_2.view(loc_2.size(0), -1, 4),
                conf_2.view(conf_2.size(0), -1, self.num_classes),
            )
        return output_1, output_2

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        
        Args:
            x: (Variable) top feature map to be upsampled.
            y: (Variable) lateral feature map.

            Returns:
                (Variable) added feature map.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k+1],
                           kernel_size=(1,3)[flag], stride=2, padding=1)]
                # layers += [nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1,3)[flag])]
                # layers += [nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    # if size == 512:
    #     layers += [BasicConv(256,128,kernel_size=1,stride=1)]
    #     layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1)]
    # elif size ==300:
    #     layers += [BasicConv(256,128,kernel_size=1,stride=1)]
    #     layers += [BasicConv(128,256,kernel_size=3,stride=1)]
    #     layers += [BasicConv(256,128,kernel_size=1,stride=1)]
    #     layers += [BasicConv(128,256,kernel_size=3,stride=1)]
    # else:
    #     print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
    #     return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    # '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
    '512': [256, 'S', 512],
    '320': [256, 'S', 512],
}

def conv_smooth(size, vgg, extra_layers):
    extra_conv_layers = []
    extra_smooth_layers = []
    # should be modified if the structure changed
    # at least reverse way
    vgg_source = [21, 28, -2]

    for k, v in enumerate(vgg_source):
        extra_conv_layers += [nn.Sequential(nn.Conv2d(vgg[v].out_channels, 256,
                                            kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 256,
                                            kernel_size=3, padding=1))]
        extra_smooth_layers += [nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))]
    for k, v in enumerate(extra_layers[-1::-2], 2):
        # for the last layer, we should with no upsampling
        extra_conv_layers += [nn.Sequential(nn.Conv2d(v.out_channels, 256,
                                            kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 256,
                                            kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True))]
        extra_smooth_layers += [nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))]

    return extra_conv_layers, extra_smooth_layers



def multibox(size, vgg, extra_layers, extra_conv_layers, extra_smooth_layers, cfg, num_classes, C_agnostic):
    loc_1 = []
    conf_1 = []
    loc_2 = []
    conf_2 = []
    vgg_source = [21, 28, -2] # conv4_3, conv5_3 and fc7
    for k, v in enumerate(vgg_source):
        loc_1 += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4,
                                kernel_size=3, padding=1)]
        conf_1 += [nn.Conv2d(vgg[v].out_channels, cfg[k] * (2 if C_agnostic else num_classes),
                                kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[-1::-2], 2):
        loc_1 += [nn.Conv2d(v.out_channels, cfg[k] * 4,
                                kernel_size=3, padding=1)]
        conf_1 += [nn.Conv2d(v.out_channels, cfg[k] * (2 if C_agnostic else num_classes),
                                kernel_size=3, padding=1)]
    for k in range(len(extra_smooth_layers)):
        loc_2 += [nn.Conv2d(256,
                             cfg[k] * 4, kernel_size=3, padding=1)]
        conf_2 +=[nn.Conv2d(256,
                             cfg[k] * num_classes, kernel_size=3, padding=1)]
        
    return vgg, extra_layers, extra_conv_layers, extra_smooth_layers, (loc_1, conf_1), (loc_2, conf_2)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    # '512': [6, 6, 6, 6, 6, 4, 4],
    '512': [3, 3, 3, 3],
    '320': [3, 3, 3, 3],
}


def build_net(phase, size=320, num_classes=21, C_agnostic=False):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 320 and size != 512:
        print("Error: Sorry only 320 or 512 are supported!")
        return
    vgg_layers = vgg(base[str(size)], 3)
    extra_layers = add_extras(size, extras[str(size)], 1024)
    extra_conv_layers, extra_smooth_layers = conv_smooth(size, vgg_layers, extra_layers)

    return HLNet(phase, size, *multibox(size, vgg_layers,
                                extra_layers, extra_conv_layers, extra_smooth_layers,
                                mbox[str(size)], num_classes, C_agnostic), num_classes, C_agnostic)
