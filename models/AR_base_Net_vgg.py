import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class SINGLEARNet(nn.Module):
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

    def __init__(self, phase, size, base, extras, ar_head, head, num_classes):
        super(SINGLEARNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
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

        self.conv1x1 = nn.ModuleList(ar_head[0])
        self.conv3x3 = nn.ModuleList(ar_head[1])
        self.conv5x5 = nn.ModuleList(ar_head[2])
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.weight = nn.ModuleList(head[2])
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
        sources = list()
        sources_2 = list()
        loc = list()
        conf = list()
        weight = list()
        weight_volatile = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.Conv4_3_Norm(x)
        sources.append(s)

        # apply vgg up to conv5_3 relu
        for k in range(23, 30):
            x = self.base[k](x)

        s = self.Conv5_3_Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.base)):
            x = self.base[k](x)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, w) in zip(sources, self.weight):
            temp = w(x)
            weight.append(temp.permute(0, 2, 3, 1).contiguous())
            weight_volatile.append(Variable(temp.data.clone(), volatile=True))

        ## 
        for i in range(len(sources)):
            t1x1 = self.conv1x1[i](sources[i])
            t1x1 *= weight_volatile[i][:,0,:,:].unsqueeze(1).expand_as(t1x1)
            t3x3 = self.conv3x3[i](sources[i])
            t3x3 *= weight_volatile[i][:,1,:,:].unsqueeze(1).expand_as(t3x3)
            t5x5 = self.conv5x5[i](sources[i])
            t5x5 *= weight_volatile[i][:,2,:,:].unsqueeze(1).expand_as(t5x5)
            sources_2.append(F.relu(t1x1+t3x3+t5x5, inplace=True))
            

        # apply multibox head to source layers
        for (x, l, c) in zip(sources_2, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        weight = torch.cat([o.view(o.size(0), -1) for o in weight], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                weight.view(weight.size(0), -1, 3)
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                weight.view(weight.size(0), -1, 3)
            )
        return output

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


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    conv1x1 = []
    conv3x3 = []
    conv5x5 = []
    weight_layers = []
    vgg_source = [21, 28, -2] # conv4_3, conv5_3 and fc7
    for k, v in enumerate(vgg_source):
        conv1x1 += [nn.Conv2d(vgg[v].out_channels,
                          128, kernel_size=1, padding=0)]
        conv3x3 += [nn.Conv2d(vgg[v].out_channels,
                          128, kernel_size=3, padding=1)]
        conv5x5 += [nn.Conv2d(vgg[v].out_channels,
                          128, kernel_size=5, padding=2)]
        loc_layers += [nn.Conv2d(128,
                             cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers +=[nn.Conv2d(128,
                             cfg[k] * num_classes, kernel_size=3, padding=1)]
        weight_layers += [nn.Conv2d(vgg[v].out_channels,
                             3, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        conv1x1 += [nn.Conv2d(v.out_channels,
                          128, kernel_size=1, padding=0)]
        conv3x3 += [nn.Conv2d(v.out_channels,
                          128, kernel_size=3, padding=1)]
        conv5x5 += [nn.Conv2d(v.out_channels,
                          128, kernel_size=5, padding=2)]
        loc_layers += [nn.Conv2d(128,
                             cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128,
                             cfg[k] * num_classes, kernel_size=3, padding=1)]
        weight_layers += [nn.Conv2d(v.out_channels,
                             3, kernel_size=3, padding=1)]
    return vgg, extra_layers, (conv1x1, conv3x3, conv5x5), (loc_layers, conf_layers, weight_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    # '512': [6, 6, 6, 6, 6, 4, 4],
    '512': [3, 3, 3, 3],
    '320': [3, 3, 3, 3],
}


def build_net(phase, size=320, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 320 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return SINGLEARNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)
