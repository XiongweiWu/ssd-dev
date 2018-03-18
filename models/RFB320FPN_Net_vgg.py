import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                #BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        out = torch.cat((x0,x1),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out



class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        '''
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        '''
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class RFB320FPNNet(nn.Module):
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

    def __init__(self, phase, size, base, extras, extra_conv_layers, extra_smooth_layers, head, num_classes):
        super(RFB320FPNNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 320:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4
        self.Norm4_3 = BasicRFB_a(512,512,stride = 1,scale=1.0)
        self.Norm7_fc = BasicRFB(1024, 1024, stride=1, scale = 1.0, visual=2)
        self.extras = nn.ModuleList(extras)

        self.extra_conv_layers = nn.ModuleList(extra_conv_layers)
        self.extra_smooth_layers = nn.ModuleList(extra_smooth_layers)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
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
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        f0 = self.Norm4_3(x)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        f1 = self.Norm7_fc(x)

        # apply extra layers and cache source layer outputs
        f23 = []
        for k, v in enumerate(self.extras):
            x = v(x)
            f23.append(x)

        f2 = f23[0]
        f3 = f23[1]

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

        sources = [feat0, feat1, feat2, feat3]

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def _upsample_add(self, x, y):
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


def conv_smooth(size, vgg, extra_layers):
    extra_conv_layers = []
    extra_smooth_layers = []
    # should be modified if the structure changed
    # at least reverse way
    vgg_source = [21, -2]

    for k, v in enumerate(vgg_source):
        extra_conv_layers += [nn.Sequential(nn.Conv2d(vgg[v].out_channels, 256,
                                            kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 256,
                                            kernel_size=3, padding=1))]
        extra_smooth_layers += [nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))]
    for k, v in enumerate(extra_layers):
        if k == 1:
            extra_conv_layers += [nn.Sequential(nn.Conv2d(v.out_channels, 256,
                                                kernel_size=3, padding=1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(256, 256,
                                                kernel_size=3, padding=1)
                                                )]
        else:
            extra_conv_layers += [nn.Sequential(nn.Conv2d(v.out_channels, 256,
                                                kernel_size=3, padding=1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(256, 256,
                                                kernel_size=3, padding=1),
                                                nn.ReLU(inplace=True))]

        extra_smooth_layers += [nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))]

    return extra_conv_layers, extra_smooth_layers

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
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
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
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=1)]
                else:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=2)]
            else:
                layers += [BasicRFB(in_channels, v, scale = 1.0, visual=2)]
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
    '512': ['S', 512, 'S', 256],
    '320': ['S', 512, 'S', 256]
}


def multibox(size, vgg, extra_layers, extra_conv_layers, extra_smooth_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(256,
                             cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256,
                    cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers):
        loc_layers += [nn.Conv2d(256, cfg[k]
                             * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, cfg[k]
                              * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, extra_conv_layers, extra_smooth_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [3, 3, 3, 3],
    '320': [3, 3, 3, 3]
}


def build_net(phase, size=320, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 320 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    vgg_layers = vgg(base[str(size)], 3)
    extra_layers = add_extras(size, extras[str(size)], 1024)
    extras_conv_layers, extra_smooth_layers = conv_smooth(size, vgg_layers, extra_layers)

    return RFB320FPNNet(phase, size, *multibox(size, vgg_layers,
                                extra_layers, extras_conv_layers, extra_smooth_layers,
                                mbox[str(size)], num_classes), num_classes)
