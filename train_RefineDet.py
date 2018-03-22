from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_320, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
import time

from utils import box_utils

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='320',
                    help='320 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
# parser.add_argument('--C_agnostic', default=False,
#                     type=bool, help='class_agnostic or not')
parser.add_argument('--C_agnostic', dest='C_agnostic', action='store_true')
parser.add_argument('--no-C_agnostic', dest='C_agnostic', action='store_false')
parser.add_argument('--log_dir', default='./logs/',
                    help='Location to save logs')
parser.add_argument('--extra', type=str,
                    help='specify extra infos to describe the network')
parser.add_argument('--bp_anchors', default=False,
                    type=bool, help='whether bp via refined anchors')
parser.set_defaults(C_agnostic=True)
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

log_file = os.path.join(args.log_dir, args.version+args.dataset+'_refine_agnostic_{}.log.{}'.format(args.C_agnostic, args.extra))
f_writer = open(log_file, 'w')

if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_320, VOC_512)[args.size == '512']
else:
    train_sets = [('2014', 'train'),('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB320HL_vgg':
    from models.RFB320HL_Net_vgg import build_net
elif args.version == 'HL_vgg':
    from models.HL_Net_vgg import build_net
else:
    print('Unkown version!')

img_dim = (320,512)[args.size=='512']
rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
p = (0.6,0.2)[args.version == 'RFB_mobile']
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9
C_agnostic = args.C_agnostic

net = build_net('train', img_dim, num_classes, C_agnostic)
print(net)
# f_writer.write(net)
# f_writer.write('\n')

if args.resume_net == None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    f_writer.write('Loading base network...\n')
    net.base.load_state_dict(base_weights)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    def weights_init2(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()

    print('Initializing weights...')
    f_writer.write('Initializing weights...\n')
# initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.loc_1.apply(weights_init)
    net.conf_1.apply(weights_init)
    net.loc_2.apply(weights_init)
    net.conf_2.apply(weights_init)
    net.extra_conv_layers.apply(weights_init)
    net.extra_smooth_layers.apply(weights_init)
    if args.version == 'RFB320HL_vgg':
        net.Norm4_3.apply(weights_init)
        net.Norm7_fc.apply(weights_init)

else:
# load resume network
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
#optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = [MultiBoxLoss(2 if C_agnostic else num_classes, 0.5, True, 0, True, 3, 0.5, False),
             MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)]
priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward(), volatile=True)

def decode(loc_data, priors, variances):
    p = priors.unsqueeze(0).expand_as(loc_data)
    boxes = torch.cat((
        p[:,:,:2] + loc_data[:, :, :2] * variances[0] * p[:, :, 2:],
        p[:,:,2:] * torch.exp(loc_data[:,:,2:] * variances[1])), 2)
    # for i in range(loc_data.size(0)):
    #     loc_data[i,:,:] = box_utils.decode(loc_data[i,:,:], priors, variance)
    #     loc_data[i,:,:].clamp_(0,1)
    return boxes

def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')
    f_writer.write('Loading Dataset...\n')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    print('Training',args.version, 'on', dataset.name)
    f_writer.write('Training'+args.version+ 'on'+ dataset.name+ '\n')
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr

    loss = [None] * 2
    loss_l = [None] * 2
    loss_c = [None] * 2

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            if (epoch % 40 == 0 and epoch > 0) or (epoch % 10 ==0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder+args.version+'_'+args.dataset + '_epoches_'+
                           repr(epoch) + '_refine_agnostic_{}.pth.{}'.format(C_agnostic, args.extra))
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)


        # load train data
        targets = [None] * 2
        images, targets[1] = next(batch_iterator)

        targets[0] = [None] * len(targets[1])
        if C_agnostic:
            for i in range(len(targets[1])):
                targets[0][i] = targets[1][i].clone()
                targets[0][i][:,4] = targets[0][i][:,4].ge(1)
        else:
            targets[0] = targets[1]
        
        #print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets[0] = [Variable(anno.cuda(),volatile=True) for anno in targets[0]]
            targets[1] = [Variable(anno.cuda(),volatile=True) for anno in targets[1]]
        else:
            images = Variable(images)
            targets[0] = [Variable(anno, volatile=True) for anno in targets[0]]
            targets[1] = [Variable(anno, volatile=True) for anno in targets[1]]
        # forward
        t0 = time.time()
        out = net(images)

        ### calculation refined anchors
        # loc_data = Variable(out[0][0].data.clone(), volatile=True)
        loc_data = out[0][0].data.clone()
        conf_data = Variable(out[0][1].data.clone(), volatile=True)
        ## decode and clamp
        r_priors = decode(loc_data, priors.data, cfg['variance'])
        if args.bp_anchors:
            r_priors = Variable(r_priors, requires_grad=True)
        else:
            r_priors = Variable(r_priors, volatile=True)

        # for i in range(loc_data.size(0)):
        #     z = box_utils.decode(loc_data.data[i,:,:], priors.data, cfg['variance'])
        #     # loc_data[i,:,:].clamp_(0,1)

        # backprop
        optimizer.zero_grad()

        loss_l[0], loss_c[0], pass_index = criterion[0](out[0], priors, targets[0])
        loss[0] = loss_l[0] + loss_c[0]
        
        loss_l[1], loss_c[1], _ = criterion[1](out[1], r_priors, targets[1], pass_index)
        loss[1] = loss_l[1] + loss_c[1]


        loss_total = loss[0] + loss[1]
        loss_total.backward()
        optimizer.step()
        t1 = time.time()
        # loc_loss += loss_l.data[0]
        # conf_loss += loss_c.data[0]
        load_t1 = time.time()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L1: %.4f C1: %.4f||' % (loss_l[0].data[0],loss_c[0].data[0]) + 
                  ' || L2: %.4f C2: %.4f||' % (loss_l[1].data[0],loss_c[1].data[0]) + 
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
            f_writer.write('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L1: %.4f C1: %.4f||' % (loss_l[0].data[0],loss_c[0].data[0]) + 
                  ' || L2: %.4f C2: %.4f||' % (loss_l[1].data[0],loss_c[1].data[0]) + 
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr) + '\n')

    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version +'_' + args.dataset+ '_refine_agnostic_{}.pth.{}'.format(C_agnostic, args.extra))

    f_writer.write('training finished!\n')
    f_writer.close()


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
