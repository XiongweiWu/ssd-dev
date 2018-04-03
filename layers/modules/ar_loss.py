import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp, match2
GPU = False
if torch.cuda.is_available():
    GPU = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class ARLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """


    def __init__(self, num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target):
        super(ARLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]

    def forward(self, predictions, priors, targets, pass_index=None):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
                pre_conf is used to use Early reject or not

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, weights = predictions
        erf = priors[:,-1].data
        priors = priors[:,:-1]
        num = loc_data.size(0)
        num_priors = (priors.size(0)) if priors.dim() == 2 else priors.size(1)
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        weight_t = torch.Tensor(num, num_priors, 4)
        
        for idx in range(num):
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            if priors.dim() == 3: 
                defaults = priors.data[idx,:,:]
            else:
                defaults = priors.data
            # if pass_index is not None:
            #     defaults = defaults[pass_index_data[idx].unsqueeze(1).expand_as(defaults)].view(-1, num)

            # if defaults.shape[0] != 6375:
            #     print('ERROR')
            match2(self.threshold,truths,defaults,erf,self.variance,labels,loc_t,conf_t,weight_t,idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            weight_t = weight_t.cuda()
        # wrap targets
        if pass_index is not None:
            pass_index_data = pass_index.data
            loc_t = loc_t[pass_index_data.unsqueeze(2).expand_as(loc_t)].view(-1, 4)
            conf_t1 = conf_t[pass_index_data]
            loc_data = loc_data[pass_index_data.unsqueeze(2).expand_as(loc_data)].view(-1, 4)
            print(conf_t1.shape[0]/num)
            # conf_data1 = conf_data[pass_index_data.unsqueeze(2).expand_as(conf_data)].view(-1, self.num_classes)

        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)
        # weight_t = Variable(weight_t, requires_grad=False)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        if pass_index is not None:
            conf_t1 = Variable(conf_t1,requires_grad=False)
            pos = conf_t1 > 0
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1,4)
            loc_t = loc_t[pos_idx].view(-1,4)
            loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        else:
            pos = conf_t > 0
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1,4)
            loc_t = loc_t[pos_idx].view(-1,4)
            loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        pos = conf_t > 0

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1,self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # ER
        if pass_index is None:
            x_max = batch_conf.data.max()
            temp = torch.exp(batch_conf[:,0]-x_max) / torch.sum(torch.exp(batch_conf-x_max),1)
            # print(temp.data.max())
            temp = temp < 0.99
            temp_idx = temp.view(num, -1)

        # Hard Negative Mining
        loss_c[pos.view(-1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        if pass_index is not None:
            loss_c[1-pass_index_data] = 0
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Weight target
        pos_idx_2 = pos.view(pos.size(0),pos.size(1)//3,3).float().sum(2).unsqueeze(2).expand_as(weights)
        neg_idx_2 = neg.view(neg.size(0),neg.size(1)//3,3).float().sum(2).unsqueeze(2).expand_as(weights)
        w_temp = weight_t.view(weight_t.size(0),weight_t.size(1)//3,3,weight_t.size(2))
        print('1: {}'.format((weight_t[:,:,-1]==1).sum()))
        print('2: {}'.format((weight_t[:,:,-1]==2).sum()))
        print('3: {}'.format((weight_t[:,:,-1]==3).sum()))
        temp_ = (w_temp[:,:,:,-1]>0).float().sum(2)
        w_t = w_temp[:,:,:,:-1].sum(2)/torch.max(temp_,torch.ones(temp_.size())).unsqueeze(2).expand_as(weights)
        w_t = Variable(w_t, requires_grad=False)

        weights_ = weights[(pos_idx_2+neg_idx_2).gt(0)].view(-1,3)
        w_t_ = w_t[(pos_idx_2+neg_idx_2).gt(0)].view(-1,3)
        # loss_w = F.mse_loss(weights_, w_t_, size_average=False)
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        # another loss weight loss

        N = num_pos.data.sum()
        loss_l/=N
        loss_c/=N
        loss_w = 1
        # loss_w/=N

        if pass_index is None:
            index = (pos + temp_idx).gt(0)
        else:
            index = None
        return loss_l, loss_c, loss_w, index