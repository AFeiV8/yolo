"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

# 平滑处理：positive：0.95  negative：0.05
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative labels smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing labels effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing labels effects
        # dx = (pred - true).abs()  # reduce missing labels and false labels effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))   # 实例化分类损失计算器
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))   # 实例化置信度损失计算器

        # Class labels smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses:损失计算
        for i, pi in enumerate(p):  # pi.shape(bs,na,wf,hf,nc+5)
            b, a, gj, gi = indices[i]  # img_ID, anchor_ID, gridy, gridx   indices:正样本信息
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj  shape:(bs,na,wf,hf)

            n = b.shape[0]  # 正样本数量
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0

                # pi[b, a, gj, gi].shape=n*(nc+5)-----按列分割----->n*(xy,wh,confidence,nc)
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # 获取预测特征层上所有正样本的预测信息

                # 定位损失：
                pxy = pxy.sigmoid() * 2 - 0.5                 # 预测的中心点坐标偏移量
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]   # 预测的高宽信息
                pbox = torch.cat((pxy, pwh), 1)               # 预测的框信息,shape(n,4)
                iou = bbox_iou(pbox, tbox[i], EIOU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # 置信度损失：检测框中存在目标的概率
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # ，将iou设置为confidence真实框的得分

                # 分类损失：
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # n*nc大小的表格
                    '''逐行赋值:
                       将tcls[i]的每个值与range控制的行数相结合，形成置1坐标位置
                       按照坐标将其对应位置置1'''
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # 置信度损失针对不同预测特征层采用不同的权重balance(4.0,1.0,0.4)
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(img-ID,class,x,y,w,h)   x y w h在0~1区间
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # 初始化增益
        # 创建一个0~na的列向量，并将其复制nt份，得到一个na行nt列的表格
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # 在原本targets信息的末尾追加上anchors_ID,相当于把原始的targets信息复制na次后在末尾追加anchors_ID
        # ai.shape：(na,nt)--->(na,nt,1)            targets.shape：(nt,6)--->(na,nt,6) + (na,nt,1)--->(na,nt,7)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # (img_ID,class,x,y,w,h,anchor_ID)

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets：tensor*0.5

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            '''将所有的x,y,w,h的对应的位置上改为预测特征层的高宽尺度，拿到targets信息的增益'''
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

            # 将targets与anchors进行匹配
            t = targets * gain  # shape(na,nt,7) 将targets映射到预测特征层上，相对坐标转预测特征层上的绝对坐标
            if nt:
                # Matches：目标边界框的匹配
                r = t[..., 4:6] / anchors[:, None]  # 计算高宽比
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # 按照高宽比限制值匹配正样本，拿到布尔值
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                '''t(na,nt,7)  j(na,nt)'''
                t = t[j]  # 根据布尔值拿到所匹配到正样本anchors对应的targets  正样本数*(img_ID,class,x,y,w,h,anchor_ID)

                # 正采样的扩充
                # Offsets：偏移量
                gxy = t[:, 2:4]  # 中心点坐标
                '''拿到预测特征层的宽和高，等价于右下角点减去所有中心点坐标，相当于拿到以右下角点为原点的坐标系坐标。
                   这样做是为了处理小数部分>0.5的坐标，在原坐标系小数部分>0.5的数，在新坐标系下<0.5。便可筛选出
                   坐标值>1且小数部分<0.5的坐标。'''
                gxi = gain[[2, 3]] - gxy
                # 筛选出坐标值>1且小数部分<0.5，拿到布尔值
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # 坐标小数部分<0.5的取向左或下的网格作为正样本的扩充
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # 坐标小数部分>0.5的取向右或上的网格作为正样本的扩充
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]  # repeat:(通道重复次数，行重复次数，列重复次数) 拿到需要做扩充的样本的targets信息
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 拿到偏移量网格
            else:
                t = targets[0]
                offsets = 0

            # Define
            # 按列分割，分成四组
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors_ID,img,class
            gij = (gxy - offsets).long()   # 取整，拿到正样本左上角点的坐标
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image,anchor,grid clamp压缩'
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # gxy - gij：中心点坐标偏移量:中心点坐标-正样本左上角点坐标   shape:(正样本数，6)
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
