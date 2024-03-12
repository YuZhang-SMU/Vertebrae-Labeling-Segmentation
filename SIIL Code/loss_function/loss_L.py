import torch
import torch.nn as nn
import torch.nn.functional as F

class myLocLoss(nn.Module):
    def __init__(self, weight_ce=1, weight_iou=1, weight_kl=10):
        super(myLocLoss, self).__init__()
        self._ce_loss = myBCELoss()
        self._iou_loss = myIoULoss()
        self._kl_loss = myKLDivergence()
        self.weight_ce = weight_ce
        self.weight_iou = weight_iou
        self.weight_kl = weight_kl

    def forward(self, inputs, targets):
        inputs = inputs[1:]
        targets = targets[1:]
        present_class = torch.any(targets != 0, dim=1)
        present_space = torch.any(targets != 0, dim=0)
        ce_loss = self._ce_loss(inputs, targets, present_class, present_space)
        iou_loss = self._iou_loss(inputs, targets, present_class, present_space)
        if targets.shape[0] != 26:
            return self.weight_ce * ce_loss + self.weight_iou * iou_loss
        else:
            kl_loss = self._kl_loss(inputs, targets, present_class, present_space)
            return self.weight_ce * ce_loss + self.weight_iou * iou_loss + self.weight_kl * kl_loss

class myBCELoss(nn.Module):
    def __init__(self):
        super(myBCELoss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets, present_class, present_space):
        ce_loss = self.BCELoss(inputs, targets)
        weight_ce = torch.ones_like(ce_loss, device=ce_loss.device)
        weight_ce[present_class, :] += 2
        return (ce_loss * weight_ce).sum() / weight_ce.sum()

class myIoULoss(nn.Module):
    def __init__(self):
        super(myIoULoss, self).__init__()

    def forward(self, inputs, targets, present_class, present_space, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum(dim=1)
        total = (inputs + targets).sum(dim=1)
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU[present_class].mean()

class myKLDivergence(nn.Module):
    def __init__(self):
        super(myKLDivergence, self).__init__()
        self.activate_fn = nn.Softmax(dim=-1)
        self.KLDivergence = torch.nn.KLDivLoss(reduction='mean')

    def forward(self, inputs, targets, present_class, present_space):
        inputs = F.sigmoid(inputs)

        inputs = inputs[present_class, :]
        inputs = inputs[:, present_space]
        targets = targets[present_class, :]
        targets = targets[:, present_space]
        inputs = self.activate_fn(inputs.permute(1, 0).sum(dim=1))
        targets = self.activate_fn(targets.permute(1, 0).sum(dim=1))
        inputs = inputs.clamp(min=1e-4)
        return self.KLDivergence(inputs.log(), targets)

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU

