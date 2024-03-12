import torch
import torch.nn as nn

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dice_loss_binary(nn.Module):
    def __init__(self):
        super(Dice_loss_binary, self).__init__()
        self.nonlin_func = nn.Sigmoid()

    def forward(self, pred_flat, target_flat, smooth=1.):
        pred_flat = self.nonlin_func(pred_flat)
        intersection = (pred_flat * target_flat).sum(1)
        unionset = pred_flat.sum(1) + target_flat.sum(1)
        return 1 - (2 * intersection + smooth) / (unionset + smooth)


class Dice_and_CrossEntropy_loss_binary(nn.Module):
    def __init__(self, weight_ce=1, weight_dice=1):
        super(Dice_and_CrossEntropy_loss_binary, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = Dice_loss_binary()
        self.crossEntropy_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        N = target.size(0)
        pred_flat = pred.reshape(N, -1)
        target_flat = target.reshape(N, -1)

        dc_loss = self.dice_loss(pred_flat, target_flat)
        ce_loss = self.crossEntropy_loss(pred_flat, target_flat) / target_flat.size(1)
        result = self.weight_ce * ce_loss.sum() + self.weight_dice * dc_loss.sum()
        return result / (2 * N)


class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()
        self.nonlin_func = nn.Sigmoid()

    def forward(self, inputs, targets, smooth=1.):
        inputs = self.nonlin_func(inputs)
        intersection = (inputs * targets).sum()
        unionset = inputs.sum() + targets.sum()
        return 1 - (2 * intersection + smooth) / (unionset + smooth)


class Dice_and_CrossEntropy_loss(nn.Module):
    def __init__(self, weight_ce=2, weight_dice=1):
        super(Dice_and_CrossEntropy_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = Dice_loss()
        self.crossEntropy_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, loc_label, smooth=1.):
        N = targets.shape[0]
        C = targets.shape[1]
        ce_loss_all = 0
        dc_loss_all = 0
        k = 0
        for i in range(N):
            for j in range(1, C):
                loc = loc_label[i, j, :].to(torch.bool)
                if torch.any(loc):
                    input = inputs[i, j, :, :, :].view(-1)
                    target = targets[i, j, :, :, :].view(-1)
                    ce_loss = self.crossEntropy_loss(input, target)
                    dc_loss = self.dice_loss(input, target)
                    ce_loss_all += ce_loss
                    dc_loss_all += dc_loss
                    k += 1
        return (self.weight_ce * ce_loss_all + self.weight_dice * dc_loss_all) / (k + smooth)