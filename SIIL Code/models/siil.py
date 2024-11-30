import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_function.loss_T import Dice_and_CrossEntropy_loss, Dice_and_CrossEntropy_loss_binary
from loss_function.loss_C import FocalLoss
from models.siil_decoder import Decoder
from models.siil_core import SIIL_core
from models.siil_encoder import generate_model
from utilize import to_one_hot_tensor

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SIIL(nn.Module):

    def __init__(self, config=None, feature_dim=256):
        super().__init__()
        self.config = config
        self.frozen_epoch = 10

        self.EN, en_ch = generate_model(model_depth=10, n_classes=26)
        self.SIIL_core = SIIL_core(feature_dim, self.config)
        self.DE = Decoder(en_ch)

        self.initialize_weights()
        self.up = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self._cls_loss = FocalLoss()
        self._dc_ce_loss_binary = Dice_and_CrossEntropy_loss_binary()
        self._dc_ce_loss = Dice_and_CrossEntropy_loss()

    def initialize_weights(self):
        self.apply(self._init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.SIIL_core.R_token_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.SIIL_core.S_token_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, data, epoch=0, index=None, val=False, test=False):
        img = data['image']['data'].permute(0, 1, 4, 2, 3).to(torch.float32).to(my_device)
        msk = data['label']['data'].permute(0, 1, 4, 2, 3).to(torch.float32).to(my_device)

        batch_size = msk.shape[0]

        msk_binary = msk.clone()
        msk_binary[msk_binary > 0] = 1
        msk_binary_low = F.max_pool3d(msk_binary, (4, 4, 4))

        msk_onehot = msk.clone()
        msk_onehot = to_one_hot_tensor(msk_onehot, range(26))
        msk_onehot_low = F.max_pool3d(msk_onehot, (4, 4, 4))

        msk_loc = (msk_onehot.sum(dim=[3, 4]) > 0).to(torch.float32).to(my_device)
        msk_loc_low = (msk_onehot_low.sum(dim=[3, 4]) > 0).to(torch.float32).to(my_device)

        msk_id = torch.zeros((batch_size, 26), device=my_device)
        idx = torch.unique(msk.contiguous().view(batch_size, -1), dim=1).to(torch.long)
        for b in range(msk.shape[0]):
            msk_id[b][idx[b]] = 1

        # ---------------------------encoder----------------------------------------
        fea_F1, fea_F2, fea_F3, fea_F, fea_C = self.EN(img)
        fea_C_bool = torch.sigmoid(fea_C) >= 0.5

        # ---------------------------core----------------------------------------
        loss_L = torch.zeros(1, dtype=torch.float32).to(my_device)
        loss_M = torch.zeros(1, dtype=torch.float32).to(my_device)
        fea_tilde_R = list()
        fea_tilde_S = list()
        batch_work = torch.ones(batch_size)
        for i in range(batch_size):
            if test and (not torch.any(fea_C_bool[i])):
                batch_work[i] = 0
                continue
            output = self.SIIL_core(fea_F[i], fea_C_bool[i], msk_loc_low[i], epoch=epoch, index=index, val=val, test=test)
            each_loss_L, each_loss_M, each_tilde_R, each_tilde_S = output
            loss_L += (each_loss_L / batch_size)
            loss_M += (each_loss_M / batch_size)
            fea_tilde_R.append(each_tilde_R.unsqueeze(0))
            fea_tilde_S.append(each_tilde_S.unsqueeze(0))

        if test and (not torch.any(batch_work)):
            return {
                'seg_one_final': torch.zeros_like(msk_binary),
                'seg_multi_final': torch.zeros_like(msk_onehot),
            }

        fea_tilde_R = torch.cat(fea_tilde_R, dim=0)
        fea_tilde_S = torch.cat(fea_tilde_S, dim=0)

        # ---------------------------decoder----------------------------------------
        fea_O_R, fea_O_S = self.DE(
            fea_tilde_R,
            fea_tilde_S,
            fea_F1[batch_work!=0],
            fea_F2[batch_work!=0],
            fea_F[batch_work!=0]
        )

        if test:
            seg_one_final = torch.zeros_like(msk_binary)
            seg_one_final[batch_work!=0] = fea_O_R

            seg_multi_final = torch.zeros_like(msk_onehot)
            seg_multi_final[batch_work != 0] = fea_O_S

            return {
                'seg_one_final': seg_one_final,
                'seg_multi_final': seg_multi_final,
            }

        # loss
        seg_loss1 = self._dc_ce_loss_binary(fea_O_R, msk_binary)
        seg_loss2 = self._dc_ce_loss(fea_O_S, msk_onehot, msk_loc)
        pseudo_loss1 = self._dc_ce_loss(fea_tilde_S, msk_onehot_low, msk_loc_low)
        pseudo_loss2 = self._dc_ce_loss_binary(fea_tilde_R, msk_binary_low)

        loss_T = seg_loss1 + seg_loss2 + pseudo_loss1 + pseudo_loss2
        loss_C = self._cls_loss(fea_C[:, 1:], msk_id[:, 1:])

        final_loss = loss_T + loss_L + 20 * loss_C + 0.05 * loss_M

        loss_dict = {
            'final_loss': final_loss,
            'loss_T': loss_T,
            'loss_L': loss_L,
            'loss_C': 20 * loss_C,
            'loss_M': 0.05 * loss_M,
        }
        print(
            'final_loss: {:.5f}'.format(final_loss.item()), '||',
            'loss_T: {:.5f}'.format(loss_T.item()), '||',
            'loss_L: {:.5f}'.format(loss_L.item()), '||',
            'loss_C: {:.5f}'.format(20 * loss_C.item()), '||',
            'loss_M: {:.5f}'.format(0.05 * loss_M.item()),
        )
        return loss_dict


