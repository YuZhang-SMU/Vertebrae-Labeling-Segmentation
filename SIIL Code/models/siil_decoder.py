import torch
import torch.nn as nn

from models.siil_core import double_conv3d

class Decoder(nn.Module):

    def __init__(self, en_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.deconv1 = double_conv3d(1 + 26 + en_ch[3], en_ch[3])
        self.deconv2 = double_conv3d(en_ch[3] + en_ch[1], en_ch[2])
        self.deconv3 = double_conv3d(en_ch[2] + en_ch[0], en_ch[1])
        self.seg_one_final = nn.Conv3d(en_ch[1], 1, kernel_size=1)

        self.deconv4 = double_conv3d(1 + 26 + en_ch[3], en_ch[3])
        self.deconv5 = double_conv3d(en_ch[3] + en_ch[1], en_ch[2])
        self.deconv6 = double_conv3d(en_ch[2] + en_ch[0], en_ch[1])
        self.seg_multi_final = nn.Conv3d(en_ch[1], 26, kernel_size=1)

    def forward(
            self,
            id_pred_low,
            seg_pred_low,
            feature_down1,
            feature_down2,
            pre_feature
        ):
        input = torch.cat([id_pred_low, seg_pred_low, pre_feature], dim=1)

        x = self.deconv1(input)
        x = torch.cat([self.up(x), feature_down2], dim=1)
        x = self.deconv2(x)
        x = torch.cat([self.up(x), feature_down1], dim=1)
        x = self.deconv3(x)
        seg_one_final = self.seg_one_final(x)

        x = self.deconv4(input)
        x = torch.cat([self.up(x), feature_down2], dim=1)
        x = self.deconv5(x)
        x = torch.cat([self.up(x), feature_down1], dim=1)
        x = self.deconv6(x)
        seg_multi_final = self.seg_multi_final(x)
        return seg_one_final, seg_multi_final