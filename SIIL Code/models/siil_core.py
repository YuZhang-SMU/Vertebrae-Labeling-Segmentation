import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from timm.models.vision_transformer import Block, Attention

from loss_function.loss_M import contrast_loss
from loss_function.loss_L import myLocLoss
from models.cross_block import CrossBlock, MultiCrossBlock
from models.largest_component import LargestComponent

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv1d_block(in_ch, out_ch):
    return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                         nn.InstanceNorm3d(out_ch, affine=True),
                         nn.LeakyReLU(inplace=True))

def conv3d_block(in_ch, out_ch):
    return nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                         nn.InstanceNorm3d(out_ch, affine=True),
                         nn.LeakyReLU(inplace=True))

class double_conv3d(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block1 = conv3d_block(in_channel, out_channel)
        self.block2 = conv3d_block(out_channel, out_channel)

    def forward(self, x):
        return self.block2(self.block1(x))

class SIIL_core(nn.Module):

    def __init__(self, feature_dim, config):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.init_R()
        self.init_S()
        self.init_MILL()
        self.init_OCPL()

    def init_R(self):
        self.R_token_proj = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=(16, 16))
        self.R_image_proj = nn.Sequential(nn.LayerNorm(self.feature_dim),
                                          nn.Linear(self.feature_dim, 16 * 16),
                                          nn.LayerNorm(16 * 16))
        self.R1_Ms = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            Attention(self.feature_dim, num_heads=8, qkv_bias=True)
        )
        self.R2_Mc = MultiCrossBlock(dim=self.feature_dim, num_heads=8, mlp_ratio=4., qkv_bias=True)

    def init_S(self):
        self.S_token_proj = nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=(8, 16, 16))
        self.S_image_proj = nn.Sequential(nn.LayerNorm(self.feature_dim),
                                          nn.Linear(self.feature_dim, 8 * 16 * 16),
                                          nn.LayerNorm(8 * 16 * 16))
        self.S1_Adapool = nn.AdaptiveAvgPool3d((8, 16, 16))
        self.S1_Mc = nn.ModuleList(
            [CrossBlock(dim=self.feature_dim, num_heads=8, mlp_ratio=4., qkv_bias=True) for _ in range(2)])
        self.S1_Ms = nn.ModuleList(
            [Block(dim=self.feature_dim, num_heads=8, mlp_ratio=4., qkv_bias=True) for _ in range(4)])
        self.S2_Mc = nn.ModuleList(
            [CrossBlock(dim=self.feature_dim, num_heads=8, mlp_ratio=4., qkv_bias=True) for _ in range(2)])
        self.S2_Ms = nn.ModuleList(
            [Block(dim=self.feature_dim, num_heads=8, mlp_ratio=4., qkv_bias=True) for _ in range(4)])

    def init_MILL(self):
        self.hat_L_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(in_features=self.feature_dim, out_features=26)
        )
        self.expansion_corrosion = nn.Sequential(nn.MaxPool1d(kernel_size=2, stride=2),
                                                 nn.Upsample(scale_factor=2, mode='nearest'))
        self.Activate_hat_L = nn.Sigmoid()
        self._loc_loss = myLocLoss()

    def init_OCPL(self):
        self.gru = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.GRU(input_size=self.feature_dim, hidden_size=self.feature_dim, num_layers=4, batch_first=True,
                   bidirectional=True)
        )
        self.gru_norm_gelu = nn.Sequential(
            nn.LayerNorm(2 * self.feature_dim),
            nn.Linear(2 * self.feature_dim, self.feature_dim),
            nn.GELU()
        )
        self.para_q = self.config.para_q
        self.para_k = self.config.para_k
        self.para_gamma = self.config.para_gamma
        self.temperature = 0.07

        # -----------------------set memory-----------------------------
        for i in range(26):
            self.register_buffer("queue" + str(i), torch.randn(self.para_q, self.feature_dim))
            self.register_buffer("queue_ptr" + str(i), torch.zeros(1, dtype=torch.long))
            self.register_buffer("cluster" + str(i), torch.randn(self.para_k, self.feature_dim))

    def forward(self, fea_F, fea_C_bool=None, msk_loc_low=None, epoch=0, index=None, val=None, test=None):
        fea_R1 = self.forward_Extraction_R(fea_F)
        fea_R2 = (self.R1_Ms(fea_R1) + fea_R1).squeeze(0)

        fea_L, fea_hat_L = self.forward_MILL(fea_R2, fea_C_bool)

        fea_S1 = self.forward_Extraction_S(fea_F.clone(), fea_L)
        fea_S_R2 = fea_L @ fea_R2
        fea_S_R2 = fea_S_R2[self.present_class_pred]
        fea_S2 = fea_S1.clone()
        for S1_Mc_blk in self.S1_Mc:
            fea_S2 = S1_Mc_blk(fea_S2, fea_S_R2.unsqueeze(0))
        for S1_Ms_blk in self.S1_Ms:
            fea_S2 = S1_Ms_blk(fea_S2)

        fea_P, each_loss_M = self.forward_OCPL(fea_S2, epoch, index)
        fea_P = fea_P[self.present_class_pred, :, :]

        fea_RP = fea_L[self.present_class_pred].permute(1, 0) @ fea_P.view(self.present_number, -1)
        fea_RP = fea_RP.view(32, self.para_k, self.feature_dim)
        fea_R3 = self.R2_Mc(fea_R2.unsqueeze(0), fea_RP.permute(1, 0, 2).unsqueeze(1))
        fea_R3 = fea_R3.squeeze(0)
        fea_S_R3 = fea_L @ fea_R3
        fea_S_R3 = fea_S_R3[self.present_class_pred].unsqueeze(0)
        fea_S3 = fea_S2.clone()
        for S2_Mc_blk in self.S2_Mc:
            fea_S3 = S2_Mc_blk(fea_S3, fea_S_R3)
        for S2_Ms_blk in self.S2_Ms:
            fea_S3 = S2_Ms_blk(fea_S3)

        # ---------------------------To image-----------------------------------
        each_tilde_R = self.forward_Deconstruction_R(fea_R3)
        each_tilde_S = self.forward_Deconstruction_S(fea_S3, fea_L.to(torch.bool))
        print(fea_hat_L.shape, msk_loc_low.shape)
        # ---------------------------loss-----------------------------------
        each_loss_L = self._loc_loss(fea_hat_L, msk_loc_low)
        if not (test or val):
            self._dequeue_and_enqueue(self.embedding, self.present_class_pred)
        return each_loss_L, each_loss_M, each_tilde_R, each_tilde_S

    def forward_OCPL(self, fea_S2, epoch, index):
        embedding = self.gru(fea_S2)
        embedding = self.gru_norm_gelu(embedding[0]).squeeze(0)
        # ---------compute cluster---------
        if index == 0 and epoch > 0:
            for i in range(26):
                queue_i = getattr(self, "queue" + str(i))
                cluster_i = getattr(self, "cluster" + str(i))
                cluster = KMeans(n_clusters=self.para_k).fit(queue_i.cpu().numpy())
                cluster_i[:] = torch.from_numpy(cluster.cluster_centers_)

        # --------------compute contrastive_loss-------------------
        fea_M = torch.zeros(size=(26, self.para_q, self.feature_dim), dtype=torch.float32, device=my_device)
        fea_P = torch.zeros(size=(26, self.para_k, self.feature_dim), dtype=torch.float32, device=my_device)
        for k in range(26):
            fea_M[k, :, :] = getattr(self, "queue" + str(k))
            fea_P[k, :, :] = getattr(self, "cluster" + str(k))
        loss_M = contrast_loss(embedding, fea_M, self.present_class_pred, self.temperature)
        self.embedding = embedding
        return fea_P, loss_M

    def forward_MILL(self, fea_R2, cls_final_bool):
        fea_hat_L = self.hat_L_proj(fea_R2.unsqueeze(0)).permute(0, 2, 1).squeeze(0)
        fea_L = torch.zeros(size=(26, 32), dtype=torch.float32, device=my_device)
        fea_hat_LL = self.Activate_hat_L(fea_hat_L)
        fea_hat_LL = self.expansion_corrosion(fea_hat_LL.unsqueeze(0)).squeeze(0)
        start, end, max_length = LargestComponent((fea_hat_LL > 0.5))
        max_length[cls_final_bool==0] = 0
        self.start1 = start
        self.end1 = end
        self.max_length1 = max_length
        # expand
        start = start - 1
        start[start < 0] = 0
        end = end + 1
        end[end > 32] = 32
        fea_L[0, :] = 1
        for j in range(1, 26):
            if end[j] > 1 and 2 < max_length[j] < 15:
                fea_L[j, start[j]: end[j]] = 1
        return fea_L, fea_hat_L

    def forward_Extraction_R(self, fea_F):
        return self.R_token_proj(fea_F.permute(1, 0, 2, 3)).squeeze(-1).squeeze(-1).unsqueeze(0)

    def forward_Extraction_S(self, fea_F, fea_L):
        fea_S1 = torch.zeros(size=(26, self.feature_dim, 8, 16, 16), dtype=torch.float32, device=my_device)
        fea_S1[0] = self.S1_Adapool(fea_F)
        for j in range(1, 26):
            fea_S1_j = fea_F[:, fea_L[j].long(), :, :]
            fea_S1[j] = self.S1_Adapool(fea_S1_j)
        fea_S1 = fea_S1[torch.any(fea_L, dim=1)]
        fea_S1 = self.S_token_proj(fea_S1).squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(0)
        self.present_number = fea_S1.shape[1]
        self.present_class_pred = torch.any(fea_L != 0, dim=1)
        return fea_S1

    def forward_Deconstruction_R(self, fea_F):
        return self.R_image_proj(fea_F).reshape(shape=(1, 32, 16, 16))

    def forward_Deconstruction_S(self, fea_S3, fea_L):
        fea_S3 = self.S_image_proj(fea_S3)
        fea_S3 = fea_S3.reshape(shape=(self.present_number, 8, 16, 16))

        fea_tilde_S = torch.zeros(size=(26, 32, 16, 16), dtype=torch.float32, device=my_device)
        fea_tilde_S_present = torch.zeros(size=(self.present_number, 32, 16, 16), dtype=torch.float32, device=my_device)

        for j in range(self.present_number):
            loc_j = fea_L[self.present_class_pred][j, :]
            fea_S3_j = fea_S3[j, :, :, :].unsqueeze(0).unsqueeze(0)
            fea_S3_j = nn.AdaptiveAvgPool3d(((loc_j).sum(), 16, 16))(fea_S3_j)
            fea_tilde_S_present[j, loc_j, :, :] = fea_S3_j.squeeze(0)
        fea_tilde_S[self.present_class_pred] = fea_tilde_S_present
        return fea_tilde_S

    @torch.no_grad()
    def _dequeue_and_enqueue(self, embedding, present_class):
        embedding = embedding.squeeze(0)
        if torch.isnan(embedding).any():
            print(embedding)
        else:
            c = torch.arange(26)[present_class].numpy()
            for i in range(self.present_number):
                queue_i = getattr(self, "queue" + str(c[i]))
                queue_ptr_i = getattr(self, "queue_ptr" + str(c[i]))

                ptr = int(queue_ptr_i)
                queue_i[ptr:ptr + 1] = queue_i[ptr:ptr + 1] * self.para_gamma + embedding[i] * (1 - self.para_gamma)
                ptr = (ptr + 1) % self.para_q

                queue_ptr_i[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.features.parameters(),
                                    self.features_mom.parameters()):
            param_k.data = param_k.data * self.para_gamma + param_q.data * (1. - self.para_gamma)

        for param_q, param_k in zip(self.extra_F.parameters(),
                                    self.extra_F_mom.parameters()):
            param_k.data = param_k.data * self.para_gamma + param_q.data * (1. - self.para_gamma)