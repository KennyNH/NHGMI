import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder
from .contrast import Contrast, MultiContrast


class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, sim_mat, gamma, rank_ratio, num_pos):
        super(HeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, attn_drop)
        self.sc = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)      
        # self.contrast = Contrast(hidden_dim, tau, lam)
        self.multicontrast = MultiContrast(hidden_dim, tau, lam, sim_mat, gamma, rank_ratio, num_pos)

    def forward(self, feats, pos, mps, nei_index):  # p a s
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        z_sc = self.sc(h_all, nei_index)
        z_mp, z_mp_locs = self.mp(h_all[0], mps, [])
        # loss = self.contrast(z_mp, z_sc, pos)
        loss = self.multicontrast(z_mp, z_mp_locs, pos) + self.multicontrast(z_mp, [z_sc], pos)
        return loss

    def get_embeds(self, feats, mps):
        # h_all = []
        # for i in range(len(feats)):
        #     h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        # z_sc = self.sc(h_all, nei_index)
        # z_mp, _ = self.mp(h_all[0], mps, z_sc)
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp, _ = self.mp(z_mp, mps, [])

        return z_mp.detach()
