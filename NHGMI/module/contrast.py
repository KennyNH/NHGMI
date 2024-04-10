import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Beta
import random


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def cos_sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        pos = pos.to_dense()
        # neg = torch.ones_like(pos).cuda() - pos
        # pos = torch.eye(pos.size()[0]).cuda()
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        matrix_mp2mp = self.sim(z_proj_mp, z_proj_mp)
        matrix_sc2sc = self.sim(z_proj_sc, z_proj_sc)

        # sim_mat = self.cos_sim(z_mp, z_mp).detach().cpu().numpy() # similarity matrix
        # num_pos = int(pos[0].sum())
        # indices = [[np.argsort(sim_mat[i])[:50 * num_pos], np.argsort(sim_mat[i])[-num_pos:]]
        #                                                      for i in range(sim_mat.shape[0])]
        # neg = torch.zeros(pos.shape).cuda()
        # pos = torch.zeros(pos.shape).cuda()
        # for i, inds in enumerate(indices):
        #     neg[i][inds[0]] = 1.
        #     pos[i][inds[1]] = 1.
        
        # matrix_mp2sc = (matrix_mp2sc)/(torch.sum(matrix_mp2sc.mul(neg), dim=1).view(-1, 1) + 1e-8)
        # lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()

        # matrix_sc2mp = (matrix_sc2mp) / (torch.sum(matrix_sc2mp.mul(neg), dim=1).view(-1, 1) + 1e-8)
        # lori_sc = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()

        matrix_mp2sc = (matrix_mp2sc) / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()

        matrix_sc2mp = (matrix_sc2mp) / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()

        return self.lam * lori_mp + (1 - self.lam) * lori_sc


class MultiContrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam, sim_mat, gamma, rank_ratio, num_pos):
        super(MultiContrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        self.gamma = gamma
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        
        self.sim_mat = sim_mat
        np.fill_diagonal(self.sim_mat, 1.1) # make diagonal max
        self.sim_mat = torch.from_numpy(self.sim_mat)
        self.num = sim_mat.shape[0] # number of anchor nodes
        self.num_neg = 2 * self.num 
        self.num_pos = num_pos
        self.num_real_neg = int(self.num // rank_ratio) # number of sorted worst-similarity nodes
        self.neg_mask_init()
        self.pos_mask_init()
        # self.neg = self.neg_mask_random_add() # use neg mask to control neg samples


    def neg_mask_init(self):
        # neg_indices = [np.vstack((np.full((self.num_real_neg), i), 
        #                             np.argsort(sim_one)[:self.num_real_neg])) 
        #                                 for i, sim_one in enumerate(self.sim_mat)]
        # indices = torch.from_numpy(np.hstack(neg_indices))
        # values = torch.ones((indices.shape[1]))
        # shape = torch.Size((self.num, self.num))
        # neg = torch.sparse.FloatTensor(indices, values, shape).to_dense()
        # self.neg = torch.cat((neg, neg), 1).cuda() # initial neg mask

        # # negative col indices for each anchor node
        # self.neg_indices = [np.hstack((np.argsort(sim_one)[:self.num_real_neg], 
        #     self.num + np.argsort(sim_one)[:self.num_real_neg])) for sim_one in self.sim_mat]
        neg = torch.where(self.sim_mat > 0, 0., 1.).cuda()
        # self.neg = torch.cat((neg, neg), 1) # initial neg mask
        self.neg = neg

    
    def pos_mask_init(self):
        pos_indices = [np.vstack((np.full((self.num_pos), i), 
                                    np.argsort(-sim_one)[:self.num_pos])) 
                                        for i, sim_one in enumerate(self.sim_mat)]
        indices = torch.from_numpy(np.hstack(pos_indices))
        values = torch.ones((indices.shape[1]))
        shape = torch.Size((self.num, self.num))
        pos = torch.sparse.FloatTensor(indices, values, shape).to_dense() # initial neg mask
        self.pos = torch.where(self.sim_mat > 0, pos, self.sim_mat).cuda() # filter zero-sim positive samples


    def neg_mask_random_add(self):
        neg = self.neg.clone()
        num_add = self.num_neg - 2 * self.num_real_neg

        random_add = torch.randn(self.num, 2 * self.num).cuda()
        random_add = random_add * neg
        random_add[random_add == 0.] = -float('inf')
        random_add = torch.softmax(random_add, 1) * num_add

        neg += random_add

        return neg


    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix


    def cos_sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator)
        return sim_matrix
    

    def cal(self, z_loc, z_glo, pos):
        z_proj_loc = self.proj(z_loc)
        z_proj_glo = self.proj(z_glo)
        matrix_loc2glo = self.sim(z_proj_loc, z_proj_glo)
        matrix_glo2loc = matrix_loc2glo.t()
        matrix_loc2loc = self.sim(z_proj_loc, z_proj_loc)
        matrix_glo2glo = self.sim(z_proj_glo, z_proj_glo)

        # matrix_loc2neg = self.sim(z_proj_loc, torch.cat((z_proj_glo, z_proj_loc), 0))
        # matrix_glo2neg = self.sim(z_proj_glo, torch.cat((z_proj_loc, z_proj_glo), 0))

        # neg = self.neg_mask_random_add() # use neg mask to control neg samples
        neg = self.neg
        pos = self.pos
        # print(neg)
        # print(neg.shape, neg.sum())
        
        # neg = torch.ones_like(pos).cuda() - pos

        matrix_loc2loc_ = matrix_loc2loc - torch.diag_embed(torch.diag(matrix_loc2loc))
        matrix_glo2glo_ = matrix_glo2glo - torch.diag_embed(torch.diag(matrix_glo2glo))

        matrix_loc2glo = (matrix_loc2glo + matrix_loc2loc_) / (torch.sum(matrix_loc2glo.mul(neg), dim=1).view(-1, 1) 
                                                        + torch.sum(matrix_loc2loc.mul(neg), dim=1).view(-1, 1) + 1e-8)
        lori_loc = -torch.log(matrix_loc2glo.mul(pos).sum(dim=-1)).mean()

        matrix_glo2loc = (matrix_glo2loc + matrix_glo2glo_) / (torch.sum(matrix_glo2loc.mul(neg), dim=1).view(-1, 1) 
                                                        + torch.sum(matrix_glo2glo.mul(neg), dim=1).view(-1, 1) + 1e-8)
        lori_glo = -torch.log(matrix_glo2loc.mul(pos).sum(dim=-1)).mean()

        # matrix_loc2glo = (matrix_loc2glo) / (torch.sum(matrix_loc2glo, dim=1).view(-1, 1) + torch.sum(matrix_loc2loc, dim=1).view(-1, 1) + 1e-8)
        # lori_loc = -torch.log(matrix_loc2glo.mul(pos).sum(dim=-1)).mean()

        # matrix_glo2loc = (matrix_glo2loc) / (torch.sum(matrix_glo2loc, dim=1).view(-1, 1) + torch.sum(matrix_glo2glo, dim=1).view(-1, 1) + 1e-8)
        # lori_glo = -torch.log(matrix_glo2loc.mul(pos).sum(dim=-1)).mean()


        return self.lam * lori_loc + (1 - self.lam) * lori_glo


    def forward(self, z_mp, z_mp_locs, pos):
        pos = pos.to_dense()
        
        loss = 0.
        for z_mp_loc in z_mp_locs:
            loss += self.cal(z_mp_loc, z_mp, pos)
        
        return loss
        