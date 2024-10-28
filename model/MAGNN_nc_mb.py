import time

import torch
import torch.nn as nn
import numpy as np

from model.base_MAGNN import MAGNN_ctr_ntype_specific
from RGCN.model import RelationalGraphConvModel


# support for mini-batched forward
# only support one layer for one ctr_ntype
class MAGNN_nc_mb_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_nc_mb_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.ctr_ntype_layer = MAGNN_ctr_ntype_specific(num_metapaths,
                                                        etypes_list,
                                                        in_dim,
                                                        num_heads,
                                                        attn_vec_dim,
                                                        rnn_type,
                                                        r_vec,
                                                        attn_drop,
                                                        use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        # ctr_ntype-specific layers
        h = self.ctr_ntype_layer(inputs)

        #对应文中（7）式
        h_fc = self.fc(h)
        return h_fc, h


class MAGNN_nc_mb(nn.Module):
    def __init__(self,
                 dataset,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 num_rel,
                 adj_full,
                 A,
                 X,
                 RGCN_argument,
                 feats_opt,
                 gamma,
                 RGCN_layers,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(MAGNN_nc_mb, self).__init__()
        self.hidden_dim = hidden_dim
        self.A=A
        self.X=X
        self.feats_opt = feats_opt

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        self.RGCN = RelationalGraphConvModel(
            trans_size=RGCN_argument[0],
            hidden_size=RGCN_argument[1],
            output_size=RGCN_argument[2],
            feat_size_list=feats_dim_list,
            num_bases=RGCN_argument[3],
            num_rel=num_rel,
            num_layer=RGCN_layers,
            dropout=RGCN_argument[4],
            adj_full=adj_full,
            dataset = dataset,
            gamma = gamma,
            featureless=False,
            cuda=RGCN_argument[5],
        )
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_nc_mb layers
        self.layer1 = MAGNN_nc_mb_layer(num_metapaths,
                                        num_edge_type,
                                        etypes_list,
                                        hidden_dim,
                                        out_dim,
                                        num_heads,
                                        attn_vec_dim,
                                        rnn_type,
                                        attn_drop=dropout_rate)

    def forward(self, inputs):
        g_list, type_mask, edge_metapath_indices_list, target_idx_list = inputs

        recon_feat_list, emb_loss, recon_loss = self.RGCN(A=self.A, X=self.X)

        # print('new module:',t2-t1)
        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=recon_feat_list[0].device)
        for i,opt in enumerate(self.feats_opt):
            node_indices = np.where(type_mask == i)[0]
            if opt==1:
                transformed_features[node_indices] = self.fc_list[i](recon_feat_list[i])
            else:
                transformed_features[node_indices] = self.fc_list[i](self.X[i].to_dense())

        #TODO 为什么要dropout feature？
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        logits, h = self.layer1((g_list, transformed_features, type_mask, edge_metapath_indices_list, target_idx_list))

        return logits, h, emb_loss,recon_loss,recon_feat_list
