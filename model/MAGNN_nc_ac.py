import time

import torch
import torch.nn as nn
import numpy as np

from model.base_MAGNN import MAGNN_ctr_ntype_specific
from RGCN.model import RelationalGraphConvModel
import torch.nn.functional as F

fc_switch = False


# multi-layer support
class MAGNN_nc_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_nc_layer, self).__init__()
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
        self.ctr_ntype_layers = nn.ModuleList()
        for i in range(len(num_metapaths_list)):
            self.ctr_ntype_layers.append(MAGNN_ctr_ntype_specific(num_metapaths_list[i],
                                                                  etypes_lists[i],
                                                                  in_dim,
                                                                  num_heads,
                                                                  attn_vec_dim,
                                                                  rnn_type,
                                                                  r_vec,
                                                                  attn_drop,
                                                                  use_minibatch=False))

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        if fc_switch:
            self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
            self.fc2 = nn.Linear(in_dim * num_heads, out_dim, bias=True)
            nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
            nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
        else:
            self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
            nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists = inputs

        # ctr_ntype-specific layers
        h = torch.zeros(type_mask.shape[0], self.in_dim * self.num_heads, device=features.device)
        for i, (g_list, edge_metapath_indices_list, ctr_ntype_layer) in enumerate(zip(g_lists, edge_metapath_indices_lists, self.ctr_ntype_layers)):
            h[np.where(type_mask == i)[0]] = ctr_ntype_layer((g_list, features, type_mask, edge_metapath_indices_list))

        if fc_switch:
            h_fc = self.fc1(features) + self.fc2(h)
        else:
            h_fc = self.fc(h)
        return h_fc, h


class MAGNN_nc_ac(nn.Module):
    def __init__(self,
                 dataset,
                 num_layers,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
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
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(MAGNN_nc_ac, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.A = A
        self.X = X
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
            num_layer=3,
            dropout=RGCN_argument[4],
            adj_full=adj_full,
            dataset = dataset,
            gamma=3.0,
            featureless=False,
            cuda=RGCN_argument[5],)
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_nc layers
        self.layers = nn.ModuleList()
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(MAGNN_nc_layer(num_metapaths_list, num_edge_type, etypes_lists, hidden_dim, hidden_dim,
                                              num_heads, attn_vec_dim, rnn_type, attn_drop=dropout_rate))
        # output projection layer
        self.layers.append(MAGNN_nc_layer(num_metapaths_list, num_edge_type, etypes_lists, hidden_dim, out_dim,
                                          num_heads, attn_vec_dim, rnn_type, attn_drop=dropout_rate))

    def forward(self, inputs, target_node_indices):
        g_lists,  type_mask, edge_metapath_indices_lists = inputs

        recon_feat_list, emb_loss, recon_loss = self.RGCN(A=self.A, X=self.X)

        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=recon_feat_list[0].device)
        for i,opt in enumerate(self.feats_opt):
            node_indices = np.where(type_mask == i)[0]
            if opt==1:
                transformed_features[node_indices] = self.fc_list[i](recon_feat_list[i])
            else:
                transformed_features[node_indices] = self.fc_list[i](self.X[i].to_dense())

        # ntype-specific transformation

        h = self.feat_drop(transformed_features)

        # hidden layers
        for l in range(self.num_layers - 1):
            h, _ = self.layers[l]((g_lists, h, type_mask, edge_metapath_indices_lists))
            h = F.elu(h)
        # output projection layer
        logits, h = self.layers[-1]((g_lists, h, type_mask, edge_metapath_indices_lists))

        # return only the target nodes' logits and embeddings
        return logits[target_node_indices], h[target_node_indices],emb_loss,recon_loss,recon_feat_list
