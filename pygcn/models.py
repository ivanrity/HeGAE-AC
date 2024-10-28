import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self,  nhid,  dropout,adj_full,feat_size_list,gamma,num_layers):
        super(GCN, self).__init__()
        self.adj_full = adj_full
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        # self.dropout = dropout
        self.gamma = gamma
        self.num_layer = num_layers
        self.relu = nn.ReLU()
        self.dropout = dropout
        self.fc_list = \
            nn.ModuleList([nn.Linear(feat_dim, nhid, bias=True) for feat_dim in feat_size_list])
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(GraphConvolution(nhid, nhid))
        for i in range(num_layers):
            self.decoder_layers.append(GraphConvolution(nhid, nhid))
        self.fc_list2 = \
            nn.ModuleList([nn.Linear(nhid,feat_dim,  bias=True) for feat_dim in feat_size_list])
        # nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.dropout = dropout
    def emb_loss_fun(self,emb):
        BCE = torch.nn.BCELoss(reduction='sum')
        adj_full = self.adj_full
        adj_list = []
        emb_loss = []
        # indicator = [1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]
        indicator = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        type_list = [0, 4057, 18385, 26108, 26128]
        for i in range(len(type_list) - 1):
            for j in range(len(type_list) - 1):
                if indicator[4 * i + j] == 1:
                    adj = adj_full[type_list[i]:type_list[i + 1], type_list[j]:type_list[j + 1]]
                    adj = torch.from_numpy(adj.todense()).to(emb[0].device)
                    # adj = adj.tocoo()
                    # adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                    #                                torch.FloatTensor(adj.data.astype(np.float)))
                    # adj = adj.to_dense().to(emb[0].device)
                    prediction = torch.sigmoid(
                        torch.mm(emb[type_list[i]:type_list[i + 1]], emb[type_list[j]:type_list[j + 1]].T))
                    emb_loss.append(BCE(prediction, adj))

                else:
                    emb_loss.append( torch.pow(torch.sigmoid(
                        torch.mm(emb[type_list[i]:type_list[i + 1]], emb[type_list[j]:type_list[j + 1]].T)), 2).mean())
        # adj_full = torch.sparse.FloatTensor(
        #     torch.LongTensor([adj_full.row.tolist(), adj_full.col.tolist()]),
        #     torch.FloatTensor(adj_full.data.astype(np.float))
        # )
        emb_loss=sum(emb_loss)/(26128*26128)
        return emb_loss

    def emb_loss_fun2(self, emb):
        BCE = torch.nn.BCELoss()
        adj_full = torch.from_numpy(self.adj_full.todense()).to(emb[0].device)
        prediction = torch.sigmoid(torch.mm(emb,emb.T))
        emb_loss = BCE(prediction,adj_full)
        return emb_loss

    def recon_loss_fun(self,recon_feat,feat):
        loss_fun=torch.nn.MSELoss()
        loss=loss_fun(recon_feat,feat.to_dense())
        return loss

    def recon_loss_fun2(self,recon_feat,feat,):
        gamma = self.gamma
        recon_feat = F.normalize(recon_feat,p=2,dim=-1)
        feat = F.normalize(feat.to_dense(),p=2,dim=1)

        loss = (1-(recon_feat*feat).sum(dim=-1)).pow(gamma)

        loss = loss.mean()
        return loss

    def forward(self, A,X,dataset):
        trans_feat = []
        for i, fc in enumerate(self.fc_list):
            trans_feat.append(fc(X[i].to_dense()))
        x = torch.cat(trans_feat, dim=0)
        for i, layer in enumerate(self.encoder_layers):
            if i != 0:
                x = x + layer(x, A)
            else:
                x = layer(x, A)
                if i != self.num_layer - 1:
                    x = F.dropout(self.relu(x), self.dropout, training=self.training)
                else:
                    x = F.dropout(x, self.dropout, training=self.training)

        for i, layer in enumerate(self.decoder_layers):
            if i != 0:
                x = x + layer(x, A)
            else:
                x = layer(x, A)
                if i != self.num_layer - 1:
                    x = F.dropout(self.relu(x), self.dropout, training=self.training)
                else:
                    x = F.dropout(x, self.dropout, training=self.training)

        if dataset == 'DBLP':
            final_emb=[X[0:4057,:],X[4057:18385,:],X[18385:26108,:],X[26108:26128,:]]
        elif dataset == 'IMDB':
            final_emb = [x[0:4278,:],x[4278:6359,:],x[6359:11616,:]]
        elif dataset =='ACM':
            final_emb = [X[0:4019,:],X[4019:11186,:],X[11186:11246,:]]
        elif dataset =='Yelp':
            final_emb = [X[0:2614,:],X[2614:3900,:],X[3900:3904,:],X[3904:3913,:]]
        recon_feat_list=[]
        for i,fc in enumerate(self.fc_list2):
            recon_feat_list.append(fc(final_emb[i].to_dense()))

        if dataset =='DBLP':
            emb_loss = self.emb_loss_fun(x)
            recon_loss = self.recon_loss_fun2(recon_feat_list[1], X[1])
        else:
            emb_loss = self.emb_loss_fun2(x)
            recon_loss = self.recon_loss_fun2(recon_feat_list[0], X[0])

        return recon_feat_list,emb_loss,recon_loss
