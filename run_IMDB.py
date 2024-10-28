import random
import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_IMDB_data
from utils.tools import  evaluate_results_nc2
from utils.utils import row_normalize
from model.MAGNN_nc_ac import MAGNN_nc_ac
import dgl

# Params
out_dim = 3
dropout_rate = 0.7
lr = 0.003
weight_decay = 0.001
etypes_lists = [[[0, 1], [2, 3]],
                [[1, 0], [1, 2, 3, 0]],
                [[3, 2], [3, 0, 1, 2]]]

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def run_model_IMDB( dataset,num_layers,hidden_dim, num_heads, attn_vec_dim, rnn_type,
                   num_epochs, patience, repeat, save_postfix,args):
    #adjlists为三个metapath的邻接表组成的list edge_metapath_indices_list为三个metapath的邻接表对应的metapath features_list为A、P、T、C四种节点对应的特征，A来自HAN统计结果，P是本文自己统计的，C用的是one-hot，T用的是glove向量化后的结果
    nx_G_lists, edge_metapath_indices_lists,adjM, type_mask, labels, train_val_test_idx,A,X,adj_full = load_IMDB_data()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
    num_nodes = A[0].shape[0]
    num_rel = len(A)
    A = row_normalize(A)
    in_dim_list = [X[i].shape[1] for i in range(len(X))]
    RGCN_argument = [args.trans_dim, args.hidden, args.emb_dim, args.bases, args.drop, args.using_cuda]
    X = [X[i].to(device) for i in range(len(X))]
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    feats_opt = args.feats_opt
    feats_opt = list(feats_opt)
    feats_opt = list(map(int, feats_opt))
    edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                                   edge_metapath_indices_lists]
    g_lists = []
    for nx_G_list in nx_G_lists:
        g_lists.append([])
        for nx_G in nx_G_list:
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())
            g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
            g_lists[-1].append(g)


    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []

    for _ in range(repeat):
        net = MAGNN_nc_ac(dataset,num_layers, [2, 2, 2], 4, etypes_lists, in_dim_list, hidden_dim, out_dim, num_heads, attn_vec_dim,num_rel,adj_full,A,X,RGCN_argument,feats_opt,
                       rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        target_node_indices = np.where(type_mask == 0)[0],

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        T=0
        for epoch in range(num_epochs):
            t0 = time.time()

            # training forward
            net.train()
            logits, embeddings,emb_loss,recon_loss,_ = net((g_lists, type_mask, edge_metapath_indices_lists), target_node_indices)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])+args.alpha*emb_loss+args.beta*recon_loss

            t1 = time.time()
            dur1.append(t1 - t0)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t2 = time.time()
            dur2.append(t2 - t1)
            T+=(t1-t0)+(t2-t1)
            # validation forward
            net.eval()
            with torch.no_grad():
                logits, embeddings,emb_loss,recon_loss,_ = net((g_lists, type_mask, edge_metapath_indices_lists), target_node_indices)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])+args.alpha*emb_loss+args.beta*recon_loss

            t3 = time.time()
            dur3.append(t3 - t2)

            # print info
            print(
                "Epoch {:05d} | Train_Loss {:.4f} | Val_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}".format(
                    epoch, train_loss.item(), val_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break


        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        with torch.no_grad():
            logits, embeddings,_,_,_ = net((g_lists, type_mask, edge_metapath_indices_lists), target_node_indices)
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std,  = evaluate_results_nc2(
                embeddings[test_idx].cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)
        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        nmi_mean_list.append(nmi_mean)
        nmi_std_list.append(nmi_std)
        ari_mean_list.append(ari_mean)
        ari_std_list.append(ari_std)


    # print out 结果.txt summary of the evaluations
    svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
    svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
    nmi_mean_list = np.array(nmi_mean_list)
    nmi_std_list = np.array(nmi_std_list)
    ari_mean_list = np.array(ari_mean_list)
    ari_std_list = np.array(ari_std_list)

    print('总时间：{}s'.format(T))
    print('----------------------------------------------------------------')
    print('SVM tests summary')
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05,0.01])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2, 0.1,0.05,0.01])]))
    print('K-means tests summary')
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean_list.mean(), nmi_std_list.mean()))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean_list.mean(), ari_std_list.mean()))



if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the IMDB dataset')
    ap.add_argument('--dataset',type=str,default = 'IMDB')
    ap.add_argument('--layers', type=int, default=2, help='Number of layers. Default is 2.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.')
    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='IMDB', help='Postfix for the saved model and result. Default is IMDB.')
    ap.add_argument("--drop", type=float, default=0.3, help="Dropout of RGCN")
    ap.add_argument("--hidden", type=int, default=64, help="Number of hidden units.")
    ap.add_argument("--bases", type=int, default=0, help="R-GCN bases")
    # ap.add_argument("--data", type=str, default="DBLP", help="dataset.")
    ap.add_argument("--emb_dim", type=int, default=64, help="dim of the embedding")
    ap.add_argument("--trans_dim", type=int, default=64, help="dim of the embedding")
    ap.add_argument("--alpha", type=float, default=2.0, help="ratio of emb_loss")
    ap.add_argument("--beta", type=float, default=0.9, help="ratio of recon_loss")
    ap.add_argument(
        "--no_cuda", action="store_true", default=False, help="Enables CUDA training."
    )
    ap.add_argument("--feats_opt", type=str, default='011', help='0100 means 1 type nodes use our processed feature')
    args = ap.parse_args()
    args.using_cuda = not args.no_cuda and torch.cuda.is_available()
    run_model_IMDB( args.dataset,args.layers, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
                   args.epoch, args.patience, args.repeat, args.save_postfix,args)
