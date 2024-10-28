from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import xlwt

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN
from tools import evaluate_results_nc2

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.002,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--metapath',type=str,default='MAM',
                    help = 'type of metapath')
parser.add_argument('--dataset',type=str,default='IMDB',
                    help='dataset')
parser.add_argument('--repeat',type=int,default=1,help='repeat of experiment')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train(epoch):
    t = time.time()
    #声明在train步骤
    model.train()
    #梯度清零
    optimizer.zero_grad()
    logits,h = model(features, adj)
    loss_train = F.nll_loss(logits[idx_train], labels[idx_train])
    acc_train = accuracy(logits[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    t2=time.time()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        logits,h = model(features, adj)

    loss_val = F.nll_loss(logits[idx_val], labels[idx_val])
    acc_val = accuracy(logits[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    print('T={}'.format(t2-t))


def test():
    model.eval()
    _,h = model(features, adj)
    test_embeddings = h[idx_test]
    svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std, \
    purity_mean, purity_std, ri_mean, ri_std, f_measure_mean, f_measure_std = evaluate_results_nc2(
        test_embeddings.detach().cpu().numpy(), labels[idx_test].detach().cpu().numpy(), num_classes=labels.max().cpu())
    svm_macro_f1_lists.append(svm_macro_f1_list)
    svm_micro_f1_lists.append(svm_micro_f1_list)
    nmi_mean_list.append(nmi_mean)
    # nmi_std_list.append(nmi_std)
    ari_mean_list.append(ari_mean)
    # ari_std_list.append(ari_std)
    purity_mean_list.append(purity_mean)
    # purity_std_list.append(purity_std)
    ri_mean_list.append(ri_mean)
    # ri_std_list.append(ri_std)
    f_measure_mean_list.append(f_measure_mean)

# Load data
svm_macro_f1_lists = []
svm_micro_f1_lists = []
nmi_mean_list = []
nmi_std_list = []
ari_mean_list = []
ari_std_list = []
purity_mean_list = []
purity_std_list = []
ri_mean_list = []
ri_std_list = []
f_measure_mean_list = []
f_measure_std_list = []
embedding_list = []
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.metapath,args.dataset)
for i in range(args.repeat):
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()

svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
nmi_mean_list = np.array(nmi_mean_list)
# nmi_std_list = np.array(nmi_std_list)
ari_mean_list = np.array(ari_mean_list)
# ari_std_list = np.array(ari_std_list)
purity_mean_list = np.array(purity_mean_list)
# purity_std_list = np.array(purity_std_list)
ri_mean_list = np.array(ri_mean_list)
# ri_std_list = np.array(ri_std_list)
f_measure_mean_list = np.array(f_measure_mean_list)
# nmi_mean_list = np.array(nmi_mean_list)
# nmi_std_list = np.array(nmi_std_list)
# ari_mean_list = np.array(ari_mean_list)
# ari_std_list = np.array(ari_std_list)

print('----------------------------------------------------------------')
print('SVM tests summary')
print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
    macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
    zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
    micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in

    zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
print('NMI: {:.6f}~{:.6f}'.format(nmi_mean_list.mean(), nmi_mean_list.std()))
print('ARI: {:.6f}~{:.6f}'.format(ari_mean_list.mean(), ari_mean_list.std()))
print('purity:{:.6f}~{:.6f}'.format(purity_mean_list.mean(), purity_mean_list.std()))
print('ri:{:.6f}~{:.6f}'.format(ri_mean_list.mean(), ri_mean_list.std()))
print('f_measure:{:.6f}~{:.6f}'.format(f_measure_mean_list.mean(), f_measure_mean_list.std()))








    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))


# Train model
