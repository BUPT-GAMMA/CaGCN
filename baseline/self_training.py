import random
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
import os
import sys

os_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os_path)

from utils import accuracy
from models_calibration import *
from utils import *
from util_calibration import _ECELoss
from util_calibration import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default="Cora",
                    help='dataset for training')
parser.add_argument('--k', type=int, default=500)
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--labelrate', type=int, default=20)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val, idx_test):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        loss_val = criterion(output[idx_val], labels[idx_val])
        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])

    print(f'epoch: {epoch}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train:.4f}',
          f'loss_val: {loss_val.item():.4f}',
          f'acc_val: {acc_val:.4f}',
          f'loss_test: {loss_test.item():4f}',
          f'acc_test: {acc_test:.4f}',
          f'time: {time.time() - t:.4f}s')
    return loss_val, acc_train, loss_test, acc_test, output


@torch.no_grad()
def test(adj, features, labels, idx_test, nclass, idx_train,
         idx_val, model_path):
    nfeat = features.shape[1]
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print(f"Test set results",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test:.4f}")

    return acc_test, loss_test


def generate_pesudo_label(output, idx_train, pseudo_labels,
                          idx_test, idx_val, nclass, topk_pc):
    train_index = torch.where(idx_train==True)
    test_index = torch.where(idx_test==True)
    val_index = torch.where(idx_val==True)
    confidence, pred_label = get_confidence(output)

    ######## self-training ########
    for j in range(0, nclass):
        index = torch.where(pred_label == j)[0]
        # the index of nodes of class j sorted in descending order according to their confidence
        index = index[torch.sort(confidence[index], descending=True)[1]]
        npseudo = 0
        for i in index:
            if i not in train_index[0] and i not in test_index[0] and i not in val_index[0] and npseudo < topk_pc:
                pseudo_labels[i] = pred_label[i]
                idx_train[i] = True
            npseudo += 1

    return idx_train, pseudo_labels

def main(dataset):
    data, adj, features, labels, idx_train, idx_val, idx_test, nxg = load_data(dataset, args.labelrate, os_path+'/')
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    nclass = labels.max().item() + 1

    pseudo_labels = labels.clone()
    pseudo_labels = pseudo_labels.to(device)
    acc_test_times_list = list()
    model_path = os_path + '/save_model/st-%s-%s-%d-w_o-s.pth' % (args.model, args.dataset, args.labelrate)

    # the value of topk (t in the original paper)
    ndata = labels.size()[0]
    avg_degree = nxg.number_of_edges() * 2 / ndata
    topk = max(ndata / (avg_degree ** 2) * args.k - torch.where(idx_train == True)[0].size()[0], 0)
    topk_pc = topk / nclass     #topk per class

    # Model and optimizer
    model = get_models(args, features.shape[1], nclass)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)

    # Train model
    best = 100
    bad_counter = 0
    for epoch in range(args.epochs):
        loss_val, acc_train, loss_test, acc_test, output = train(epoch, model, optimizer, adj, features,
                                                                 pseudo_labels, idx_train,
                                                                 idx_val, idx_test)
        if loss_val < best:
            torch.save(model.state_dict(), model_path)
            best = loss_val
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    # Testing
    acc_test, loss_test = test(adj, features, labels, idx_test, nclass,
                               idx_train, idx_val, model_path)

    ######  Find pesudo label  ########
    model = get_models(args, features.shape[1], nclass)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    output = model(features, adj)
    idx_train, pseudo_labels = generate_pesudo_label(output, idx_train, pseudo_labels,
                                                     idx_test, idx_val, nclass, topk_pc)


    # Model and optimizer
    model = get_models(args, features.shape[1], nclass)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    best = 100
    bad_counter = 0
    for epoch in range(args.epochs):
        loss_val, acc_train, loss_test, acc_test, output = train(epoch, model, optimizer, adj, features,
                                                                 pseudo_labels, idx_train,
                                                                 idx_val, idx_test)

        if loss_val < best:
            torch.save(model.state_dict(), model_path)
            best = loss_val
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    # Testing
    acc_test, loss_test = test(adj, features, labels, idx_test, nclass, idx_train,
                                    idx_val, model_path)
    acc_test_times_list.append(acc_test)



if __name__ == '__main__':
    main(args.dataset)




