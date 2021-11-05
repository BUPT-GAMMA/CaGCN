import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import networkx as nx
from dgl.data import CiteseerGraphDataset
from dgl.data import CoraGraphDataset
from dgl.data import PubmedGraphDataset
from dgl.data import CoraFullDataset
from models import GCN, GAT, SpGAT, GCN_T, GraphSAGE



def load_data(dataset ,labelrate, os_path=''):
    if dataset == 'Cora':
        data = CoraGraphDataset()
    elif dataset == 'Citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'Pubmed':
        data = PubmedGraphDataset()
    elif dataset == 'CoraFull':
        data = CoraFullDataset()

    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']

    if dataset != 'CoraFull':
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        if labelrate != 20:
            labelrate -= 20
            nclass = data.num_classes
            start = int(torch.where(val_mask==True)[0][-1] + 1)
            train_mask[start:start+labelrate*nclass] = True


    else:
        datalength = labels.size()
        train_mask, val_mask, test_mask = torch.full((1, datalength[0]), fill_value=False, dtype=bool), torch.full(
            (1, datalength[0]), fill_value=False, dtype=bool) \
            , torch.full((1, datalength[0]), fill_value=False, dtype=bool)

        if os_path == None:
            os_path = 'data/corafull/'
        else:
            os_path += 'data/corafull/'
        with open(os_path + 'train%s.txt' % labelrate, 'r') as f:
            train_index = f.read().splitlines()
            train_index = list(map(int, train_index))
            train_mask[0][train_index] = 1
            train_mask = train_mask[0]
        with open(os_path + 'test%s.txt' % labelrate, 'r') as f:
            test_index = f.read().splitlines()
            test_index = list(map(int, test_index))
            test_mask[0][test_index] = 1
            test_mask = test_mask[0]
        with open(os_path + 'val%s.txt' % labelrate, 'r') as f:
            val_index = f.read().splitlines()
            val_index = list(map(int, val_index))
            val_mask[0][val_index] = 1
            val_mask = val_mask[0]
    nxg = g.to_networkx()
    adj = nx.to_scipy_sparse_matrix(nxg, dtype=np.float)

    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return data, adj, features, labels, train_mask, val_mask, test_mask, nxg


def preprocess_adj(adj, with_ego=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    if with_ego:
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    else:
        adj_normalized = normalize_adj(adj)
    return adj_normalized


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def accuracy(pred, targ):
    pred = torch.softmax(pred, dim=1)
    pred_max_index = torch.max(pred, 1)[1]
    ac = ((pred_max_index == targ).float()).sum().item() / targ.size()[0]
    return ac


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_models(args, nfeat, nclass, g, cal_model=None, scaling_model=False):
    if scaling_model:
        nhid = 16
        model_name = cal_model
    else:
        nhid = args.hidden
        model_name = args.model
    if model_name == 'GCN':
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=args.dropout)
    elif model_name == 'GAT':
        model = SpGAT(nfeat=nfeat,
                      nhid=args.hidden,
                      nclass=nclass,
                      dropout=args.dropout,
                      nheads=args.nb_heads,
                      alpha=args.alpha)
    elif model_name == 'GraphSAGE':
        model = GraphSAGE(g=g,
                          in_feats=nfeat,
                          n_hidden=nhid,
                          n_classes=nclass,
                          activation=F.relu,
                          dropout=args.dropout,
                          aggregator_type='gcn'
        )


    return model

def get_total_variation(data, confidence):
    ###get Laplacian matrix###
    g = data[0]
    adj = g.adj().to_dense()
    D = adj.sum(dim=1).diag()
    L = (D - adj).to_sparse()
    confidence = confidence.unsqueeze(dim=1).cpu()
    total_variation = torch.mm(L, confidence)
    total_variation = torch.mm(torch.transpose(total_variation, 0, 1), confidence)

    return total_variation
