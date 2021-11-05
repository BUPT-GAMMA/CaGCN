import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from utils import accuracy
from models_calibration import *
from utils import *
from util_calibration import _ECELoss
from util_calibration import *
import os

global result
result = []
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GraphSAGE')
parser.add_argument('--scaling_method', type=str, default=None)
parser.add_argument('--dataset', type=str, default="Cora",
                    help='dataset for training')
parser.add_argument('--stage', type=int, default=1,
                    help='times of repeat training')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--lr_for_cal', type=float, default=0.01)
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--labelrate', type=int, default=20)
parser.add_argument('--n_bins', type=int, default=20)
parser.add_argument('--Lambda', type=float, default=0.5,
                    help='the weight for ranking loss')
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--threshold', type=float, default=0.95)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)
conf_histogram = None

def generate_pesudo_label(output, idx_train, pseudo_labels, idx_test, idx_val):
    train_index = torch.where(idx_train==True)
    test_index = torch.where(idx_test==True)
    val_index = torch.where(idx_val==True)
    confidence, pred_label = get_confidence(output)
    index = torch.where(confidence>args.threshold)[0]
    for i in index:
        if i not in train_index[0] and i not in test_index[0] and i not in val_index[0]:
            pseudo_labels[i] = pred_label[i]
            idx_train[i] = True
    return idx_train, pseudo_labels

def train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val, idx_test):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    ece_criterion = _ECELoss(args.n_bins).cuda()
    ece = ece_criterion(output[idx_train], labels[idx_train])

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
def test(g, adj, features, labels, idx_test, nclass, idx_train,
         idx_val, model_b_scaling, model_a_scaling):
    nfeat = features.shape[1]
    if args.scaling_method == None:
        state_dict = torch.load(model_b_scaling)
        model = get_models(args, features.shape[1], nclass, g)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        output = model(features, adj)
        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        confidence = torch.softmax(output[idx_test], dim=1).cpu()
        confidence = torch.max(confidence, 1)[0]
        pred_label = torch.max(output[idx_test], 1)[1]
        correct_index = labels[idx_test] == pred_label
        correct_index = correct_index.cpu()
        plot_histograms(confidence[correct_index], confidence[np.invert(correct_index)],
                        'Uncal. - %s - %d - %s' % (args.dataset, args.labelrate, args.model), ['Correct', "InCorrect"])
        plot_acc_calibration(idx_test, output, labels, args.n_bins,
                             'Uncal. - %s - %d - %s' % (args.dataset, args.labelrate, args.model))
    else:
        state_dict = torch.load(model_a_scaling)
        base_model  = get_models(args, nfeat, nclass, g, scaling_model=False)
        if args.scaling_method == 'TS':
            model = ModelWithTemperature(base_model, args.n_bins)
        elif args.scaling_method == 'MS':
            model = MatrixScaling(base_model, nclass, args.n_bins)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        output = model(features, adj)
        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

    ece_criterion = _ECELoss(args.n_bins).cuda()
    ece = ece_criterion(output[idx_test], labels[idx_test]).item()
    brier_score = brier_score_criterion(output[idx_test], labels[idx_test], nclass).item()


    print(f"Test set results",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test:.4f}",
          f"ece = {ece:.4f}",
          f"brier_score = {brier_score:.4f}")

    return acc_test, loss_test, ece, brier_score


def temperature_scaling(g, adj, features, labels, nclass, idx_val, idx_train,
                        idx_test, model_b_scaling, model_a_scaling, n_time):
    state_dict = torch.load(model_b_scaling)
    base_model = get_models(args, features.shape[1], nclass, g, scaling_model=False)
    base_model.load_state_dict(state_dict)
    base_model.to(device)
    model = ModelWithTemperature(base_model, args.n_bins)
    if n_time == args.stage:
        model.set_parameters(features, adj, labels, idx_val, args.lr_for_cal, args.max_iter)
    else:
        model.set_parameters(features, adj, labels, idx_train, args.lr_for_cal, args.max_iter)
    torch.save(model.state_dict(), model_a_scaling)

def matrix_scaling(g, adj, features, labels, nclass, idx_val, idx_train,
                        idx_test, model_b_scaling, model_a_scaling):
    state_dict = torch.load(model_b_scaling)
    base_model = get_models(args, features.shape[1], nclass, g, scaling_model=False)
    base_model.load_state_dict(state_dict)
    base_model.to(device)
    model = MatrixScaling(base_model, nclass, args.n_bins)
    model.set_parameters(features, adj, labels, idx_val)
    torch.save(model.state_dict(), model_a_scaling)

def main(dataset):
    data, adj, features, labels, idx_train, idx_val, idx_test, nxg = load_data(dataset, args.labelrate)
    features = features.to(device)
    adj = adj.to(device)
    g = data[0].to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    nclass = labels.max().item() + 1

    pseudo_labels = labels.clone()
    pseudo_labels = pseudo_labels.to(device)
    acc_test_times_list = list()
    times_list = list()
    n_time = 0
    seed = np.random.randint(0, 10000)
    model_b_scaling = './save_model/%s-%s-%s-%d-w_o-s.pth' % (args.model, args.dataset, args.threshold, args.labelrate)
    model_a_scaling = './save_model/%s-%s-%s-%d-w-s.pth' % (args.model, args.dataset, args.threshold, args.labelrate)
    for times in range(0, args.stage):
        n_time += 1
        times_list.append(n_time)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Model and optimizer
        model = get_models(args, features.shape[1], nclass, g)
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
                torch.save(model.state_dict(), model_b_scaling)
                best = loss_val
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break

        ######  self-training to find pesudo label  ########
        if args.scaling_method != None:
            if args.scaling_method == 'TS':
                temperature_scaling(g, adj, features, labels, nclass, idx_val,
                                idx_train, idx_test, model_b_scaling, model_a_scaling, n_time)
                model = ModelWithTemperature(get_models(args, features.shape[1], nclass, g), args.n_bins)
            elif args.scaling_method == 'MS':
                matrix_scaling(g, adj, features, labels, nclass, idx_val,
                                idx_train, idx_test, model_b_scaling, model_a_scaling)
                model = MatrixScaling(get_models(args, features.shape[1], nclass, g), nclass, args.n_bins)
            state_dict = torch.load(model_a_scaling)
        else:
            state_dict = torch.load(model_b_scaling)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        output = model(features, adj)
        idx_train, pseudo_labels = generate_pesudo_label(output, idx_train, pseudo_labels, idx_test, idx_val)

        # Testing
        acc_test, loss_test, ece, brier_score = test(g, adj, features, labels, idx_test, nclass, idx_train,
                                        idx_val, model_b_scaling, model_a_scaling)
        acc_test_times_list.append(acc_test)

        print('nll: %f \nece: %f \nbrier_score: %f' % (loss_test.item(), ece, brier_score))

    print('acc_test:', acc_test_times_list)


if __name__ == '__main__':
    if args.scaling_method == 'MS':
        key = 'm'
    elif args.scaling_method == 'TS':
        key = 't'
    else:
        key = 'w-o'
    main(args.dataset)




