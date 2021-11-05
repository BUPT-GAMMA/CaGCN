import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from sklearn.isotonic import IsotonicRegression
import seaborn as sns
import networkx as nx
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import *

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece +=  torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def brier_score_criterion(logits, labels, nclass):
    logits = torch.softmax(logits, dim=1)
    labels = F.one_hot(labels, nclass)
    loss_criterion = nn.MSELoss()
    brier_score = loss_criterion(logits, labels) * nclass
    return brier_score

def irova_calibrate(logit, label, logit_eval):
    p = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    p_eval = np.exp(logit_eval) / np.sum(np.exp(logit_eval), 1)[:, None]

    for ii in range(p_eval.shape[1]):
        ir = IsotonicRegression(out_of_bounds='clip')
        y_ = ir.fit_transform(p[:, ii], label[:, ii])
        p_eval[:, ii] = ir.predict(p_eval[:, ii]) + 1e-9 * p_eval[:, ii]

    return p_eval

def plot_acc_calibration(idx_test, output, labels, n_bins, title):
    output = torch.softmax(output, dim=1)
    pred_label = torch.max(output[idx_test], 1)[1]
    p_value = torch.max(output[idx_test], 1)[0]
    ground_truth = labels[idx_test]
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
    for index, value in enumerate(p_value):
        #value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) -0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]

    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    plt.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
            alpha=0.7, width=0.03, color='dodgerblue', label='Outputs')
    plt.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.03, color='lightcoral', label='Expected')
    plt.plot([0,1], [0,1], ls='--',c='k')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    #title = 'Uncal. - Cora - 20 - GCN'
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.savefig('output/' + title +'.png', format='png', dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    plt.show()

def plot_histograms(content_a, content_b, title, labeltitle, n_bins=50, norm_hist=True):
    # Plot histogram of correctly classified and misclassified examples
    global conf_histogram

    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    sns.distplot(content_a, kde=False, bins=n_bins, norm_hist=False, fit=None, label=labeltitle[0])
    sns.distplot(content_b, kde=False, bins=n_bins, norm_hist=False,  fit=None, label=labeltitle[1])
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.savefig('output/' + title +'.png', format='png', transparent=True, dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    plt.show()



def get_confidence(output, with_softmax=False):

    if not with_softmax:
        output = torch.softmax(output, dim=1)

    confidence, pred_label = torch.max(output, dim=1)

    return confidence, pred_label



def intra_distance_loss(output, labels):
    #loss = torch.ones(1, requires_grad = True)
    output = torch.softmax(output, dim=1)
    pred_max_index = torch.max(output, 1)[1]
    correct_i = torch.where(pred_max_index==labels)
    incorrect_i = torch.where(pred_max_index!=labels)
    output = torch.sort(output, dim=1, descending=True)
    pred,sub_pred = output[0][:,0], output[0][:,1]
    loss = (torch.sum(1 - pred[correct_i] + sub_pred[correct_i]) + torch.sum(pred[incorrect_i]-sub_pred[incorrect_i])) / labels.size()[0]
    return loss
