import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_models
from util_calibration import _ECELoss
from torch import nn, optim
from torch.nn import functional as F

class CaGCN(nn.Module):
    def __init__(self, args, nclass, base_model):
        super(CaGCN, self).__init__()
        self.base_model = base_model
        self.scaling_model = get_models(args, nclass, 1, 'GCN', True)

        for para in self.base_model.parameters():
            para.requires_grad = False


    def forward(self, x, adj):
        logits = self.base_model(x ,adj)
        t = self.scaling_model(logits, adj)
        t = torch.log(torch.exp(t) + torch.tensor(1.1))
        output = logits * t
        return output


class MatrixScaling(nn.Module):
    def __init__(self, model, nfeat, n_bins):
        super(MatrixScaling, self).__init__()
        self.model = model
        self.nfeat = nfeat
        self.W = nn.Parameter(torch.ones(nfeat, nfeat))
        self.b = nn.Parameter(torch.ones(nfeat))
        self.n_bins = n_bins
        self.Lambda = 1e-2
        self.mu = 2e-1
        self.max_iter = 400

    def forward(self, features, adj):
        logits = self.model(features, adj)
        return self.matrix_scale(logits)

    def matrix_scale(self, logits):
        """
        Perform matrix scaling on logits
        """
        # Expand temperature to match the size of logits
        logits = torch.matmul(logits, self.W)
        logits = logits + self.b
        return logits

    def set_parameters(self, features, adj, labels, idx_val):
        """
        Tune the matrix of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss(self.n_bins).cuda()

        # First: collect all the logits and labels for the validation set

        with torch.no_grad():
            logits = self.model(features, adj)

        logits = logits[idx_val]
        labels = labels[idx_val]

        # Calculate NLL and ECE before temperature scaling
        before_scaling_nll = nll_criterion(logits, labels).item()
        before_scaling_ece = ece_criterion(logits, labels).item()
        print('Before scaling - NLL: %.3f, ECE: %.3f' % (before_scaling_nll, before_scaling_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.Adam([self.W, self.b], lr=0.01)

        for it in range(self.max_iter):
            optimizer.zero_grad()
            scale_logits = self.matrix_scale(logits)
            odir_w = (torch.sum(self.W**2) - torch.sum(torch.diag(self.W)**2)) / (self.nfeat * (self.nfeat-1))
            odir_b = torch.mm(torch.unsqueeze(self.b, 0), torch.unsqueeze(self.b, 1)) / self.nfeat
            odir_loss = self.Lambda * odir_w + self.mu * odir_b
            loss = nll_criterion(scale_logits, labels) + odir_loss
            loss.backward()
            optimizer.step()

        # Calculate NLL and ECE after temperature scaling
        after_scaling_nll = nll_criterion(self.matrix_scale(logits), labels).item()
        after_scaling_ece = ece_criterion(self.matrix_scale(logits), labels).item()
        print('After scaling - NLL: %.3f, ECE: %.3f' % (after_scaling_nll, after_scaling_ece))

        return self


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, n_bins):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.n_bins = n_bins

    def forward(self, features, adj):
        logits = self.model(features, adj)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_parameters(self, features, adj, labels, idx_val, lr, max_iter):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss(self.n_bins).cuda()

        # First: collect all the logits and labels for the validation set

        with torch.no_grad():
            logits = self.model(features, adj)

        logits = logits[idx_val]
        labels = labels[idx_val]

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

