import torch
from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_dim, n_outs=5, n_classes=3, hidden_dim=128):
        super().__init__()
        self.feature_extractor = nn.Linear(in_dim, hidden_dim)
        self.classifers = nn.ModuleList([nn.Linear(hidden_dim, n_classes)] * n_outs)

    def forward(self, x):
        feat = self.feature_extractor(x)
        outs = [F.softmax(classifier(feat)) for classifier in self.classifiers]
        return outs


model = Classifier(2)

