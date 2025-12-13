
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, device, residual=False):
        super().__init__()
        self.residual = residual and (in_dim == out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.n_classes = out_dim
        self.device = device
        self.conv = GCNConv(in_dim, out_dim)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        else:
            self.bn = None

    def forward(self, x, edge_index):
        x_in = x
        x = self.conv(x, edge_index)

        if self.bn:
            x = self.bn(x)

        x = F.relu(x)

        if self.residual:
            x = x + x_in

        return self.dropout(x)


class OutputMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2):
        """
        num_layers = number of hidden fully connected layers
        Each hidden layer halves the dimension of the previous one.
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            next_dim = current_dim // 2           # shrink size by half
            layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim                # update for next layer

        # Final output layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
    
        x = self.layers[-1](x)
        return x


class GCN(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        params,
    ):
        super().__init__()
        self.n_classes = out_dim
        self.device = params.device
        self.embed = nn.Linear(in_dim, params.hidden_dim)

        self.layers = nn.ModuleList([
            GCNLayer(params.hidden_dim, params.hidden_dim, params.activation, params.dropout, 
                     params.use_batchnorm, residual=params.use_residual, device=params.device)
            for _ in range(params.num_layers)
        ])

        self.classifier = OutputMLP(params.hidden_dim, out_dim, num_layers=params.num_layers)
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

        for layer in self.layers:
            layer.conv.reset_parameters()
            if layer.bn:
                layer.bn.reset_parameters()

        for layer in self.classifier.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


    def forward(self, x, edge_index):
        x = self.embed(x)
        x = F.dropout(x, p=0.5, training=self.training)

        for layer in self.layers:
            x = layer(x, edge_index)

        return self.classifier(x)


    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero(as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

    
