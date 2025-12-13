import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, params):
        super().__init__()

        self.dropout = params.dropout
        self.n_classes = out_dim
        self.device = params.device
        if params.use_batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        else:
            self.bn = None

        # Input embedding layer
        self.embedding = nn.Linear(in_dim, params.hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(params.num_layers):
            self.hidden_layers.append(nn.Linear(params.hidden_dim, params.hidden_dim))
        
        # Output layer
        self.out_layer = nn.Linear(params.hidden_dim, out_dim)
        
        # Initialize parameters
        self.reset_parameters()
    

    def reset_parameters(self):
        """Initialize weights using Glorot/Xavier initialization"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)


    def forward(self, feature, edge_index=None):

        # Input embedding with activation and dropout
        h = self.embedding(feature)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Process through hidden layers
        for layer in self.hidden_layers:
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Output layer (no activation, no dropout)
        h = self.out_layer(h)
        if self.bn:
                h = self.bn(h)
        
        return h
    
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
    
