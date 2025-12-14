import torch
from dataclasses import dataclass, field
from typing import List, Dict, Callable
import torch.nn.functional as F

# Samplers
from samplers.node_based_sampling import (
    RandomNodeSampler,
    RandomDegreeNodeSampler,
    RandomPageRankSampler
)
from samplers.edge_based_sampling import(
    RandomEdgeSampler,
    RandomNodeEdgeSampler,
    HybridSampling
)
from samplers.traversal_based_sampling import(
    RandomWalkSampler,
    RandomJumpSampler,
    RandomNeighborSampler,
    ForestFireSampler
)

# Models
from models.mlp import MLP
from models.gcn import GCN


@dataclass
class Config:
    
    # ----------------------------------------------------------------------
    # Basic experiment settings
    # ----------------------------------------------------------------------
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds: list[int] = field(default_factory=lambda: [i for i in range(1, 11)])

    # ----------------------------------------------------------------------
    # Dataset configuration
    # ----------------------------------------------------------------------
    datasets: List[str] = field(default_factory=lambda: ["PubMed", "Cora", "ogbn_arxiv", "CiteSeer", "Physics", "CS", "Computers", "Photo"]) # "PubMed", "Cora" "ogbn_arxiv, "CiteSeer", "Physics", "Computers", "Photo", "CS"
    sampling_ratios: List[float] = field(default_factory=lambda: [1, 0.5, 0.3, 0.1, 0.05]) # 0.5, 0.3, 0.1, 0.05
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # ----------------------------------------------------------------------
    # Samplers
    # ----------------------------------------------------------------------
    samplers: Dict[str, Callable] = field(default_factory=lambda: {
        "random_node": RandomNodeSampler,
        "random_degree": RandomDegreeNodeSampler,
        "random_pagerank": RandomPageRankSampler,
        "random_edge": RandomEdgeSampler,
        "random_node_edge": RandomNodeEdgeSampler,
        "random_hybrid": HybridSampling,
        "random_walk": RandomWalkSampler,
        "random_jump": RandomJumpSampler,
        "random_neigbour": RandomNeighborSampler,
        "forest_fire": ForestFireSampler
    })

    # Parameters specific to certain samplers
    page_rank_max_iter: int = 100
    page_rank_damping_factor: float = 0.85
    hyb_p: float = 0.8
    rw_restart_p: float = 0.15
    rj_jump_p: float = 0.15
    ff_forward_p: float = 0.5


    # ----------------------------------------------------------------------
    # Models
    # ----------------------------------------------------------------------
    models: Dict[str, Callable] = field(default_factory=lambda: {
        "mlp": MLP,
        "gcn": GCN,
    })


    # Architecture hyperparameters (shared across models that use them)
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    activation: Callable = field(default=F.relu)
    use_residual: bool = True
    use_batchnorm: bool = True


    # ----------------------------------------------------------------------
    # Training configuration
    # ----------------------------------------------------------------------
    optimizer: Callable = torch.optim.Adam
    lr: float = 0.01
    min_lr: float = 1e-5
    lr_schedule_patience: int = 50
    lr_reduce_factor: float = 0.5
    weight_decay: float = 5e-4
    n_epochs: int = 1000
    print_every: int = 20
    epoch_patience: int = 100
    min_delta: float = 0.1

