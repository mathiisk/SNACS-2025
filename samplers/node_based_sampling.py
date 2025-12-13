import torch
import networkx as nx
from torch_geometric.utils import to_networkx

class RandomNodeSampler:
    
    def __init__(self, params):
        pass

    def sample(self, data, num_nodes_to_sample):
        """
        Sample nodes randomly, weighted by degree.

        Args:
            data: PyG Data object containing edge_index
            num_nodes_to_sample: number of nodes to sample

        Returns:
            sampled_nodes: 1D tensor of sampled node indices
        """

        num_total_nodes = data.num_nodes
        sampled_nodes = torch.randperm(num_total_nodes, device=data.x.device)[:num_nodes_to_sample]
        return sampled_nodes


class RandomDegreeNodeSampler:

    def __init__(self, params):
        pass

    def sample(self, data, num_nodes_to_sample):
        """
        Sample nodes randomly, weighted by degree.

        Args:
            data: PyG Data object containing edge_index
            num_nodes_to_sample: number of nodes to sample

        Returns:
            sampled_nodes: 1D tensor of sampled node indices
        """
        num_nodes = data.num_nodes
        src, dst = data.edge_index

        # Compute node degrees
        deg = torch.zeros(num_nodes, device=data.x.device)
        # Adds one for the corrponding index from src or dst tensor
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float)) 

        # Avoid zero probability for isolated nodes
        deg = deg + 1e-6  

        # Normalize to get probabilities
        probs = deg / deg.sum()

        # Sample nodes according to probabilities without replacement
        sampled_nodes = torch.multinomial(probs, num_nodes_to_sample, replacement=False)
        return sampled_nodes


class RandomPageRankSampler:
    """
    Randomly samples nodes from a PyG Data graph based on PageRank scores.
    Higher PageRank nodes are more likely to be sampled.
    """

    def __init__(self, params):

        self.damping_factor = self.jump_p = getattr(params, "page_rank_damping_factor", 0.85) 
        self.max_iter = self.jump_p = getattr(params, "page_rank_max_iter", 100) 
        
    def sample(self, data, num_nodes_to_sample):
        """
        Sample nodes based on PageRank probabilities.

        Args:
            data: PyG Data object
            num_nodes_to_sample: number of nodes to sample

        Returns:
            sampled_nodes: 1D tensor of sampled node indices
        """
        # Convert PyG Data to NetworkX graph
        G = to_networkx(data, to_undirected=False)

        # Compute PageRank scores
        pr = nx.pagerank(G, alpha=self.damping_factor, max_iter=self.max_iter)

        # Convert scores to a probability tensor
        pr_scores = torch.tensor([pr[i] for i in range(data.num_nodes)], device=data.x.device)
        pr_probs = pr_scores / pr_scores.sum()

        # Sample nodes according to PageRank probabilities without replacement
        sampled_nodes = torch.multinomial(pr_probs, num_nodes_to_sample, replacement=False)
        return sampled_nodes

