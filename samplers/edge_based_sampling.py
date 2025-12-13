import torch


class RandomEdgeSampler:
    """
    Sample nodes by first sampling random edges and then taking
    all endpoints of those edges.
    """

    def __init__(self, params=None):
        # Nothing to configure yet, but we keep params for API symmetry.
        pass

    def sample(self, data, num_nodes_to_sample):
  
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        device = edge_index.device

        # Boolean mask of which nodes we've already picked
        mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)

        while mask.sum().item() < num_nodes_to_sample:
            # Sample num_nodes_to_sample edges
            edge_ids = torch.randint(
                low=0,
                high=num_edges,
                size=(num_nodes_to_sample,),
                device=device,
            )
            # Take both endpoints of each chosen edge.
            nodes = edge_index[:, edge_ids].reshape(-1)
            mask[nodes] = True

        # Turn the boolean mask into a flat list of node indices
        sampled_nodes = mask.nonzero(as_tuple=False).view(-1)

        # If we overshot (we probably will), randomly trim back down to the requested size.
        if sampled_nodes.numel() > num_nodes_to_sample:
            perm = torch.randperm(sampled_nodes.numel(), device=device)
            sampled_nodes = sampled_nodes[perm[:num_nodes_to_sample]]

        return sampled_nodes


class RandomNodeEdgeSampler:
    """
    First pick a random node, then pick a random edge incident to that node
    (if it has any), and keep both endpoints.
    """

    def __init__(self, params=None):
        pass

    def sample(self, data, num_nodes_to_sample):

        edge_index = data.edge_index
        src, dst = edge_index
        device = src.device
        num_nodes = data.num_nodes

        # Sort edges by source node so we can do quick "neighbors of u" lookups.
        perm = src.argsort()
        src_sorted = src[perm]
        dst_sorted = dst[perm]

        # Degree of each node (out-degree in this directed view).
        deg = torch.bincount(src_sorted, minlength=num_nodes)
        cumdeg = deg.cumsum(0)
        start = cumdeg - deg  # where neighbors of node u start in dst_sorted

        # Boolean mask of which nodes we've already picked
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        while mask.sum().item() < num_nodes_to_sample:
            # Pick a random node u
            u = torch.randint(0, num_nodes, (1,), device=device).item()

            if deg[u].item() == 0:
                # Isolated node
                mask[u] = True
            else:
                # Node u has deg[u] outgoing edges in the sorted structure.
                du = deg[u].item()
                offset = torch.randint(0, du, (1,), device=device).item() # sample a neigbour
                eid_sorted = start[u].item() + offset 
                v = dst_sorted[eid_sorted]

                # Add both u and its randomly chosen neighbor v.
                mask[u] = True
                mask[v] = True

        sampled_nodes = mask.nonzero(as_tuple=False).view(-1)

        # If we gathered too many, randomly shave off the extras.
        if sampled_nodes.numel() > num_nodes_to_sample:
            perm_nodes = torch.randperm(sampled_nodes.numel(), device=device)
            sampled_nodes = sampled_nodes[perm_nodes[:num_nodes_to_sample]]

        return sampled_nodes


class HybridSampling:
    """
    Hybrid node sampler:

    * with probability `hyb_p`: do a RandomNodeEdgeSampler step;
    * with probability `1 - hyb_p`: do a RandomEdgeSampler step;
    """

    def __init__(self, params):
        self.hyb_p = getattr(params, "hyb_p", 0.8)

    def sample(self, data, num_nodes_to_sample):

        edge_index = data.edge_index
        src, dst = edge_index
        device = src.device
        num_nodes = data.num_nodes
        num_edges = edge_index.size(1)

        # Precompute the same sorted adjacency structure as before.
        perm = src.argsort()
        src_sorted = src[perm]
        dst_sorted = dst[perm]

        deg = torch.bincount(src_sorted, minlength=num_nodes)
        cumdeg = deg.cumsum(0)
        start = cumdeg - deg

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        while mask.sum().item() < num_nodes_to_sample:

            if torch.rand(1, device=device).item() < self.hyb_p:
                # Node-edge step: pick u, then one of its edges 
                u = torch.randint(0, num_nodes, (1,), device=device).item()
                if deg[u].item() == 0:
                    # Isolated node: just add u.
                    mask[u] = True
                else:
                    du = deg[u].item()
                    offset = torch.randint(0, du, (1,), device=device).item()
                    eid_sorted = start[u].item() + offset
                    v = dst_sorted[eid_sorted]
                    mask[u] = True
                    mask[v] = True
            else:
                # Edge step: pick a random edge 
                eid = torch.randint(0, num_edges, (1,), device=device)
                u, v = edge_index[:, eid].view(-1)
                mask[u] = True
                mask[v] = True

        sampled_nodes = mask.nonzero(as_tuple=False).view(-1)
        if sampled_nodes.numel() > num_nodes_to_sample:
            perm_nodes = torch.randperm(sampled_nodes.numel(), device=device)
            sampled_nodes = sampled_nodes[perm_nodes[:num_nodes_to_sample]]

        return sampled_nodes



