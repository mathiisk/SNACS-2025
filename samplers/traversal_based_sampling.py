import torch

class RandomWalkSampler:
    """
    Basic random walk sampler with occasional restarts.

    Starting from a random node, we walk along random outgoing edges.
    With probability `restart_p` we ignore the current position and
    restart from a fresh random node.
    """

    def __init__(self, params):
        # Probability of jumping to a fresh random node
        self.restart_p = getattr(params, "rw_restart_p", 0.15)

 def sample(self, data, num_nodes_to_sample):
        edge_index = data.edge_index
        src, dst = edge_index
        device = src.device
        num_nodes = data.num_nodes

        # Build an undirected adjacency by duplicating edges.
        src_u = torch.cat([src, dst])
        dst_u = torch.cat([dst, src])

        # Sort edges by source node to get contiguous neighbor ranes
        perm = src_u.argsort()
        src_sorted = src_u[perm]
        dst_sorted = dst_u[perm]

        # Degree and prefix sums to locate neighbors for each node.
        deg = torch.bincount(src_sorted, minlength=num_nodes)
        cumdeg = deg.cumsum(0)
        start = cumdeg - deg

        # Mask tracks which nodes we have already visited.
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # Start the random walk from a random node
        start_point = torch.randint(0, num_nodes, (1,), device=device).item()
        cur = start_point

        counter = 0
        
        while mask.sum().item() < num_nodes_to_sample:
            # Mark current node as sampled.
            mask[cur] = True
            counter += 1
            if counter > num_nodes and mask.sum().item() < num_nodes_to_sample:
                cur = torch.randint(0, num_nodes, (1,), device=device).item()

            if torch.rand(1, device=device).item() < self.restart_p or deg[cur].item() == 0:
                # Restart: pick a new random node.
                cur = start_point
            else:
                # Follow one random outgoing edge from the current node.
                du = deg[cur].item()
                offset = torch.randint(0, du, (1,), device=device).item()
                eid = start[cur].item() + offset
                cur = dst_sorted[eid].item()

        # Convert mask to node indices.
        sampled_nodes = mask.nonzero(as_tuple=False).view(-1)

        # If we overshoot, randomly reduce to the requested size.
        if sampled_nodes.numel() > num_nodes_to_sample:
            perm_nodes = torch.randperm(sampled_nodes.numel(), device=device)
            sampled_nodes = sampled_nodes[perm_nodes[:num_nodes_to_sample]]

        return sampled_nodes


class RandomJumpSampler:
    """
    Random walk sampler with explicit "jump to a different node" moves.
    """

    def __init__(self, params):
        # Probability of performing a jump instead of a normal walk step.
        self.jump_p = getattr(params, "rj_jump_p", 0.15)

    def sample(self, data, num_nodes_to_sample):

        edge_index = data.edge_index
        src, dst = edge_index
        device = src.device
        num_nodes = data.num_nodes

        # Build undirected adjacency.
        src_u = torch.cat([src, dst])
        dst_u = torch.cat([dst, src])

        perm = src_u.argsort()
        src_sorted = src_u[perm]
        dst_sorted = dst_u[perm]

        deg = torch.bincount(src_sorted, minlength=num_nodes)
        cumdeg = deg.cumsum(0)
        start = cumdeg - deg

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # Initial node where the walk starts.
        cur = torch.randint(0, num_nodes, (1,), device=device).item()

        while mask.sum().item() < num_nodes_to_sample:
            # Mark current node as sampled.
            mask[cur] = True

            # Decide whether to jump or to follow an edge.
            do_jump = torch.rand(1, device=device).item() < self.jump_p or deg[cur].item() == 0
            if do_jump:
                # Jump: pick a different node
                cur = torch.randint(0, num_nodes, (1,), device=device).item()

            else:
                # Follow one random outgoing edge.
                du = deg[cur].item()
                offset = torch.randint(0, du, (1,), device=device).item()
                eid = start[cur].item() + offset
                cur = dst_sorted[eid].item()

        # Turn mask into indices.
        sampled_nodes = mask.nonzero(as_tuple=False).view(-1)

        if sampled_nodes.numel() > num_nodes_to_sample:
            perm_nodes = torch.randperm(sampled_nodes.numel(), device=device)
            sampled_nodes = sampled_nodes[perm_nodes[:num_nodes_to_sample]]

        return sampled_nodes


class RandomNeighborSampler:
    """
    At each step, pick a random node and then include all of its neighbours.
    """

    def __init__(self, params=None):
        pass

    def sample(self, data, num_nodes_to_sample):
        edge_index = data.edge_index
        src, dst = edge_index
        device = src.device
        num_nodes = data.num_nodes

        # Build undirected adjacency.
        src_u = torch.cat([src, dst])
        dst_u = torch.cat([dst, src])

        perm = src_u.argsort()
        src_sorted = src_u[perm]
        dst_sorted = dst_u[perm]

        deg = torch.bincount(src_sorted, minlength=num_nodes)
        cumdeg = deg.cumsum(0)
        start = cumdeg - deg

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        while mask.sum().item() < num_nodes_to_sample:
            # Randomly choose a center node.
            u = torch.randint(0, num_nodes, (1,), device=device).item()
            mask[u] = True

            # Add all neighbours of u (if any).
            du = deg[u].item()
            if du > 0:
                s = start[u].item()
                neighbors = dst_sorted[s : s + du]
                mask[neighbors] = True

        sampled_nodes = mask.nonzero(as_tuple=False).view(-1)

        if sampled_nodes.numel() > num_nodes_to_sample:
            perm_nodes = torch.randperm(sampled_nodes.numel(), device=device)
            sampled_nodes = sampled_nodes[perm_nodes[:num_nodes_to_sample]]

        return sampled_nodes


class ForestFireSampler:
    """
    Forest fire sampling.

    We start "fires" at random seed nodes and let them spread to neighbours
    with probability `forward_p`. When a frontier burns out, we start a new
    fire at another random node until we have enough sampled nodes.
    """

    def __init__(self, params):
        # Probability that an edge from a burning node "catches fire"
        self.forward_p = getattr(params, "ff_forward_p", 0.5)

    def sample(self, data, num_nodes_to_sample):
        edge_index = data.edge_index
        src, dst = edge_index
        device = src.device
        num_nodes = data.num_nodes

        # Build undirected adjacency by duplicating edges.
        src_u = torch.cat([src, dst])
        dst_u = torch.cat([dst, src])

        perm = src_u.argsort()
        src_sorted = src_u[perm]
        dst_sorted = dst_u[perm]

        # Degree and index ranges for neighbours.
        deg = torch.bincount(src_sorted, minlength=num_nodes)
        cumdeg = deg.cumsum(0)
        start = cumdeg - deg

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # Frontier holds "burning" nodes whose neighbours we will try next
        frontier = []

        while mask.sum().item() < num_nodes_to_sample:
            # If nothing is currently burning, start a fresh fire.
            if not frontier:
                seed = torch.randint(0, num_nodes, (1,), device=device).item()
                mask[seed] = True
                if deg[seed].item() > 0:
                    frontier.append(seed)
                continue

            # Take one active fire node from the frontier.
            u = frontier.pop()
            du = deg[u].item()
            if du == 0:
                # No neighbours to propagate to.
                continue

            s = start[u].item()
            neighbors = dst_sorted[s : s + du]

            # Decide which neighbours
            probs = torch.rand(du, device=device)
            burn_mask = probs < self.forward_p
            if burn_mask.any():
                burned = neighbors[burn_mask]
                # Only add nodes that were not in the sample yet.
                newly = burned[~mask[burned]]
                if burn_mask.any():
                    burned = neighbors[burn_mask]
                    # Only add nodes that were not in the sample yet.
                    newly = burned[~mask[burned]]
                    if newly.numel() > 0:
                        mask[newly] = True
                        # New burning nodes are added to the frontier.
                        frontier.extend(newly.tolist())

        sampled_nodes = mask.nonzero(as_tuple=False).view(-1)

        if sampled_nodes.numel() > num_nodes_to_sample:
            perm_nodes = torch.randperm(sampled_nodes.numel(), device=device)
            sampled_nodes = sampled_nodes[perm_nodes[:num_nodes_to_sample]]

        return sampled_nodes

