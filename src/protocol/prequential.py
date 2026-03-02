
# src/protocol/prequential.py


'''
How this works:
it's time for episode 5, this target is protein 102. 
here are the 200 query drugs you need to blindly rank

 '''
import torch
import numpy as np

class ProteinEpisode:
    def __init__(self, t, protein_idx, support_edges, support_labels, query_edges, query_labels):
        self.t = t
        self.protein_idx = protein_idx
        self.support_edges = support_edges   # [2, num_support] (protein_idx, drug_idx)
        self.support_labels = support_labels # [num_support] (pIC50 values)
        self.query_edges = query_edges       # [2, num_query]
        self.query_labels = query_labels     # [num_query]

def build_multiplex_stream(data, binds_metric="binds_pic50", support_k=8, min_edges=10, seed=42):
    """
    Builds the prequential stream of proteins.
    Only includes proteins with at least `min_edges` to ensure we have a valid query set.
    """
    rng = np.random.default_rng(seed)
    
    binds_ei = data["protein", binds_metric, "drug"].edge_index
    binds_y = data["protein", binds_metric, "drug"].edge_label
    
    # Count degree per protein
    num_proteins = data["protein"].num_nodes
    degrees = torch.bincount(binds_ei[0], minlength=num_proteins)
    
    # Valid proteins for the stream
    valid_proteins = torch.where(degrees >= min_edges)[0].tolist()
    
    # Shuffle for arrival order (can also sort by degree for dense-to-sparse curriculum)
    rng.shuffle(valid_proteins)
    
    episodes = []
    for t, p_idx in enumerate(valid_proteins):
        # Get all edges for this protein
        mask = binds_ei[0] == p_idx
        p_edges = binds_ei[:, mask]
        p_labels = binds_y[mask]
        
        # Shuffle edges to split support/query
        n_edges = p_edges.size(1)
        perm = torch.randperm(n_edges, generator=torch.Generator().manual_seed(seed + t))
        
        k_sup = min(support_k, n_edges // 2) # Ensure we have enough for query
        sup_idx = perm[:k_sup]
        qry_idx = perm[k_sup:]
        
        ep = ProteinEpisode(
            t=t,
            protein_idx=p_idx,
            support_edges=p_edges[:, sup_idx],
            support_labels=p_labels[sup_idx],
            query_edges=p_edges[:, qry_idx],
            query_labels=p_labels[qry_idx]
        )
        episodes.append(ep)
        
    return episodes