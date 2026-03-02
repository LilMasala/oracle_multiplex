
# src/protocol/prequential.py


'''
How this works:
it's time for episode 5, this target is protein 102. 
here are the 200 query drugs you need to blindly rank

 '''
import torch
import numpy as np

class ProteinEpisode:
    def __init__(self, t, protein_idx, edges, labels):
        self.t = t
        self.protein_idx = protein_idx
        # ALL known edges for this protein (treated as zero-shot queries first)
        self.edges = edges       # [2, num_edges] (protein_idx, drug_idx)
        self.labels = labels     # [num_edges] (pIC50 values)

def build_multiplex_stream(data, binds_metric="binds_pic50", min_edges=10, seed=42):
    """
    Builds the prequential stream of proteins for a Pure Cold-Start Screener.
    No support/query split: the model must rank ALL known drugs for a protein blindly.
    """
    rng = np.random.default_rng(seed)
    
    binds_ei = data["protein", binds_metric, "drug"].edge_index
    binds_y = data["protein", binds_metric, "drug"].edge_label
    
    # Count degree per protein
    num_proteins = data["protein"].num_nodes
    degrees = torch.bincount(binds_ei[0], minlength=num_proteins)
    
    # Valid proteins for the stream (min 10 edges to ensure meaningful CI calculation)
    valid_proteins = torch.where(degrees >= min_edges)[0].tolist()
    
    # Randomly shuffle so stream difficulty is uniformly distributed!
    rng.shuffle(valid_proteins)
    
    episodes = []
    for t, p_idx in enumerate(valid_proteins):
        # Extract ALL edges and labels for this specific protein
        mask = binds_ei[0] == p_idx
        p_edges = binds_ei[:, mask]
        p_labels = binds_y[mask]
        
        ep = ProteinEpisode(
            t=t,
            protein_idx=p_idx,
            edges=p_edges,
            labels=p_labels
        )
        episodes.append(ep)
        
    return episodes