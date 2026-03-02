import torch

class MultiplexPillarSampler:
    """
    Extracts the 'Vertical Pillar' context for a target protein.
    Returns the raw neighbor IDs, their features, and the bipartite graph of their known binding edges.
    """
    def __init__(self, hetero_data, binds_metric="binds_pic50"):
        self.data = hetero_data
        self.binds_metric = binds_metric
        
        # Pre-extract edge indices for fast lookup
        self.form_ei = self.data["protein", "similar", "protein"].edge_index
        self.role_ei = self.data["protein", "go_shared", "protein"].edge_index
        self.binds_ei = self.data["protein", self.binds_metric, "drug"].edge_index
        self.binds_y = self.data["protein", self.binds_metric, "drug"].edge_label
        
        # <-- ADDED: We must grab the raw features so we can pass them to Phase 2
        self.protein_x = self.data["protein"].x

    def _get_neighbors(self, edge_index, target_idx):
        """Finds all outgoing neighbors for a specific target node."""
        mask = edge_index[0] == target_idx
        return edge_index[1][mask]

    def _get_neighbor_binding_edges(self, neighbor_indices):
        """
        Retrieves the exact bipartite edges (Neighbor -> Drug) and their pIC50 labels.
        This provides the raw material for the Phase 2 'Leak' attention mechanism.
        """
        if neighbor_indices.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
            
        # Find all binding edges where the source protein is in our neighbor list
        # Using torch.isin for fast vectorized matching
        mask = torch.isin(self.binds_ei[0], neighbor_indices)
        
        n_binds_ei = self.binds_ei[:, mask]
        n_binds_y = self.binds_y[mask]
        
        return n_binds_ei, n_binds_y

    def get_pillar_context(self, target_idx):
        """
        Returns the structural and functional context for a given protein.
        """
        device = self.protein_x.device
        
        # 1. Get Form (Structural) Neighbors and their raw binding history
        form_neighbors = self._get_neighbors(self.form_ei, target_idx)
        form_binds_ei, form_binds_y = self._get_neighbor_binding_edges(form_neighbors)
        
        # <-- ADDED: Fetch the actual CATH/ESM features for these neighbors
        if form_neighbors.numel() > 0:
            form_features = self.protein_x[form_neighbors]
        else:
            form_features = torch.empty((0, self.protein_x.size(1)), device=device)

        # 2. Get Role (Functional) Neighbors and their raw binding history
        role_neighbors = self._get_neighbors(self.role_ei, target_idx)
        role_binds_ei, role_binds_y = self._get_neighbor_binding_edges(role_neighbors)
        
        # <-- ADDED: Fetch the actual GO features for these neighbors
        if role_neighbors.numel() > 0:
            role_features = self.protein_x[role_neighbors]
        else:
            role_features = torch.empty((0, self.protein_x.size(1)), device=device)

        return {
            "target_idx": target_idx,
            "target_features": self.protein_x[target_idx], 
            
            # Form Floor Topology & Raw History
            "form_neighbors": form_neighbors,
            "form_features": form_features,                
            "form_binds_ei": form_binds_ei,
            "form_binds_y": form_binds_y,
            
            # Role Floor Topology & Raw History
            "role_neighbors": role_neighbors,
            "role_features": role_features,                
            "role_binds_ei": role_binds_ei,
            "role_binds_y": role_binds_y
        }
    
    def add_revealed_edges(self, new_edges, new_labels):
        """
        Dynamically updates the graph's known binding history.
        This allows future proteins to 'leak' these exact labels if this protein becomes their neighbor.
        """
        device = self.binds_ei.device
        self.binds_ei = torch.cat([self.binds_ei, new_edges.to(device)], dim=1)
        self.binds_y = torch.cat([self.binds_y, new_labels.to(device)], dim=0)