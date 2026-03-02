import torch

class MultiplexPillarSampler:
    """
    Extracts the 'Vertical Pillar' context for a target protein.
    Returns the raw neighbor IDs, their features, and the bipartite graph of their known binding edges.
    """
    def __init__(self, hetero_data, binds_metric="binds_pic50", temporal_decay=0.01):
        self.data = hetero_data
        self.binds_metric = binds_metric
        
        # Pre-extract edge indices for fast lookup
        self.form_ei = self.data["protein", "similar", "protein"].edge_index
        self.role_ei = self.data["protein", "go_shared", "protein"].edge_index
        self.binds_ei = self.data["protein", self.binds_metric, "drug"].edge_index
        self.binds_y = self.data["protein", self.binds_metric, "drug"].edge_label
        self.binds_w = getattr(self.data["protein", self.binds_metric, "drug"], "edge_weight", None)
        if self.binds_w is None:
            self.binds_w = torch.ones_like(self.binds_y, dtype=torch.float)
        self.temporal_decay = temporal_decay
        self.current_episode = 0
        self.edge_birth_t = torch.zeros(self.binds_y.size(0), device=self.binds_y.device)
        
        # <-- ADDED: We must grab the raw features so we can pass them to Phase 2
        self.protein_x = self.data["protein"].x
        self._bind_index_cache = {}
        self._build_bind_index_cache()

    def _build_bind_index_cache(self):
        self._bind_index_cache = {}
        for edge_idx, prot_idx in enumerate(self.binds_ei[0].tolist()):
            if prot_idx not in self._bind_index_cache:
                self._bind_index_cache[prot_idx] = []
            self._bind_index_cache[prot_idx].append(edge_idx)

    def begin_episode(self, episode_idx):
        self.current_episode = int(episode_idx)

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
            return (
                torch.empty((2, 0), dtype=torch.long, device=self.binds_ei.device),
                torch.empty((0,), dtype=torch.float, device=self.binds_y.device),
                torch.empty((0,), dtype=torch.float, device=self.binds_w.device),
            )

        edge_ids = []
        for n_idx in neighbor_indices.tolist():
            edge_ids.extend(self._bind_index_cache.get(int(n_idx), []))

        if not edge_ids:
            return (
                torch.empty((2, 0), dtype=torch.long, device=self.binds_ei.device),
                torch.empty((0,), dtype=torch.float, device=self.binds_y.device),
                torch.empty((0,), dtype=torch.float, device=self.binds_w.device),
            )

        edge_ids = torch.tensor(edge_ids, dtype=torch.long, device=self.binds_ei.device)
        n_binds_ei = self.binds_ei[:, edge_ids]
        n_binds_y = self.binds_y[edge_ids]
        base_w = self.binds_w[edge_ids]
        age = (self.current_episode - self.edge_birth_t[edge_ids]).clamp(min=0)
        decay = torch.exp(-self.temporal_decay * age)
        n_binds_w = base_w * decay
        return n_binds_ei, n_binds_y, n_binds_w

    def get_pillar_context(self, target_idx):
        """
        Returns the structural and functional context for a given protein.
        """
        device = self.protein_x.device
        
        # 1. Get Form (Structural) Neighbors and their raw binding history
        form_neighbors = self._get_neighbors(self.form_ei, target_idx)
        form_binds_ei, form_binds_y, form_binds_w = self._get_neighbor_binding_edges(form_neighbors)
        
        # <-- ADDED: Fetch the actual CATH/ESM features for these neighbors
        if form_neighbors.numel() > 0:
            form_features = self.protein_x[form_neighbors]
        else:
            form_features = torch.empty((0, self.protein_x.size(1)), device=device)

        # 2. Get Role (Functional) Neighbors and their raw binding history
        role_neighbors = self._get_neighbors(self.role_ei, target_idx)
        role_binds_ei, role_binds_y, role_binds_w = self._get_neighbor_binding_edges(role_neighbors)

        form_set = set(form_neighbors.tolist())
        role_set = set(role_neighbors.tolist())
        union = len(form_set | role_set)
        jaccard = float(len(form_set & role_set) / union) if union > 0 else 0.0
        
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
            "form_binds_w": form_binds_w,
            
            # Role Floor Topology & Raw History
            "role_neighbors": role_neighbors,
            "role_features": role_features,                
            "role_binds_ei": role_binds_ei,
            "role_binds_y": role_binds_y,
            "role_binds_w": role_binds_w,
            "cross_floor_jaccard": torch.tensor(jaccard, device=device, dtype=torch.float),
        }
    
    def add_revealed_edges(self, new_edges, new_labels, new_weights=None):
        """
        Dynamically updates the graph's known binding history.
        This allows future proteins to 'leak' these exact labels if this protein becomes their neighbor.
        """
        device = self.binds_ei.device
        if new_weights is None:
            new_weights = torch.ones_like(new_labels, dtype=torch.float)

        start_idx = self.binds_ei.size(1)
        self.binds_ei = torch.cat([self.binds_ei, new_edges.to(device)], dim=1)
        self.binds_y = torch.cat([self.binds_y, new_labels.to(device)], dim=0)
        self.binds_w = torch.cat([self.binds_w, new_weights.to(device)], dim=0)
        new_birth = torch.full((new_labels.size(0),), float(self.current_episode), device=device)
        self.edge_birth_t = torch.cat([self.edge_birth_t, new_birth], dim=0)

        src_nodes = new_edges[0].tolist()
        for i, src in enumerate(src_nodes):
            if src not in self._bind_index_cache:
                self._bind_index_cache[src] = []
            self._bind_index_cache[src].append(start_idx + i)
