"""
Diversity-aware experience replay buffer for the prequential stream.

Rather than replaying uniformly random past proteins, this maintains a set
that maximally covers the protein feature space — so the model keeps seeing
examples from across the distribution (kinases, GPCRs, enzymes, etc.) rather
than drifting toward whatever types dominate the recent stream.

Representation
--------------
Each protein is sketched into a low-dim space via a fixed random projection
of its target_features (2816 → sketch_dim, default 64). This is fast and
captures the overall protein type without needing any learned embeddings.

Eviction (when buffer is full)
-------------------------------
Evict the protein whose removal costs the least diversity — specifically the
one with the highest cosine similarity to its nearest neighbour in the buffer.
This is a greedy coreset: every addition keeps the set as spread-out as
possible.

Sampling
--------
Farthest-first traversal: start from a random seed protein, then iteratively
pick the protein in the buffer that is farthest (lowest max-similarity) from
the already-selected set. Returns a maximally diverse subset of size n.
"""
import torch
import torch.nn.functional as F


class DiverseReplayBuffer:
    def __init__(
        self,
        max_size: int,
        protein_dim: int,
        sketch_dim: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.max_size   = max_size
        self.sketch_dim = sketch_dim
        self.device     = device

        # Fixed random projection: protein_dim → sketch_dim
        # Seeded separately so it doesn't affect the main training PRNG.
        gen = torch.Generator()
        gen.manual_seed(1337)
        self.proj = torch.randn(protein_dim, sketch_dim, generator=gen) / (protein_dim ** 0.5)
        self.proj = self.proj.to(device)

        self._indices: list[int]        = []   # protein_idx for each slot
        self._sketches: list[torch.Tensor] = [] # [sketch_dim] cpu tensors

    # ── public API ─────────────────────────────────────────────────────────

    def add(self, protein_idx: int, target_features: torch.Tensor):
        """Add a newly-revealed protein to the buffer."""
        sketch = self._sketch(target_features)
        if len(self._indices) < self.max_size:
            self._indices.append(protein_idx)
            self._sketches.append(sketch)
        else:
            self._evict_and_add(protein_idx, sketch)

    def sample(self, n: int) -> list[int]:
        """
        Return up to n protein indices via farthest-first sampling —
        maximally diverse subset of the buffer.
        """
        M = len(self._indices)
        if M == 0:
            return []
        n = min(n, M)
        if n == M:
            return list(self._indices)
        return self._farthest_first(n)

    def __len__(self) -> int:
        return len(self._indices)

    # ── internals ──────────────────────────────────────────────────────────

    def _sketch(self, features: torch.Tensor) -> torch.Tensor:
        """Project protein features to sketch space (returns cpu tensor)."""
        with torch.no_grad():
            s = (features.to(self.device).float() @ self.proj)
        return s.cpu()

    def _sketches_matrix(self) -> torch.Tensor:
        """Stack all sketches into [M, sketch_dim] on cpu."""
        return torch.stack(self._sketches)  # [M, D]

    def _evict_and_add(self, new_idx: int, new_sketch: torch.Tensor):
        """
        Consider the buffer + the new protein as a candidate set of max_size+1.
        Evict whichever element is most similar to its nearest neighbour
        (i.e. contributes least unique coverage). If that element is the new
        protein itself, simply don't add it.
        """
        sketches = self._sketches_matrix()                              # [N, D]
        candidate = torch.cat([sketches, new_sketch.unsqueeze(0)])      # [N+1, D]

        normed = F.normalize(candidate.float(), dim=1)
        sims   = normed @ normed.t()                                    # [N+1, N+1]
        sims.fill_diagonal_(-2.0)                                       # exclude self

        max_sim_to_any = sims.max(dim=1).values                        # [N+1]
        evict = int(max_sim_to_any.argmax())

        if evict == len(self._indices):
            # New protein is redundant — keep existing buffer unchanged
            return
        self._indices[evict]  = new_idx
        self._sketches[evict] = new_sketch

    def _farthest_first(self, n: int) -> list[int]:
        """Greedy farthest-first traversal in sketch space."""
        sketches = self._sketches_matrix()                              # [M, D]
        normed   = F.normalize(sketches.float(), dim=1)                 # [M, D]
        M        = len(self._indices)

        # Seed: random starting point
        seed = torch.randint(M, (1,)).item()
        selected = [seed]
        # min distance from each point to the selected set
        min_sim_to_sel = (normed @ normed[seed]).clone()               # [M]

        for _ in range(n - 1):
            # Mask already selected points so they're never re-picked
            for s in selected:
                min_sim_to_sel[s] = 2.0
            nxt = int(min_sim_to_sel.argmin())
            selected.append(nxt)
            # Update min distances
            new_sims        = normed @ normed[nxt]                     # [M]
            min_sim_to_sel  = torch.minimum(min_sim_to_sel, new_sims)

        return [self._indices[i] for i in selected]
