import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam
from torch.distributions import constraints


class ExpertScorer(nn.Module):
    def __init__(self, protein_dim, drug_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        # z_p input shape: [N, z_refined(protein_dim) + v_prior(drug_dim) + delta_mean(protein_dim)]
        self.p_proj = nn.Linear((2 * protein_dim) + drug_dim, hidden_dim)
        # z_d input shape: [N, drug_dim + 2 * drug_dim] after concatenating [drug_emb, v_prior, delta_mean]
        self.d_proj = nn.Linear((2 * drug_dim) + protein_dim, hidden_dim)
        self.interaction = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_p, z_d):
        p = self.p_proj(z_p)
        d = self.d_proj(z_d)
        if p.size(0) == 1 and d.size(0) > 1:
            p = p.expand(d.size(0), -1)
        joint_rep = self.interaction(p, d)
        return self.mlp(joint_rep).squeeze(-1)


class MultiplexMoEGate(nn.Module):
    """
    Trust-aware gate.
    Input = [z_t, v_prior, delta_mean, trust_vector].
    """

    def __init__(self, protein_dim, drug_dim, trust_dim=5, num_experts=4, hidden_dim=128, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Sequential(
            nn.Linear((2 * protein_dim) + drug_dim + trust_dim, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, z_t, v_prior, delta_mean, trust_vector):
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        if v_prior.dim() == 1:
            v_prior = v_prior.unsqueeze(0)
        if delta_mean.dim() == 1:
            delta_mean = delta_mean.unsqueeze(0)
        if trust_vector.dim() == 1:
            trust_vector = trust_vector.unsqueeze(0)

        # gate_input: [B, 2*protein_dim + drug_dim + trust_dim]
        gate_input = torch.cat([z_t, v_prior, delta_mean, trust_vector], dim=-1)
        logits = self.router(gate_input)

        k = min(self.top_k, logits.size(-1))
        if k < logits.size(-1):
            top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            sparse_logits.scatter_(dim=-1, index=top_idx, src=top_vals)
            logits = sparse_logits

        return F.softmax(logits, dim=-1)


class MultiplexRoutingHead(nn.Module):
    def __init__(self, protein_dim, drug_dim, num_experts=4, top_k=2, trust_dim=5):
        super().__init__()
        self.gate = MultiplexMoEGate(
            protein_dim=protein_dim,
            drug_dim=drug_dim,
            trust_dim=trust_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.experts = nn.ModuleList([ExpertScorer(protein_dim, drug_dim) for _ in range(num_experts)])

    def forward(self, z_refined, protein_raw_features, v_prior, delta_mean, query_drug_features, trust_vector):
        if z_refined.dim() == 1:
            z_refined = z_refined.unsqueeze(0)

        gate_probs = self.gate(protein_raw_features, v_prior, delta_mean, trust_vector)

        num_queries = query_drug_features.size(0)
        prior_rep = v_prior.unsqueeze(0).expand(num_queries, -1)
        delta_rep = delta_mean.unsqueeze(0).expand(num_queries, -1)
        expert_protein_input = torch.cat([z_refined.expand(num_queries, -1), prior_rep, delta_rep], dim=-1)
        expert_drug_input = torch.cat([query_drug_features, prior_rep, delta_rep], dim=-1)
        expert_predictions = [expert(expert_protein_input, expert_drug_input) for expert in self.experts]
        expert_tensor = torch.stack(expert_predictions, dim=1)
        final_scores = torch.sum(expert_tensor * gate_probs, dim=-1)
        return final_scores, gate_probs, expert_tensor


class BayesianMultiplexRouter(PyroModule):
    """
    Truncated stick-breaking DPMM router with deterministic experts.

    - Generative model observes concatenated context [z_t, v_prior, delta_mean, trust_vector].
    - Routing is computed once at protein-level; broadcast only for expert score aggregation.
    """

    def __init__(
        self,
        protein_dim,
        drug_dim,
        trust_dim=5,
        max_experts=16,
        hidden_dim=128,
        top_k=2,
        dp_concentration=1.0,
        obs_scale=1.0,
    ):
        super().__init__()
        self.protein_dim = protein_dim
        self.drug_dim = drug_dim
        self.trust_dim = trust_dim
        self.max_experts = max_experts
        self.top_k = top_k
        self.dp_concentration = dp_concentration

        self.router_input_dim = (2 * protein_dim) + drug_dim + trust_dim
        self.router_net = PyroModule[nn.Sequential](
            nn.Linear(self.router_input_dim, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_experts),
        )
        self.experts = PyroModule[nn.ModuleList]([
            ExpertScorer(protein_dim, drug_dim) for _ in range(max_experts)
        ])

        # Generative emission parameters for concatenated observations.
        self.component_loc = PyroParam(torch.zeros(max_experts, self.router_input_dim))
        self.component_scale = PyroParam(
            torch.full((max_experts, self.router_input_dim), obs_scale),
            constraint=constraints.positive,
        )

        # Variational parameters for q(beta).
        self.q_beta_a = PyroParam(torch.ones(max_experts - 1), constraint=constraints.positive)
        self.q_beta_b = PyroParam(torch.ones(max_experts - 1), constraint=constraints.positive)

    def _stick_breaking(self, beta_samples):
        prefix = torch.cumprod(1.0 - beta_samples + 1e-8, dim=-1)
        remaining = torch.cat([torch.ones_like(prefix[..., :1]), prefix[..., :-1]], dim=-1)
        weights = beta_samples * remaining
        tail = prefix[..., -1:]
        return torch.cat([weights, tail], dim=-1)

    def _masked_softmax(self, logits):
        k = min(self.top_k, logits.size(-1))
        if k < logits.size(-1):
            top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            sparse_logits.scatter_(dim=-1, index=top_idx, src=top_vals)
            logits = sparse_logits
        return F.softmax(logits, dim=-1)

    def _router_input(self, z_t, v_prior, delta_mean, trust_vector):
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        if v_prior.dim() == 1:
            v_prior = v_prior.unsqueeze(0)
        if delta_mean.dim() == 1:
            delta_mean = delta_mean.unsqueeze(0)
        if trust_vector.dim() == 1:
            trust_vector = trust_vector.unsqueeze(0)
        # router input: [B, 2*protein_dim + drug_dim + trust_dim]
        return torch.cat([z_t, v_prior, delta_mean, trust_vector], dim=-1)

    def model(self, z_t, v_prior, delta_mean, trust_vector):
        pyro.module("bayesian_router", self)
        obs_x = self._router_input(z_t, v_prior, delta_mean, trust_vector)

        alpha = obs_x.new_tensor(self.dp_concentration)
        beta = pyro.sample(
            "beta",
            dist.Beta(
                torch.ones(self.max_experts - 1, device=obs_x.device),
                alpha.expand(self.max_experts - 1),
            ).to_event(1),
        )
        stick_weights = self._stick_breaking(beta)

        with pyro.plate("batch", obs_x.size(0)):
            z = pyro.sample("z", dist.Categorical(stick_weights))
            pyro.sample(
                "context_obs",
                dist.Normal(self.component_loc[z], self.component_scale[z]).to_event(1),
                obs=obs_x,
            )

    def guide(self, z_t, v_prior, delta_mean, trust_vector):
        obs_x = self._router_input(z_t, v_prior, delta_mean, trust_vector)

        beta = pyro.sample("beta", dist.Beta(self.q_beta_a, self.q_beta_b).to_event(1))
        stick_weights = self._stick_breaking(beta)

        logits = self.router_net(obs_x)
        q_z = self._masked_softmax(logits)
        q_z = q_z * stick_weights.unsqueeze(0)
        q_z = q_z / (q_z.sum(dim=-1, keepdim=True) + 1e-8)

        with pyro.plate("batch", obs_x.size(0)):
            pyro.sample("z", dist.Categorical(q_z))

    def expected_stick_weights(self):
        beta_mean = self.q_beta_a / (self.q_beta_a + self.q_beta_b + 1e-8)
        return self._stick_breaking(beta_mean)

    def route_probs(self, z_t, v_prior, delta_mean, trust_vector):
        """Protein-level routing probabilities. Returns shape [1, K]."""
        obs_x = self._router_input(z_t, v_prior, delta_mean, trust_vector)[:1]
        logits = self.router_net(obs_x)
        local_probs = self._masked_softmax(logits)

        stick_weights = self.expected_stick_weights().unsqueeze(0)
        probs = local_probs * stick_weights
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return probs

    def forward(self, z_refined, protein_raw_features, v_prior, delta_mean, query_drug_features, trust_vector):
        if z_refined.dim() == 1:
            z_refined = z_refined.unsqueeze(0)

        protein_gate_probs = self.route_probs(protein_raw_features, v_prior, delta_mean, trust_vector)

        num_queries = query_drug_features.size(0)
        prior_rep = v_prior.unsqueeze(0).expand(num_queries, -1)
        delta_rep = delta_mean.unsqueeze(0).expand(num_queries, -1)
        expert_protein_input = torch.cat([z_refined.expand(num_queries, -1), prior_rep, delta_rep], dim=-1)
        expert_drug_input = torch.cat([query_drug_features, prior_rep, delta_rep], dim=-1)
        expert_predictions = [expert(expert_protein_input, expert_drug_input) for expert in self.experts]
        expert_tensor = torch.stack(expert_predictions, dim=1)

        # [N, K] * [1, K] -> [N, K], keeping a single protein-level gate decision.
        final_scores = torch.sum(expert_tensor * protein_gate_probs, dim=-1)
        return final_scores, protein_gate_probs, expert_tensor
