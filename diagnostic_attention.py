"""
Diagnostic: Overfit TNP on 2 proteins x 5 drugs with hardcoded affinities.

Prints the exact attention logits and weight matrix at every step so we can
prove whether query tokens are attending to context tokens or if something is
silently blocking information flow.

Hypothesis under test
---------------------
GraphBiasedMHA computes:
    col_bias = log_ppr_alpha * log(ctx_ppr)

log_ppr_alpha is initialized to +0.1.
ctx_ppr values come from PPR scores which are in (0, 1), so log(ctx_ppr) < 0.
Therefore col_bias < 0 for ALL context tokens.
The query self-attention column (pad=0) gets ZERO bias.
After softmax each query token is pushed toward attending to itself, not context.

If the hypothesis is correct we will see:
    attn_weight[qry_row, ctx_cols].sum() << attn_weight[qry_row, self_col]

Run:  python diagnostic_attention.py
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── tiny dimensions so everything prints readable ──────────────────────────
PROT_DIM  = 8
DRUG_DIM  = 4
TOKEN_DIM = 16
NHEAD     = 2
NLAYERS   = 1
N_CTX     = 5   # protein-0 bindings (context)
N_QRY     = 5   # protein-1 drugs    (query)
N_STEPS   = 300

# hardcoded affinities protein-0 (context) and protein-1 (query / ground truth)
CTX_AFF = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0])
QRY_AFF = torch.tensor([5.5, 6.5, 7.5, 8.5, 9.5])   # slightly offset


# ── verbatim copy of GraphBiasedMHA with full diagnostics exposed ──────────
class InstrumentedMHA(nn.Module):
    """Identical to GraphBiasedMHA but stores all intermediate tensors."""

    def __init__(self, token_dim, nhead, dropout=0.0):
        super().__init__()
        assert token_dim % nhead == 0
        self.nhead    = nhead
        self.head_dim = token_dim // nhead
        self.scale    = self.head_dim ** -0.5
        self.dropout  = dropout

        self.in_proj  = nn.Linear(token_dim, 3 * token_dim, bias=True)
        self.out_proj = nn.Linear(token_dim, token_dim, bias=True)

        self.log_ppr_alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.trust_scale   = nn.Parameter(torch.ones(1) * 0.1)

        # ── diagnostic storage ─────────────────────────────────────────────
        self.diag = {}

    def forward(self, x, n_ctx, ctx_ppr, ctx_trust, attn_mask=None):
        B, N, D = x.shape
        H, d    = self.nhead, self.head_dim

        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, d).transpose(1, 2)
        k = k.view(B, N, H, d).transpose(1, 2)
        v = v.view(B, N, H, d).transpose(1, 2)

        # V gate
        gate_vals = None
        if n_ctx > 0:
            gate_vals = torch.sigmoid(self.trust_scale * ctx_trust)
            ones      = torch.ones(N - n_ctx, device=x.device)
            gate_full = torch.cat([gate_vals, ones], dim=0)
            v = v * gate_full.view(1, 1, N, 1)

        raw_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [1,H,N,N]

        # PPR bias — THIS IS WHAT WE ARE DIAGNOSING
        ppr_bias_vals = None
        if n_ctx > 0:
            ppr_bias_vals = self.log_ppr_alpha * torch.log(ctx_ppr.clamp(min=1e-8))
            pad            = torch.zeros(N - n_ctx, device=x.device)
            col_bias_full  = torch.cat([ppr_bias_vals, pad], dim=0)
            biased_logits  = raw_logits + col_bias_full.view(1, 1, 1, N)
        else:
            biased_logits = raw_logits

        logits_before_mask = biased_logits.clone()

        if attn_mask is not None:
            biased_logits = biased_logits.masked_fill(
                attn_mask.view(1, 1, N, N), float("-inf")
            )

        weights = F.softmax(biased_logits, dim=-1)

        # ── store everything ───────────────────────────────────────────────
        self.diag["raw_logits"]        = raw_logits.detach()
        self.diag["ppr_bias_vals"]     = ppr_bias_vals.detach() if ppr_bias_vals is not None else None
        self.diag["logits_pre_mask"]   = logits_before_mask.detach()
        self.diag["logits_post_mask"]  = biased_logits.detach()
        self.diag["attn_weights"]      = weights.detach()
        self.diag["gate_vals"]         = gate_vals.detach() if gate_vals is not None else None
        self.diag["log_ppr_alpha"]     = self.log_ppr_alpha.item()
        self.diag["trust_scale"]       = self.trust_scale.item()

        weights_dp = F.dropout(weights, p=self.dropout, training=self.training)
        out = torch.matmul(weights_dp, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class InstrumentedTNP(nn.Module):
    """Minimal TNP with InstrumentedMHA. Mirrors ProteinLigandTNP exactly."""

    def __init__(self, prot_dim, drug_dim, token_dim, nhead, nlayers):
        super().__init__()
        self.token_dim = token_dim

        self.ctx_protein_proj  = nn.Linear(prot_dim,     token_dim // 2)
        self.ctx_drug_proj     = nn.Linear(drug_dim,     token_dim // 2)
        self.ctx_affinity_proj = nn.Linear(1,            token_dim // 4)
        self.ctx_fusion = nn.Linear(
            token_dim // 2 + token_dim // 2 + token_dim // 4, token_dim
        )

        self.qry_protein_proj = nn.Linear(prot_dim,  token_dim // 2)
        self.qry_drug_proj    = nn.Linear(drug_dim,  token_dim // 2)
        self.qry_fusion = nn.Linear(token_dim // 2 + token_dim // 2, token_dim)

        self.type_embed = nn.Embedding(2, token_dim)

        self.layers = nn.ModuleList([
            self._make_layer(token_dim, nhead) for _ in range(nlayers)
        ])
        self.final_norm = nn.LayerNorm(token_dim)
        self.output_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Linear(token_dim // 2, 2),
        )

        # Unit 3 params (density gating)
        self.density_proj    = nn.Linear(1, token_dim)
        self.cold_start_bias = nn.Parameter(torch.zeros(1))
        self.centroid_alpha  = nn.Parameter(torch.tensor(-2.0))

    def _make_layer(self, token_dim, nhead):
        return nn.ModuleDict({
            "norm1": nn.LayerNorm(token_dim),
            "norm2": nn.LayerNorm(token_dim),
            "attn":  InstrumentedMHA(token_dim, nhead, dropout=0.0),
            "ff":    nn.Sequential(
                nn.Linear(token_dim, token_dim * 2),
                nn.GELU(),
                nn.Linear(token_dim * 2, token_dim),
            ),
        })

    def _layer_forward(self, layer, x, n_ctx, ctx_ppr, ctx_trust, mask):
        h = layer["norm1"](x)
        x = x + layer["attn"](h, n_ctx, ctx_ppr, ctx_trust, mask)
        x = x + layer["ff"](layer["norm2"](x))
        return x

    def _build_mask(self, n_ctx, n_qry, device):
        N    = n_ctx + n_qry
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        if n_qry > 0:
            mask[n_ctx:, n_ctx:] = True
            idx = torch.arange(n_ctx, N, device=device)
            mask[idx, idx] = False
            mask[:n_ctx, n_ctx:] = True
        return mask

    def forward(self, ctx_p, ctx_d, ctx_a, qry_p, qry_d, ctx_ppr, ctx_trust):
        device = qry_p.device
        n_ctx  = ctx_p.size(0)
        n_qry  = qry_p.size(0)
        density = math.tanh(n_ctx / 32.0)

        p = self.qry_protein_proj(qry_p)
        d = self.qry_drug_proj(qry_d)
        qry_tokens = self.qry_fusion(torch.cat([p, d], dim=-1))
        qry_tokens = qry_tokens + self.type_embed(
            torch.ones(n_qry, dtype=torch.long, device=device)
        )
        density_t  = torch.tensor([[density]], dtype=torch.float32, device=device)
        qry_tokens = qry_tokens + self.density_proj(density_t)

        if n_ctx > 0:
            p2 = self.ctx_protein_proj(ctx_p)
            d2 = self.ctx_drug_proj(ctx_d)
            a2 = self.ctx_affinity_proj(ctx_a)
            ctx_tokens = self.ctx_fusion(torch.cat([p2, d2, a2], dim=-1))
            ctx_tokens = ctx_tokens + self.type_embed(
                torch.zeros(n_ctx, dtype=torch.long, device=device)
            )
            tokens = torch.cat([ctx_tokens, qry_tokens], dim=0).unsqueeze(0)
            mask   = self._build_mask(n_ctx, n_qry, device)
        else:
            tokens    = qry_tokens.unsqueeze(0)
            mask      = None
            ctx_ppr   = torch.zeros(0, device=device)
            ctx_trust = torch.zeros(0, device=device)

        for layer in self.layers:
            tokens = self._layer_forward(layer, tokens, n_ctx, ctx_ppr, ctx_trust, mask)

        tokens  = self.final_norm(tokens).squeeze(0)
        qry_out = tokens[n_ctx:]
        pred    = self.output_head(qry_out)
        mu        = pred[:, 0]
        log_sigma = pred[:, 1]
        log_sigma = log_sigma + self.cold_start_bias * (1.0 - density)
        sigma     = F.softplus(log_sigma) + 1e-4
        return mu, sigma

    @property
    def last_attn_diag(self):
        return self.layers[-1]["attn"].diag


# ── TNPLoss (verbatim) ─────────────────────────────────────────────────────
class TNPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_nll     = nn.Parameter(torch.zeros(1))
        self.log_var_listnet = nn.Parameter(torch.zeros(1))
        self.log_var_lambda  = nn.Parameter(torch.zeros(1))

    def forward(self, mu, sigma, labels):
        nll = (0.5 * math.log(2 * math.pi)
               + torch.log(sigma)
               + (labels - mu) ** 2 / (2 * sigma ** 2))
        nll_loss = nll.mean()

        T = 2.0
        target_probs   = F.softmax(labels / T, dim=0)
        log_pred_probs = F.log_softmax(mu / T, dim=0)
        listnet_loss   = -(target_probs * log_pred_probs).sum()

        pairs = (labels.unsqueeze(0) - labels.unsqueeze(1)) > 0.5
        if pairs.any():
            pred_diff   = mu.unsqueeze(0) - mu.unsqueeze(1)
            lambda_loss = F.softplus(-pred_diff[pairs]).mean()
        else:
            lambda_loss = torch.tensor(0.0)

        w_nll     = torch.exp(-self.log_var_nll)
        w_listnet = torch.exp(-self.log_var_listnet)
        w_lambda  = torch.exp(-self.log_var_lambda)
        total = (
            (w_nll * nll_loss + 0.5 * self.log_var_nll)
            + (w_listnet * listnet_loss + 0.5 * self.log_var_listnet)
            + (w_lambda * lambda_loss + 0.5 * self.log_var_lambda)
        )
        return total, {
            "nll": nll_loss.item(), "listnet": listnet_loss.item(),
            "lambda": lambda_loss.item(),
            "w_nll": w_nll.item(), "w_listnet": w_listnet.item(),
            "w_lambda": w_lambda.item(),
        }


# ── toy data ───────────────────────────────────────────────────────────────
torch.manual_seed(0)

prot_0 = torch.randn(PROT_DIM)        # context protein
prot_1 = torch.randn(PROT_DIM)        # query protein
drugs  = torch.randn(N_CTX, DRUG_DIM) # same 5 drugs for both

ctx_protein  = prot_0.unsqueeze(0).expand(N_CTX, -1)   # [5, PROT_DIM]
ctx_drug     = drugs                                    # [5, DRUG_DIM]
ctx_affinity = CTX_AFF.unsqueeze(1)                     # [5, 1]
ctx_ppr      = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.05])# typical PPR scores
ctx_trust    = torch.ones(N_CTX)                        # trust=1

qry_protein  = prot_1.unsqueeze(0).expand(N_QRY, -1)   # [5, PROT_DIM]
qry_drug     = drugs                                    # same 5 drugs
labels       = QRY_AFF                                  # [5]


# ── build model + loss ─────────────────────────────────────────────────────
model   = InstrumentedTNP(PROT_DIM, DRUG_DIM, TOKEN_DIM, NHEAD, NLAYERS)
loss_fn = TNPLoss()
optim   = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()), lr=3e-3
)


def print_banner(title):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def print_attention_diagnostics(step, mu, sigma, labels, diag, loss_meta):
    n_ctx = N_CTX
    n_qry = N_QRY
    N     = n_ctx + n_qry

    print_banner(f"STEP {step:04d}")

    # ── 1. PPR bias: the smoking gun ──────────────────────────────────────
    alpha = diag["log_ppr_alpha"]
    ppr_b = diag["ppr_bias_vals"]
    print(f"\n[PPR GRAPH BIAS]  log_ppr_alpha = {alpha:+.4f}")
    print(f"  ctx_ppr values : {ctx_ppr.tolist()}")
    if ppr_b is not None:
        print(f"  log(ctx_ppr)   : {torch.log(ctx_ppr.clamp(1e-8)).tolist()}")
        print(f"  alpha*log(ppr) : {ppr_b.tolist()}")
        sign = "NEGATIVE" if (ppr_b < 0).all() else "POSITIVE"
        print(f"  → bias is {sign} for all context tokens")
        print(f"  → query SELF column gets bias = 0.0")
        print(f"  VERDICT: context tokens are {'PENALISED' if sign=='NEGATIVE' else 'BOOSTED'} vs self-attn")

    # ── 2. V gate ─────────────────────────────────────────────────────────
    ts   = diag["trust_scale"]
    gate = diag["gate_vals"]
    print(f"\n[V GATE]  trust_scale = {ts:+.4f}")
    if gate is not None:
        print(f"  sigmoid(ts * trust): {gate.tolist()}")
        print(f"  → context V values scaled by mean {gate.mean():.4f}")

    # ── 3. Raw Q·K logits (head 0, query rows only) ───────────────────────
    raw  = diag["raw_logits"][0, 0]            # [N, N], head-0
    qry_raw = raw[n_ctx:, :]                    # [n_qry, N]
    print(f"\n[RAW Q·K LOGITS head-0, query rows only, shape {list(qry_raw.shape)}]")
    print(f"  Columns 0..{n_ctx-1} = context   |   Column {n_ctx}..{N-1} = self+other-qry")
    for i in range(n_qry):
        ctx_part  = qry_raw[i, :n_ctx].tolist()
        self_val  = qry_raw[i, n_ctx + i].item()
        ctx_str   = " ".join(f"{v:+6.3f}" for v in ctx_part)
        print(f"  qry[{i}]  ctx=[{ctx_str}]  self={self_val:+6.3f}")

    # ── 4. Logits after PPR bias, before mask ─────────────────────────────
    pre  = diag["logits_pre_mask"][0, 0]
    qpre = pre[n_ctx:, :]
    print(f"\n[LOGITS AFTER PPR BIAS, BEFORE MASK, head-0]")
    for i in range(n_qry):
        ctx_part = qpre[i, :n_ctx].tolist()
        self_val = qpre[i, n_ctx + i].item()
        ctx_str  = " ".join(f"{v:+6.3f}" for v in ctx_part)
        print(f"  qry[{i}]  ctx=[{ctx_str}]  self={self_val:+6.3f}")

    # ── 5. Final attention weights (after mask + softmax) ─────────────────
    W    = diag["attn_weights"][0, 0]     # [N, N]
    qW   = W[n_ctx:, :]                   # [n_qry, N]
    print(f"\n[ATTENTION WEIGHTS after softmax, head-0 — query rows]")
    print(f"  (cols 0-{n_ctx-1} = context, col {n_ctx}-{N-1} = query self/peer)")
    ctx_mass_total = 0.0
    for i in range(n_qry):
        ctx_mass  = qW[i, :n_ctx].sum().item()
        self_mass = qW[i, n_ctx + i].item()
        ctx_str   = " ".join(f"{v:.4f}" for v in qW[i, :n_ctx].tolist())
        ctx_mass_total += ctx_mass
        print(f"  qry[{i}]  ctx=[{ctx_str}]  ctx_sum={ctx_mass:.4f}  self={self_mass:.4f}")
    avg_ctx_mass = ctx_mass_total / n_qry
    verdict = "ATTENDING TO CONTEXT ✓" if avg_ctx_mass > 0.3 else "IGNORING CONTEXT ✗  ← LOBOTOMISED"
    print(f"\n  avg ctx attention mass = {avg_ctx_mass:.4f}  →  {verdict}")

    # ── 6. Mask sanity check ──────────────────────────────────────────────
    # Verify: mask[qry_i, ctx_j] must be False (not blocked)
    # We test this by checking if ctx columns are -inf after masking
    post = diag["logits_post_mask"][0, 0]
    qpost = post[n_ctx:, :n_ctx]   # query rows, context cols
    has_neginf = (qpost == float("-inf")).any().item()
    print(f"\n[MASK CHECK] any -inf in query→context logits: {has_neginf}")
    print(f"  (True = mask is INCORRECTLY blocking query→context attention)")

    # ── 7. Predictions vs targets ─────────────────────────────────────────
    print(f"\n[PREDICTIONS vs TARGETS]")
    for i in range(n_qry):
        print(f"  drug[{i}]  mu={mu[i].item():+7.3f}  sigma={sigma[i].item():.3f}"
              f"  target={labels[i].item():.1f}  err={mu[i].item()-labels[i].item():+.3f}")

    # ── 8. Homoscedastic loss weights ─────────────────────────────────────
    print(f"\n[HOMOSCEDASTIC WEIGHTS]")
    print(f"  w_nll={loss_meta['w_nll']:.4f}  w_listnet={loss_meta['w_listnet']:.4f}"
          f"  w_lambda={loss_meta['w_lambda']:.4f}")
    print(f"  (weights stuck at 1.0 → log_var not moving → gradients may be dead)")

    # ── 9. Gradient norms ─────────────────────────────────────────────────
    print(f"\n[GRADIENT NORMS after backward]")
    for name, p in list(model.named_parameters()) + list(loss_fn.named_parameters()):
        if p.grad is not None:
            gn = p.grad.norm().item()
            flag = " ← DEAD" if gn < 1e-6 else ""
            print(f"  {name:45s}  |grad|={gn:.2e}{flag}")
        else:
            print(f"  {name:45s}  grad=None")


# ── training loop ──────────────────────────────────────────────────────────
PRINT_STEPS = {0, 1, 2, 10, 50, 100, 200, N_STEPS - 1}

print_banner("TNP ATTENTION DIAGNOSTIC — 2 proteins, 5 drugs, hardcoded affinities")
print(f"Context affinities (protein-0): {CTX_AFF.tolist()}")
print(f"Query   affinities (protein-1): {QRY_AFF.tolist()}")
print(f"ctx_ppr values: {ctx_ppr.tolist()}")
print(f"log_ppr_alpha init: {model.layers[0]['attn'].log_ppr_alpha.item():.4f}")
print(f"\nExpected at cold-start: alpha=0.1, log(ppr)≈-1 to -3")
print(f"  → PPR bias = {0.1 * math.log(0.4):.4f} to {0.1 * math.log(0.05):.4f} per ctx token")
print(f"  → self-attn column bias = 0.0")
print(f"  → model is PUSHED AWAY from context at init\n")

for step in range(N_STEPS):
    model.train()
    loss_fn.train()
    optim.zero_grad()

    mu, sigma = model(ctx_protein, ctx_drug, ctx_affinity,
                      qry_protein, qry_drug, ctx_ppr, ctx_trust)
    total, meta = loss_fn(mu, sigma, labels)
    total.backward()

    if step in PRINT_STEPS:
        print_attention_diagnostics(step, mu.detach(), sigma.detach(), labels,
                                    model.last_attn_diag, meta)

    optim.step()

print_banner("FINAL VERDICT")
mu_final, _ = model(ctx_protein, ctx_drug, ctx_affinity,
                    qry_protein, qry_drug, ctx_ppr, ctx_trust)
from src.training.metrics import calculate_ci
ci = calculate_ci(labels, mu_final.detach())
print(f"\n  CI after {N_STEPS} steps on 5-point overfit task: {ci:.4f}")
print(f"  (perfect overfit = 1.0, random = 0.5, ours should be 1.0 if attention works)")
if ci < 0.8:
    print("\n  CONCLUSION: model CANNOT overfit 5 data points.")
    print("  Context information is NOT reaching query tokens.")
    print("  Root cause: PPR graph bias is NEGATIVE for all context tokens,")
    print("  making self-attention dominate over cross-attention to context.")
    print("\n  FIX: negate the bias sign so high-PPR tokens get POSITIVE logit bonus.")
    print("  Change:  col_bias = log_ppr_alpha * log(ctx_ppr)")
    print("  To:      col_bias = -log_ppr_alpha * log(ctx_ppr)   # or flip alpha init")
    print("  Or:      col_bias = log_ppr_alpha * (ctx_ppr - ctx_ppr.mean())")
else:
    print("\n  CONCLUSION: attention IS working — look elsewhere for CI=0.54.")
