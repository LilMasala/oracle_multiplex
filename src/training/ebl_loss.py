import torch
import torch.nn as nn
import torch.nn.functional as F

class EBLLoss(nn.Module):
    def __init__(self, ebl_alpha=0.5, temperature=0.05, eps=0.15, rank_weight=1.0): 
        super().__init__()
        self.ebl_alpha = ebl_alpha
        self.temperature = temperature 
        self.eps = eps 
        # Bumped rank_weight slightly because ListNet gradients are smaller than margin loss
        self.rank_weight = rank_weight 

    def _compute_listnet_loss(self, preds, labels):
        """
        Calculates ListNet (Top-K focused ranking) loss.
        Converts scores to probability distributions and calculates Cross-Entropy.
        """
        if len(preds) < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
        # 1. Convert True Labels (pIC50) into a target probability distribution
        # We use a slight temperature scaling on labels to sharpen the top hits
        target_probs = F.softmax(labels / 0.5, dim=0)
        
        # 2. Convert Model Predictions into a predicted probability distribution
        pred_probs = F.log_softmax(preds, dim=0)
        
        # 3. Calculate Cross-Entropy Loss
        # The model is heavily penalized if it doesn't assign high probability 
        # to the drugs that have high actual pIC50 values.
        listnet_loss = -torch.sum(target_probs * pred_probs)
        
        return listnet_loss

    def forward(self, final_preds, true_labels, gate_probs, expert_tensor):
        mse_per_expert = (expert_tensor - true_labels.unsqueeze(1)) ** 2
        mean_mse = mse_per_expert.mean(dim=0, keepdim=True) 

        with torch.no_grad():
            raw_target_probs = F.softmax(-mean_mse / self.temperature, dim=-1)
            K = expert_tensor.size(1)
            target_gate_probs = raw_target_probs * (1.0 - self.eps) + (self.eps / K)

        # 2. Swap in the ListNet Top-K Ranking Loss!
        ranking_loss = self._compute_listnet_loss(final_preds, true_labels)

        # 3. Gate Penalty 
        gate_loss = -torch.sum(target_gate_probs * torch.log(gate_probs + 1e-12), dim=-1).mean()

        # 4. Decoupled Expert Loss
        expert_loss = torch.sum(mean_mse * target_gate_probs.detach(), dim=-1).mean()

        # 5. Total Loss 
        total_loss = expert_loss + (self.ebl_alpha * gate_loss) + (self.rank_weight * ranking_loss)

        return total_loss, expert_loss.item(), gate_loss.item()