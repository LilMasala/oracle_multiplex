import torch
import torch.nn as nn
import torch.nn.functional as F

class EBLLoss(nn.Module):
    def __init__(self, ebl_alpha=0.1):
        super().__init__()
        # ebl_alpha controls how harshly we punish the Gate for bad routing.
        # Too high, and the Gate thrashes. Too low, and Experts don't specialize.
        self.ebl_alpha = ebl_alpha

    def forward(self, final_preds, true_labels, gate_probs, expert_tensor):
        """
        final_preds: [M] The weighted final prediction
        true_labels: [M] The actual pIC50 values
        gate_probs: [1, K] The dispatcher's routing probabilities
        expert_tensor: [M, K] The raw predictions from each of the K experts
        """
        # 1. Base Regression Loss (How wrong was the final answer?)
        base_loss = F.mse_loss(final_preds, true_labels)
        
        # 2. The Autopsy (Explanation-Based Learning)
        # We use torch.no_grad() because we don't want the experts to update 
        # based on this specific calculation; we are only grading the GATE here.
        with torch.no_grad():
            # Calculate squared error for each expert on each drug
            # expert_tensor is [M, K], true_labels.unsqueeze(-1) is [M, 1]
            expert_errors = (expert_tensor - true_labels.unsqueeze(-1)) ** 2  # [M, K]
            
            # We want the gate to pick the expert that is BEST on average for this whole batch
            # of support drugs for this specific protein.
            mean_expert_errors = expert_errors.mean(dim=0, keepdim=True) # [1, K]
            
            # Target distribution: Softmax over the negative errors
            # (Lower error = higher target probability)
            target_gate_probs = F.softmax(-mean_expert_errors, dim=-1) # [1, K]
            
        # 3. Gate Penalty (Cross Entropy/KL Divergence)
        # Force the gate's probabilities to match the target "Oracle" probabilities
        # Add epsilon to prevent log(0) NaN explosions
        gate_loss = -torch.sum(target_gate_probs * torch.log(gate_probs + 1e-12), dim=-1).mean()
        
        # 4. Total Loss
        total_loss = base_loss + (self.ebl_alpha * gate_loss)
        
        # Returning the individual components is great for logging to Weights & Biases or Tensorboard!
        return total_loss, base_loss.item(), gate_loss.item()