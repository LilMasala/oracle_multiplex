import torch
import numpy as np

def calculate_ci(y_true, y_pred):
    """
    Concordance Index (CI): The probability that the model ranks a pair 
    of drugs in the correct order of affinity.
    0.5 = Random Guessing | 1.0 = Perfect Ranking
    """
    if len(y_true) < 2:
        return 0.5
        
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    ind = np.argsort(y_true)
    y_true = y_true[ind]
    y_pred = y_pred[ind]
    
    i = len(y_true) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y_true[i] > y_true[j]:
                z += 1.0
                if y_pred[i] > y_pred[j]:
                    S += 1.0
                elif y_pred[i] == y_pred[j]:
                    S += 0.5
            j -= 1
        i -= 1
        j = i - 1

    return S / z if z > 0 else 0.5


def calculate_ef_at_k(y_true, y_pred, k=0.1):
    """
    Enrichment Factor at top k% (e.g., 0.1 for EF10).
    EF = (hits_in_top_k / n_top_k) / (total_hits / total_n)
    """
    if len(y_true) < 10: return 1.0 # Not enough data for meaningful EF
    
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    # Define what a "hit" is (top 10% of true binders in this episode)
    n_top_k = int(np.ceil(k * len(y_true)))
    threshold = np.partition(y_true, -n_top_k)[-n_top_k]
    is_hit = y_true >= threshold
    
    # Find indices of model's top k% predictions
    top_k_indices = np.argsort(y_pred)[-n_top_k:]
    
    # Calculate hits found in model's top k%
    hits_found = np.sum(is_hit[top_k_indices])
    total_hits = np.sum(is_hit)
    
    # EF formula
    ef = (hits_found / n_top_k) / (total_hits / len(y_true))
    return ef