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