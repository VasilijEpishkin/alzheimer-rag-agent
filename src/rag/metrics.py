# Metric definitions for evaluation
def precision_at_k(actual, predicted, k=5):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if len(pred_set) == 0: return 0.0
    return len(act_set & pred_set) / k

def ndcg_at_k(actual, predicted, k=5):
    # Simplified NDCG Implementation
    pass
