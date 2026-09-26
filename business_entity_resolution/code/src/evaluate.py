import pandas as pd

def macro_f_0_5(gt_dict, pred_dict):
    scores = []
    for s1, gt_matches in gt_dict.items():
        pred_matches = pred_dict.get(s1, set())
        
        if len(gt_matches) == 0:
            if len(pred_matches) == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            tp = len(gt_matches.intersection(pred_matches))
            fp = len(pred_matches - gt_matches)
            fn = len(gt_matches - pred_matches)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision == 0 and recall == 0:
                f_0_5 = 0.0
            else:
                f_0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)
            scores.append(f_0_5)
            
    return sum(scores) / len(scores) if scores else 0.0
