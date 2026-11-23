# src.metric
import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # encoding into one idx, count by bincount
    idx = y_true * num_classes + y_pred
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)

def metrics_from_cm(cm):
    """
    Compute metrics from confusion matrix.
    cm = [[TN, FP],
          [FN, TP]]
    """
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        raise ValueError("Only supporting confusion matrix 2x2 (binary).")

    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }