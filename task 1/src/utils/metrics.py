from sklearn.metrics import precision_recall_fscore_support, classification_report
from typing import List

def calculate_ner_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> dict:
    """
    Flatten lists and calculate standard metrics for NER.
    Assumes inputs are lists of token-level labels for each sentence.
    """
    y_true_flat = [label for sentence in y_true for label in sentence]
    y_pred_flat = [label for sentence in y_pred for label in sentence]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average="weighted", zero_division=0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def print_classification_report(y_true: List[List[str]], y_pred: List[List[str]]) -> None:
    """Prints a detailed text report showing the main classification metrics."""
    y_true_flat = [label for sentence in y_true for label in sentence]
    y_pred_flat = [label for sentence in y_pred for label in sentence]
    
    print(classification_report(y_true_flat, y_pred_flat, zero_division=0))