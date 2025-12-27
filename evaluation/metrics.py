from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def get_metrics(y_true, y_pred):
    """
    y_true: shape (num_samples,) integer labels
    y_pred: shape (num_samples,) integer labels
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(report).transpose(), cm

