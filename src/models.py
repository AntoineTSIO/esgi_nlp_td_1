from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def create_model(model_type="rf"):
    """Create a model based on the specified type."""
    if model_type == "rf":
        return RandomForestClassifier()
    elif model_type == "lr":
        return LogisticRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
