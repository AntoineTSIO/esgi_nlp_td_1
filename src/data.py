import pandas as pd


def make_dataset(filename):
    """Load the dataset from the specified CSV file."""
    return pd.read_csv(filename)
