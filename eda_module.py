import pandas as pd

def basic_eda(df):
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'null_counts': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'describe': df.describe().to_dict()
    }
    return summary
