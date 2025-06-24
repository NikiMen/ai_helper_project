import numpy as np

def detect_column_type(column):
    try:
        np.array(column, dtype=float)
        return "numeric"
    except:
        return "categorical"

def validate_numeric(value):
    try:
        float(value)
        return True
    except:
        return False