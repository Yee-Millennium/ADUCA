import numpy as np
from src.problems.utils.data import Data

def libsvm_parser(path, n, d):
    features = np.zeros((n, d))
    values = np.zeros(n)

    with open(path, 'r') as f:
        data_str = f.readlines()

    for i in range(n):
        feature_value_str = data_str[i].strip().split(" ")
        values[i] = float(feature_value_str[0])

        for fv in feature_value_str[2:]:
            idx, feature = fv.split(':')
            features[i, int(idx) - 1] = float(feature)  # Convert to 0-based index for Python

    return Data(features, values)