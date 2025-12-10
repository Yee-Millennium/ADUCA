import bz2
import gzip
from pathlib import Path

import numpy as np
from src.problems.utils.data import Data


def _open_libsvm(path):
    """
    Open a LIBSVM file that might be plain text, gzip, or bzip2.
    """
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(3)
    if magic.startswith(b"\x1f\x8b"):  # gzip magic
        return gzip.open(path, "rt", encoding="latin-1", errors="ignore")
    if magic.startswith(b"BZh"):  # bzip2 magic
        return bz2.open(path, "rt", encoding="latin-1", errors="ignore")
    return path.open("r", encoding="latin-1", errors="ignore")


def libsvm_parser(path, n, d):
    features = np.zeros((n, d))
    values = np.zeros(n)

    # Auto-handle compressed inputs and tolerate non-UTF8 bytes.
    with _open_libsvm(path) as f:
        data_str = f.readlines()

    for i in range(min(n, len(data_str))):
        parts = data_str[i].strip().split()
        if not parts:
            continue
        values[i] = float(parts[0])

        for fv in parts[1:]:
            idx, feature = fv.split(":")
            features[i, int(idx) - 1] = float(feature)  # Convert to 0-based index for Python

    return Data(features, values)
