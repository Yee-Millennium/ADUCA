import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

def read_libsvm_into_yXT_sparse(filepath: str, dim_dataset: int, num_dataset: int) -> csc_matrix:
    """
    Read a LibSVM binary dataset into a sparse data matrix transposed (i.e., y * X^T).

    Args:
        filepath (str): Path to the LibSVM-formatted file.
        dim_dataset (int): Number of features (dimensions).
        num_dataset (int): Number of examples.

    Returns:
        scipy.sparse.csc_matrix: Sparse matrix in CSC format with shape (dim_dataset, num_dataset).
    """
    # Initialize lists to store the sparse matrix data
    train_indices = []
    feature_indices = []
    values = []

    # Open and read the file line by line
    with open(filepath, 'r') as file:
        for line_num, line in enumerate(file, start=0):
            # Strip whitespace and split the line by spaces
            split_line = line.strip().split()
            
            # Skip empty lines
            if not split_line:
                continue
            
            # The first element is the label
            label_str = split_line[0]
            try:
                label = int(label_str)
            except ValueError:
                raise ValueError(f"Invalid label '{label_str}' on line {line_num + 1}")
            
            # Process each feature-value pair
            for item in split_line[1:]:
                try:
                    index_str, value_str = item.split(':')
                    # Convert to 0-based index
                    index = int(index_str) - 1
                    if index < 0 or index >= dim_dataset:
                        raise IndexError(f"Feature index {index + 1} out of bounds on line {line_num + 1}")
                    value = label * float(value_str)
                except ValueError:
                    raise ValueError(f"Invalid feature-value pair '{item}' on line {line_num + 1}")
                except IndexError as ie:
                    raise IndexError(str(ie))
                
                # Append the data to the lists
                train_indices.append(line_num)
                feature_indices.append(index)
                values.append(value)

    # Create the sparse matrix in COO format and convert to CSC
    A_T = coo_matrix((values, (feature_indices, train_indices)), shape=(dim_dataset, num_dataset)).tocsc()
    return A_T