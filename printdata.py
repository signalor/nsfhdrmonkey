import numpy as np;
from typing import Optional, Tuple
import os


def load_data(data_dir: str, monkey_name: str) -> Tuple[np.ndarray, ...]:
    """Load and split data for a specific monkey."""

    filepath = os.path.join(data_dir, f"train_data_{monkey_name}.npz")
    data = np.load(filepath)["arr_0"]

    # Shuffle with fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    data = data[indices]

    # Split: 80% train, 10% val, 10% test
    n = len(data)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"Data shapes for {monkey_name}:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Test: {test_data.shape}")

    return train_data, val_data, test_data



traindata, valdat, tests =load_data("./public_data", "beignet")
print(len(traindata[0][0]))
