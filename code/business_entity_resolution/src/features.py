"""Stage-1 pair features (full implementation in Task 11)."""
import numpy as np


def add_frequency_columns(s1n, pooln):
    """Add log pool-frequency of name_core to both frames (how common a business name is)."""
    vc = pooln["name_core"].value_counts()
    pooln["name_freq"] = np.log1p(pooln["name_core"].map(vc).to_numpy(np.float32))
    s1n["name_freq"] = np.log1p(s1n["name_core"].map(vc).fillna(0).to_numpy(np.float32))
