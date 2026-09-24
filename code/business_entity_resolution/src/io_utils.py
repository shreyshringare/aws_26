"""I/O helpers: fast TSV loading, ground-truth parsing and submission writing."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv


def load_tsv(path):
    """Load a challenge TSV into pandas with plain-string columns and no NaNs.

    Uses the pyarrow reader (quoting disabled, since fields may contain quotes)
    which is ~10x faster than pandas' C parser on the multi-GB source files.
    """
    t = pacsv.read_csv(
        str(path),
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(strings_can_be_null=False),
    )
    df = t.to_pandas()
    for c in df.columns:
        df[c] = df[c].astype(str)
    return df


def load_split(data_dir, split):
    """Load the three source files of a split ('train' or 'test').

    Returns (s1, pool) where pool is S2 and S3 concatenated.
    """
    d = Path(data_dir) / split
    s1 = load_tsv(d / f"{split}_source1.tsv")
    s2 = load_tsv(d / f"{split}_source2.tsv")
    s3 = load_tsv(d / f"{split}_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True)
    return s1, pool


def load_ground_truth(data_dir):
    """Return ground truth as a dict {s1_id: set(matched ids)} (empty set allowed)."""
    gt = load_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    return {
        s: set(m.split(",")) if m else set()
        for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
    }


def gt_pairs(gt):
    """Explode a ground-truth dict into a DataFrame of (s1_id, cand_id) positive pairs."""
    rows = [(s, m) for s, ms in gt.items() for m in ms]
    return pd.DataFrame(rows, columns=["s1_id", "cand_id"])


def write_lists(path, s1_ids, lists, header):
    """Write a submission-style TSV: one row per S1 id, comma-joined ID list.

    `lists` maps s1_id -> iterable of ids; ids are de-duplicated preserving order.
    Rows follow the order of `s1_ids`.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s in s1_ids:
            ids = lists.get(s, ())
            ids = list(dict.fromkeys(ids))
            f.write(f"{s}\t{','.join(ids)}\n")


def pairs_to_lists(pairs, s1_col="s1_id", cand_col="cand_id"):
    """Group a pair DataFrame into {s1_id: [cand ids]}."""
    if len(pairs) == 0:
        return {}
    g = pairs.groupby(s1_col, sort=False)[cand_col].agg(list)
    return g.to_dict()


def id_prefix_is_pool(ids):
    """Boolean mask: which ids are S2-/S3- ids (defensive check before writing)."""
    ids = np.asarray(ids, dtype=str)
    return np.char.startswith(ids, "S2-") | np.char.startswith(ids, "S3-")
