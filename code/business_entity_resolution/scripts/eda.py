"""Exploratory data analysis for the entity resolution dataset.

Answers the questions that drive pipeline design: singleton rate, whether the
one-to-one assumption holds, cross-country matches, field missingness and
non-Latin script prevalence. Prints a sample of matched pairs for inspection.
"""
import sys
import time

import pandas as pd
import pyarrow.csv as pacsv

DATA = sys.argv[1] if len(sys.argv) > 1 else "../../dataset"


def load(path):
    """Load a TSV as a pandas frame with Arrow-backed string columns."""
    t = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(strings_can_be_null=False),
    )
    return t.to_pandas(types_mapper=pd.ArrowDtype)


def main():
    """Run all EDA checks and print results."""
    t0 = time.time()
    s1 = load(f"{DATA}/train/train_source1.tsv")
    s2 = load(f"{DATA}/train/train_source2.tsv")
    s3 = load(f"{DATA}/train/train_source3.tsv")
    gt = load(f"{DATA}/train/train_ground_truth.tsv")
    print(f"loaded in {time.time()-t0:.0f}s")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"\n== {name} rows={len(df):,}")
        print(df["country"].value_counts().to_string())
        for c in ["business_name", "business_address"]:
            col = df[c].astype(str)
            print(f"  {c}: empty={ (col.str.len()==0).mean():.4f} "
                  f"non_ascii={col.str.contains(r'[^\x00-\x7f]', regex=True).mean():.4f} "
                  f"mean_len={col.str.len().mean():.1f}")

    gt["matched_entity_ids"] = gt["matched_entity_ids"].astype(str)
    lists = gt["matched_entity_ids"].str.split(",")
    n = lists.map(lambda l: 0 if l == [""] else len(l))
    print(f"\n== GT rows={len(gt):,} S1 rows={len(s1):,}")
    print(f"singleton rate={ (n==0).mean():.4f}")
    print("match count dist:\n", n.value_counts().sort_index().head(15).to_string())

    pairs = gt.assign(m=lists).explode("m")
    pairs = pairs[pairs["m"] != ""][["source1_entity_id", "m"]]
    pairs.columns = ["s1", "m"]
    print(f"pairs={len(pairs):,}  S2 share={pairs['m'].str.startswith('S2').mean():.3f}")
    dup = pairs["m"].value_counts()
    print(f"one-to-one: S2/S3 ids linked to >1 S1 = {(dup>1).sum():,} of {len(dup):,}")
    n2 = pairs["m"].str.startswith("S2")
    per = pd.DataFrame({"s2": n2.groupby(pairs["s1"]).sum(), "s3": (~n2).groupby(pairs["s1"]).sum()})
    print("S2 per matched S1:\n", per["s2"].value_counts().sort_index().head(8).to_string())
    print("S3 per matched S1:\n", per["s3"].value_counts().sort_index().head(8).to_string())
    print(f"S2 records matched: {n2.sum()/len(s2):.3f}  S3 records matched: {(~n2).sum()/len(s3):.3f}")

    ss = pd.concat([s2, s3])[["entity_id", "business_name", "business_address", "country"]]
    ss.columns = ["m", "m_name", "m_addr", "m_country"]
    s1c = s1.rename(columns={"entity_id": "s1"})
    j = pairs.merge(s1c, on="s1").merge(ss, on="m")
    print(f"joined pairs={len(j):,}  cross-country={ (j['country'].astype(str)!=j['m_country'].astype(str)).mean():.4f}")
    print("singleton rate by country:")
    g = gt.merge(s1c[["s1", "country"]], left_on="source1_entity_id", right_on="s1")
    g["n"] = n.values
    print(g.groupby("country")["n"].agg(lambda x: (x == 0).mean()).to_string())
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    samp = j.sample(40, random_state=1)
    for _, r in samp.iterrows():
        print(f"\n[{r['country']}] {r['business_name']} | {r['business_address']}\n"
              f"  -> {r['m'][:2]} {r['m_name']} | {r['m_addr']}")


if __name__ == "__main__":
    main()
