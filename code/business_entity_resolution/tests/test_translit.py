"""Tests for the learned transliteration table."""
from src.translit import fit_table, load_table, save_table


def test_fit_table_majority_alignment(tmp_path):
    s1 = ["Ram Marketing Private Limited"] * 3 + ["Shyam Traders Private Limited"] * 3
    pool = ["राम मार्केटिंग प्राइवेट लिमिटेड"] * 3 + ["श्याम ट्रेडर्स प्राइवेट लिमिटेड"] * 3
    table = fit_table(s1, pool, min_support=3, min_share=0.6)
    assert table["प्राइवेट"] == "private" and table["लिमिटेड"] == "limited"
    assert table["राम"] == "ram"
    p = tmp_path / "t.tsv"
    save_table(table, p)
    assert load_table(p) == table


def test_fit_table_skips_unequal_lengths():
    assert fit_table(["A B"] * 5, ["राम"] * 5) == {}
