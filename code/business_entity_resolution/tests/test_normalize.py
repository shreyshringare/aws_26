"""Tests for name/address normalization dictionaries and rules."""
import pandas as pd

from src.normalize import normalize_address, normalize_frame, normalize_name, phonetic, split_raw, to_ascii


def test_name_legal_split_and_dots():
    norm, core, alias, legal, first = normalize_name("Pvt. EFS Print Ventures Ltd.")
    assert core == "efs print ventures"
    assert legal == "pvt ltd" and first == 1
    assert normalize_name("Green & Auclair L.L.C.")[1:4] == ("green auclair", "", "llc")
    assert normalize_name("E.U.R.L. Maison")[3] == "eurl"


def test_name_junk_accents_and_web():
    assert normalize_name("-- Holloway Peak Inc Seafood")[1] == "holloway peak seafood"
    assert normalize_name("Norah Pletcher Báin Care")[0] == "norah pletcher bain care"
    assert normalize_name("wilfordhancock.com")[1] == "wilfordhancock"


def test_name_translit_table_used():
    table = {"प्राइवेट": "private", "लिमिटेड": "limited"}
    assert normalize_name("राम प्राइवेट लिमिटेड", table)[3] == "pvt ltd"


def test_name_alias_split_dba_aka_fka_formerly():
    _, core, alias, _, _ = normalize_name("Golden Gate Cafe dba The Bay Diner")
    assert core == "golden gate cafe" and alias == "bay diner"
    assert normalize_name("Smith Consulting a/k/a Smith & Co")[2] == "smith"  # "co" strips as a legal suffix
    assert normalize_name("Acme Corp fka Acme Industries")[2] == "acme industries"
    assert normalize_name("Willow Tree formerly known as Old Oak Salon")[2] == "old oak salon"
    assert normalize_name("Patel Traders nee Patel & Sons")[2] == "patel sons"
    assert normalize_name("Riverside Clinic doing business as River Health")[2] == "river health"
    # no marker -> empty alias, core unaffected
    assert normalize_name("Plain Business Name Inc")[1:3] == ("plain business name", "")


def test_name_french_etablissements_is_legal_like():
    _, core, _, legal, first = normalize_name("Etablissements Dupont")
    assert core == "dupont" and legal == "ets" and first == 1
    assert normalize_name("Ets Martin Freres")[3] == "ets"


def test_address_canon_states_numbers():
    a = normalize_address("3029 Veterans Parkway, City Of Marshfield, Wisconsin")
    b = normalize_address("3029 Veterans Pkwy, City Of Marshfield, WI")
    assert a[0] == b[0]
    addr_norm, alpha, nums, postcode, street, land = a
    assert nums == "3029" and street == "3029_veterans" and postcode == ""
    assert "wi" in alpha.split()


def test_address_french_and_landmark_and_pin():
    assert normalize_address("63 R. DE DIEPPE, LILLE")[0] == normalize_address("63 Rue de Dieppe, Lille")[0]
    a = normalize_address("12 MG Road, Near SBI ATM, Pune 411001")
    assert a[5] == "near sbi atm" and a[3] == "411001" and "sbi" not in a[0]


def test_address_french_bis_ter_skipped_in_street_key():
    with_mod = normalize_address("5 bis Rue de la Paix, Paris")
    without_mod = normalize_address("5 Rue de la Paix, Paris")
    assert with_mod[4] == without_mod[4]  # "bis" doesn't get picked as the street-key token
    assert with_mod[4] != ""  # sanity: a real street key was still found (not accidentally empty)


def test_address_india_house_number_and_floor_and_township():
    a = normalize_address("HN 753 E-1, Bharat Nagar")
    assert "hn" not in a[0].split()
    b = normalize_address("F-13 Kadam Chawl 1st Flr Bhuleshwar")
    assert "fl" in b[0].split() and "flr" not in b[0].split()
    c = normalize_address("3170 Exacta Lane, Raleigh Township, NC")
    assert "twp" in c[0].split() and "township" not in c[0].split()


def test_split_raw_keeps_devanagari_words():
    assert split_raw("मॉडर्न फाउंडेशन") == ["मॉडर्न", "फाउंडेशन"]


def test_normalize_frame_columns():
    df = pd.DataFrame({"entity_id": ["S3-1"], "business_name": ["मॉडर्न X"],
                       "business_address": [""], "country": ["India"]})
    out = normalize_frame(df, None, n_jobs=1)
    assert out.loc[0, "is_s3"] == 1 and out.loc[0, "script_flag"] == 1
    assert out.loc[0, "addr_norm"] == "" and out.loc[0, "rid"] == 0


def test_phonetic_skeleton_bridges_transliteration():
    for native, latin in [("मार्केटिंग", "marketing"), ("लिमिटेड", "limited"), ("परफेक्ट", "perfect"),
                          ("సిస్టమ్స్", "systems")]:
        assert phonetic(to_ascii(native)) == phonetic(latin)
    assert phonetic("Philip") == phonetic("filip") == "flp"
    assert phonetic("khanna") == "kn" and phonetic("aqua") == "ak" and phonetic("box12") == "bks12"
