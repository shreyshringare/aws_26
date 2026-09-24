"""Tests for name/address normalization dictionaries and rules."""
import pandas as pd

from src.normalize import normalize_address, normalize_frame, normalize_name, split_raw


def test_name_legal_split_and_dots():
    norm, core, legal, first = normalize_name("Pvt. EFS Print Ventures Ltd.")
    assert core == "efs print ventures"
    assert legal == "pvt ltd" and first == 1
    assert normalize_name("Green & Auclair L.L.C.")[1:3] == ("green auclair", "llc")
    assert normalize_name("E.U.R.L. Maison")[2] == "eurl"


def test_name_junk_accents_and_web():
    assert normalize_name("-- Holloway Peak Inc Seafood")[1] == "holloway peak seafood"
    assert normalize_name("Norah Pletcher Báin Care")[0] == "norah pletcher bain care"
    assert normalize_name("wilfordhancock.com")[1] == "wilfordhancock"


def test_name_translit_table_used():
    table = {"प्राइवेट": "private", "लिमिटेड": "limited"}
    assert normalize_name("राम प्राइवेट लिमिटेड", table)[2] == "pvt ltd"


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
