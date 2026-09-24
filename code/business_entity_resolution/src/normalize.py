"""Country-agnostic text normalization for business names and addresses.

Every dictionary in this module is hand-written (no external data) and is listed in
the methodology document. Canonical forms are short abbreviations so that "Street",
"St" and French "Saint" collapse to the same token on both sides of a pair.
"""
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from unidecode import unidecode

LEGAL = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "llc": "llc", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "llp": "llp", "lp": "lp", "plc": "plc",
    "pllc": "pllc", "pc": "pc", "opc": "opc", "sarl": "sarl", "sas": "sas",
    "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "cie": "cie", "compagnie": "cie",
}
NAME_STOP = {"and", "the", "of", "de", "du", "des", "la", "le", "les", "et", "d", "l",
             "www", "com", "net", "org"}
JUNK = {"null", "none", "nan"}

ADDR_CANON = {
    "street": "st", "saint": "st", "road": "rd", "avenue": "av", "ave": "av",
    "boulevard": "bd", "blvd": "bd", "drive": "dr", "lane": "ln", "court": "ct",
    "place": "pl", "parkway": "pkwy", "highway": "hwy", "circle": "cir", "trail": "trl",
    "terrace": "ter", "square": "sq", "apartment": "apt", "suite": "ste", "sainte": "ste",
    "floor": "fl", "flr": "fl", "building": "bldg", "north": "n", "south": "s", "east": "e",
    "west": "w", "rue": "r", "allee": "all", "impasse": "imp", "chemin": "ch", "route": "rte",
    "faubourg": "fbg", "sector": "sec", "township": "twp",
}
ADDR_DROP = {"no", "hno", "hn", "h", "door", "unit", "number", "num", "null", "none", "nan", "po", "box"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy",
}

_ABBR_DOT = re.compile(r"(?<=\b[^\W\d_])\.")  # dot after a single letter: L.L.C. -> LLC
_LANDMARK = re.compile(r"^\s*(near|nr|opp|opposite|behind|beside|besides|next to|in front of|adjacent to)\b",
                       re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_NON_LATIN = r"[^\x00-ɏ]"


def _sep_table():
    """Translation table mapping BMP punctuation, symbols, separators and control chars to a space."""
    table = {}
    for cp in range(0x10000):
        cat = unicodedata.category(chr(cp))
        if cat[0] in "PSZ" or cat in ("Cc", "Cf"):
            table[cp] = " "
    return table


_SEP = _sep_table()


def split_raw(text):
    """NFKC + lowercase + punctuation split, before transliteration (native-script tokens kept intact)."""
    if not text:
        return []
    s = unicodedata.normalize("NFKC", text).lower().replace("&", " and ")
    return _ABBR_DOT.sub("", s).translate(_SEP).split()


@lru_cache(maxsize=1 << 20)
def to_ascii(tok):
    """Transliterate one token to lowercase ASCII alphanumerics (may yield several tokens)."""
    return re.sub(r"[^a-z0-9]+", " ", unidecode(tok).lower()).strip()


_PHON_MAP = str.maketrans({"c": "k", "q": "k", "x": "ks", "w": "v", "z": "s"})
_H_AFTER_CONS = re.compile(r"(?<=[bdfgjklmnprstv])h")
_VOWELS = re.compile(r"[aeiouy]")
_REPEAT = re.compile(r"(.)\1+")


@lru_cache(maxsize=1 << 20)
def phonetic(tok):
    """Language-agnostic consonant skeleton of an ASCII token (bridges transliteration variants).

    ph->f, c/q->k, x->ks, w->v, z->s; 'h' after a consonant dropped (kh, th, sh, bh...);
    vowels (incl. y) removed except the first character; repeated letters collapsed.
    E.g. unidecode("परफेक्ट") = "prphektt" and "perfect" both give "prfkt".
    """
    s = tok.lower().replace("ph", "f").translate(_PHON_MAP)
    s = _H_AFTER_CONS.sub("", s)
    s = s[:1] + _VOWELS.sub("", s[1:])
    return _REPEAT.sub(r"\1", s)


def tokens(text, table=None):
    """Normalized ASCII tokens: learned transliteration table first, unidecode fallback, junk removed."""
    out = []
    for tok in split_raw(text):
        if not tok.isascii():
            tok = (table.get(tok) if table else None) or to_ascii(tok)
            out.extend(tok.split())
        else:
            out.append(tok)
    return [t for t in out if t not in JUNK]


def normalize_name(text, table=None):
    """Return (name_norm, name_core, legal, legal_first) for a business name."""
    toks = tokens(text, table)
    legal = [LEGAL[t] for t in toks if t in LEGAL]
    core = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
    if not core:
        core = [t for t in toks if t not in LEGAL] or toks
    first = int(bool(toks) and toks[0] in LEGAL)
    return " ".join(toks), " ".join(core), " ".join(dict.fromkeys(legal)), first


def _map_states(toks):
    """Replace US state full names (one or two words) by their 2-letter codes."""
    out, i = [], 0
    while i < len(toks):
        two = " ".join(toks[i:i + 2])
        if i + 1 < len(toks) and two in US_STATES:
            out.append(US_STATES[two])
            i += 2
        else:
            out.append(US_STATES.get(toks[i], toks[i]))
            i += 1
    return out


def normalize_address(text, table=None):
    """Return (addr_norm, addr_alpha, nums, postcode, street_key, landmark) for an address."""
    keep, land = [], []
    for seg in (text or "").split(","):
        (land if _LANDMARK.match(seg) else keep).append(seg)
    toks = [t for t in tokens(" , ".join(keep), table) if t not in ADDR_DROP]
    toks = [ADDR_CANON.get(t, t) for t in _map_states(toks)]
    nums = [d for t in toks for d in _DIGITS.findall(t)]
    alpha = [t for t in toks if not any(c.isdigit() for c in t)]
    postcode = ""
    for i, d in enumerate(nums):
        if len(d) == 6 or (len(d) == 5 and i > 0):
            postcode = d
    street = ""
    for i, t in enumerate(toks):
        if any(c.isdigit() for c in t):
            nxt = next((u for u in toks[i + 1:] if u.isalpha() and len(u) > 1), "")
            if nxt:
                street = f"{_DIGITS.findall(t)[0]}_{nxt}"
                break
    landmark = " ".join(tokens(" ".join(land), table))
    return " ".join(toks), " ".join(alpha), " ".join(nums), postcode, street, landmark


def _norm_chunk(names, addrs, table):
    """Normalize one chunk of names and addresses (runs in a worker process)."""
    return [normalize_name(x, table) for x in names], [normalize_address(x, table) for x in addrs]


def normalize_frame(df, table=None, n_jobs=1, chunk=100_000):
    """Normalize a raw source frame into the `norm` contract (see spec §4); rid = row position."""
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_norm_chunk)(names[i:i + chunk], addrs[i:i + chunk], table)
        for i in range(0, len(names), chunk)
    )
    n = pd.DataFrame([r for p in parts for r in p[0]],
                     columns=["name_norm", "name_core", "legal", "legal_first"])
    a = pd.DataFrame([r for p in parts for r in p[1]],
                     columns=["addr_norm", "addr_alpha", "nums", "postcode", "street_key", "landmark"])
    out = pd.concat([n, a], axis=1)
    out.insert(0, "rid", np.arange(len(df), dtype=np.int64))
    out.insert(1, "entity_id", df["entity_id"].to_numpy())
    out.insert(2, "country", df["country"].to_numpy())
    out["name_raw"] = df["business_name"].to_numpy()
    out["addr_raw"] = df["business_address"].to_numpy()
    out["legal_first"] = out["legal_first"].astype(np.int8)
    out["is_s3"] = df["entity_id"].str.startswith("S3-").to_numpy().astype(np.int8)
    raw = df["business_name"] + " " + df["business_address"]
    out["script_flag"] = raw.str.contains(_NON_LATIN, regex=True).to_numpy().astype(np.int8)
    return out
