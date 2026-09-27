import pandas as pd
import re
import unidecode

SUFFIX_RE = re.compile(r'\b(corp|corporation|inc|pvt|ltd|co|llc|gmbh|sarl|sas|sasu|eurl|plc|bv|nv|srl|spa)\b')
NON_ALNUM_RE = re.compile(r'[^a-z0-9\s]')
MULTI_SPACE_RE = re.compile(r'\s+')
NUM_RE = re.compile(r'\b\d+\b')

ADDR_ABBR = {
    re.compile(r'\brd\b'): 'road',
    re.compile(r'\bst\b'): 'street',
    re.compile(r'\bave\b'): 'avenue',
    re.compile(r'\bflr\b'): 'floor',
    re.compile(r'\bopp\b'): 'opposite',
    re.compile(r'\bnr\b'): 'near',
    re.compile(r'\bblvd\b'): 'boulevard',
    re.compile(r'\bapt\b'): 'apartment',
    re.compile(r'\bste\b'): 'suite',
    re.compile(r'\bdr\b'): 'drive',
    re.compile(r'\bln\b'): 'lane',
}


def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = unidecode.unidecode(name)
    name = name.replace("&", " and ")
    name = SUFFIX_RE.sub('', name)
    name = NON_ALNUM_RE.sub(' ', name)
    name = MULTI_SPACE_RE.sub(' ', name).strip()
    return name


def normalize_address(address):
    if pd.isna(address):
        return ""
    address = str(address).lower()
    address = unidecode.unidecode(address)
    address = address.replace("&", " and ")
    address = NON_ALNUM_RE.sub(' ', address)
    for pat, repl in ADDR_ABBR.items():
        address = pat.sub(repl, address)
    address = MULTI_SPACE_RE.sub(' ', address).strip()
    return address


def normalize_country(country):
    if pd.isna(country):
        return "unknown"
    c = str(country).strip().lower()
    return c if c else "unknown"


def extract_numbers(text):
    if pd.isna(text) or text == "":
        return frozenset()
    return frozenset(NUM_RE.findall(str(text)))


def first_number(text):
    """First number token in text, or '' if none. Cheap proxy for a street number."""
    if pd.isna(text) or text == "":
        return ""
    m = NUM_RE.search(str(text))
    return m.group(0) if m else ""


def last_number(text):
    """Last number token in text, or '' if none. Cheap proxy for a postal code."""
    if pd.isna(text) or text == "":
        return ""
    nums = NUM_RE.findall(str(text))
    return nums[-1] if nums else ""


def preprocess_df(df):
    df = df.copy()
    if 'business_name' in df.columns:
        df['clean_name'] = df['business_name'].apply(normalize_name)
        df['nums_name'] = df['clean_name'].apply(extract_numbers)
    if 'business_address' in df.columns:
        df['clean_address'] = df['business_address'].apply(normalize_address)
        df['nums_address'] = df['clean_address'].apply(extract_numbers)
        df['street_num'] = df['business_address'].apply(first_number)
        df['postal_code'] = df['business_address'].apply(last_number)
    if 'country' in df.columns:
        df['country_norm'] = df['country'].apply(normalize_country)
    else:
        df['country_norm'] = "unknown"
    return df
