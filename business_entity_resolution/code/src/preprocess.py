import pandas as pd
import re
import unidecode

def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = unidecode.unidecode(name)
    name = name.replace("&", " and ")
    suffixes = r'\b(corp|corporation|inc|pvt|ltd|co|llc|gmbh|sarl)\b'
    name = re.sub(suffixes, '', name)
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_address(address):
    if pd.isna(address):
        return ""
    address = str(address).lower()
    address = unidecode.unidecode(address)
    address = address.replace("&", " and ")
    address = re.sub(r'[^a-z0-9\s]', ' ', address)
    
    abbr = {
        r'\brd\b': 'road',
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bflr\b': 'floor',
        r'\bopp\b': 'opposite',
        r'\bnr\b': 'near'
    }
    for k, v in abbr.items():
        address = re.sub(k, v, address)
    
    address = re.sub(r'\s+', ' ', address).strip()
    return address

def extract_numbers(text):
    if pd.isna(text):
        return set()
    text = str(text)
    nums = set(re.findall(r'\b\d+\b', text))
    return nums

def preprocess_df(df):
    df = df.copy()
    if 'business_name' in df.columns:
        df['clean_name'] = df['business_name'].apply(normalize_name)
    if 'business_address' in df.columns:
        df['clean_address'] = df['business_address'].apply(normalize_address)
    
    if 'business_name' in df.columns:
        df['nums_name'] = df['business_name'].apply(extract_numbers)
    if 'business_address' in df.columns:
        df['nums_address'] = df['business_address'].apply(extract_numbers)
    return df
