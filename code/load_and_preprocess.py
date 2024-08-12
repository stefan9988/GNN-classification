import json
import networkx as nx
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

def load_first_n_lines(file_path, n_lines=1000):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # Skip the opening bracket
        file.readline()
        
        for _ in tqdm(range(n_lines), desc=f"Loading first {n_lines} lines"):
            line = file.readline().strip()
            if line.endswith(','):
                line = line[:-1]  # Remove trailing comma
            if line == ']':  # End of array
                break
            try:
                # Remove the leading comma if present
                if line.startswith(','):
                    line = line[1:]
                item = json.loads(line)
                processed_item = {
                    'id': item.get('id'),
                    'title': item.get('title'),
                    'year': item.get('year'),
                    'n_citation': item.get('n_citation'),
                    'doc_type': item.get('doc_type'),
                    'publisher': item.get('publisher'),
                    'doi': item.get('doi'),
                    'venue': item.get('venue', {}).get('raw') if isinstance(item.get('venue'), dict) else item.get('venue'),
                    'authors': ', '.join([author.get('name', '') for author in item.get('authors', [])]),
                    'fos': max(item.get('fos', []), key=lambda x: x.get('w', 0), default={'name': ''})['name']
                }
                data.append(processed_item)
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {line[:100]}...")  # Print first 100 characters of problematic line
    
    return pd.DataFrame(data)
    
def extract_top_20_fos(df):
    top_20 = df.fos.value_counts().nlargest(20).index.tolist()
    def replace_with_other(x):
        return x if x in top_20 else 'Other'
    df['fos'] = df['fos'].apply(replace_with_other)
    
    return df
    
if __name__ == '__main__':
    
    df = load_first_n_lines('data/citation.json', n_lines=4800)
    print(df.head())
    print(df.isna().sum())
    print(f"\nDataFrame shape: {df.shape}")
    df = extract_top_20_fos(df)
    print(df.fos.unique())
    