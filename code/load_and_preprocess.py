import json
import pandas as pd
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
                    'references': item.get('references'),
                    'venue': item.get('venue', {}).get('raw') if isinstance(item.get('venue'), dict) else item.get('venue'),
                    'authors': ', '.join([author.get('name', '') for author in item.get('authors', [])]),
                    'fos': max(item.get('fos', []), key=lambda x: x.get('w', 0), default={'name': ''})['name']
                }
                data.append(processed_item)
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {line[:100]}...")  
    
    return pd.DataFrame(data)
    
def extract_top_21_fos(df):
    # Get the top 20 most frequent values in 'fos' column
    top_20 = set(df['fos'].value_counts().nlargest(20).index)
    
    # Create a mapping dictionary for faster lookup
    fos_map = {fos: (fos if fos in top_20 and fos != '' else 'Other') for fos in df['fos'].unique()}
    
    # Apply the mapping and filter in one step
    return df[df['fos'].map(fos_map) != 'Other'].assign(fos=df['fos'].map(fos_map))

    

if __name__ == '__main__':
    # Load and preprocess data
    df = load_first_n_lines('data/citation.json', n_lines=4800)
    df = extract_top_21_fos(df)
    
    print("Data loaded and preprocessed")
    print(df.fos.value_counts().nlargest(20))
    print(f"DataFrame shape: {df.shape}")