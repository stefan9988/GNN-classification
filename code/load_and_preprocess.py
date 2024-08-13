import json
import config
import pandas as pd
from tqdm import tqdm


def load_and_save_filtered_data(file_path, fos_list, output_path, n_lines=None):
    data = []
    fos_set = set(fos_list)  # Convert list to set for faster lookup
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Skip the opening bracket
        file.readline()
        
        for line_num in tqdm(range(n_lines) if n_lines else iter(int, 1), desc="Processing lines"):
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
                
                # Find the fos with the largest w
                max_fos = max(item.get('fos', []), key=lambda x: x.get('w', 0), default={'name': '', 'w': 0})
                
                # Check if the max_fos is in our list of interest
                if max_fos['name'] in fos_set:
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
                        'fos': max_fos['name']
                    }
                    data.append(processed_item)
            except json.JSONDecodeError as e:
                print(f"Error decoding line {line_num + 1}: {line[:100]}...")  
    
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    return df
    

if __name__ == '__main__':
    fos_list = [
    "Cluster analysis", "Computer science", "Key distribution in wireless sensor networks",
    "Cloud computing", "The Internet", "Wireless sensor network", "Artificial neural network",
    "Control theory", "Population", "Image segmentation", "Information system",
    "Feature extraction", "Deep learning", "Energy consumption", "Fuzzy logic",
    "Wireless network", "Scale-space segmentation", "Mobile robot",
    "Convolutional neural network", "Cognitive radio"
]
    # Load and preprocess data
    df = load_and_save_filtered_data('data/citation.json', fos_list, config.filtered_data_path, config.n_lines)
    # df = extract_top_21_fos(df)
    
    print("Data loaded and preprocessed")
    print(df.fos.value_counts().nlargest(21))
    print(f"DataFrame shape: {df.shape}")


    df = pd.read_csv(config.filtered_data_path, low_memory=False)
    print(df.head())
    
    is_all_int = df['id'].apply(lambda x: float(x).is_integer()).all()
    print(is_all_int)
    