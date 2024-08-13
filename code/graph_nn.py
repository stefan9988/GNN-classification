import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sentence_transformers import SentenceTransformer
import ast
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from sklearn.metrics import fbeta_score

import config

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_graph(df):
    G = nx.Graph()
    node_mapping = {}  # To map original IDs to 0-based indices
    
    for i, (_, row) in enumerate(df.iterrows()):
        node_mapping[row['id']] = i
        G.add_node(i, 
                   title=row['title'], 
                   year=row['year'],
                   n_citation=row['n_citation'],
                   doc_type=row['doc_type'],
                   publisher=row['publisher'],
                   venue=row['venue'],
                   authors=row['authors'],
                   fos=row['fos'])
    
    for _, row in df.iterrows():
        if isinstance(row['references'], str):
            references = ast.literal_eval(row['references'])
        else:
            references = row['references'] or []
        
        try:
            for ref in references:
                if ref in node_mapping:
                    G.add_edge(node_mapping[row['id']], node_mapping[ref])
        except TypeError:
            if references in node_mapping:
                G.add_edge(node_mapping[row['id']], node_mapping[references])
            
    return G, node_mapping

sentence_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2').to(device)

def extract_features(df):
    title_features = sentence_transformer_model.encode(df['title'].fillna('').tolist(), batch_size=32, show_progress_bar=True)
    author_features = sentence_transformer_model.encode(df['authors'].fillna('').tolist(), batch_size=32, show_progress_bar=True)
    venue_features = sentence_transformer_model.encode(df['venue'].fillna('').tolist(), batch_size=32, show_progress_bar=True)

    scaler = MinMaxScaler()
    year_citation_features = scaler.fit_transform(df[['year', 'n_citation']].fillna(0))

    doc_type_features = pd.get_dummies(df['doc_type'], prefix='doc_type')

    combined_features = np.hstack([
        title_features, 
        author_features, 
        venue_features, 
        year_citation_features,
        doc_type_features
    ])

    return torch.tensor(combined_features, dtype=torch.float, device=device)

def encode_labels(df):
    le = LabelEncoder()
    labels = le.fit_transform(df['fos'])
    return torch.tensor(labels, dtype=torch.long, device=device)

def transform_to_pytorch_geometric(G, features, labels):
    node_to_index = {node: i for i, node in enumerate(G.nodes())}
    edge_index = [[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in G.edges()]
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    
    data = Data(x=features, edge_index=edge_index, y=labels)
    data.num_classes = labels.max().item() + 1
    return data


class SimpleGNN_SAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleGNN_SAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, 16)
        self.conv2 = SAGEConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.6):
        super(SimpleGNN, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=dropout)
        # 8 heads * 8 features = 64 features as output
        self.conv2 = GATConv(8*8, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ComplexGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=4, hidden_dim=64, heads=8, dropout=0.6):
        super(ComplexGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.skips = ModuleList()

        # Input layer
        self.convs.append(GATConv(num_features, hidden_dim, heads=heads, dropout=dropout))
        self.batch_norms.append(BatchNorm1d(hidden_dim * heads))
        self.skips.append(Linear(num_features, hidden_dim * heads))

        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.batch_norms.append(BatchNorm1d(hidden_dim * heads))
            self.skips.append(Linear(hidden_dim * heads, hidden_dim * heads))

        # Output layer
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        self.skips.append(Linear(hidden_dim * heads, hidden_dim))

        # Final classification layer
        self.final_conv = GCNConv(hidden_dim, num_classes)
        
        # Global pooling for graph-level tasks (if needed)
        self.pool = global_mean_pool
        
        # Additional MLPs for enhanced expressiveness
        self.mlp = torch.nn.Sequential(
            Linear(num_classes, hidden_dim),
            BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers):
            # Main convolution
            conv = self.convs[i](x, edge_index)
            
            # Skip connection
            skip = self.skips[i](x)
            
            # Combine main path and skip connection
            x = conv + skip
            
            # Apply batch norm and activation
            x = self.batch_norms[i](x)
            x = F.elu(x)
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply final GCN layer
        x = self.final_conv(x, edge_index)
        
        # Global pooling (for graph-level tasks, if needed)
        if batch is not None:
            x = self.pool(x, batch)
        
        # Additional MLP for enhanced expressiveness
        x = self.mlp(x)
        
        return F.log_softmax(x, dim=-1)

def train(model, optimizer, criterion, loader, data):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[batch]
        loss = criterion(out, data.y[batch])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, data):
    model.eval()
    correct = 0
    predictions = []
    true_labels = []
    for batch in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index)[batch]
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y[batch]).sum().item()
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(data.y[batch].cpu().numpy())
    accuracy = correct / len(loader.dataset)
    f2_score = fbeta_score(true_labels, predictions, beta=2, average='weighted')
    return accuracy, f2_score

def plot_metrics(train_metrics, val_metrics, test_metrics):
    epochs = range(1, len(train_metrics['accuracy']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_metrics['accuracy'], 'b-', label='Train')
    plt.plot(epochs, val_metrics['accuracy'], 'r-', label='Validation')
    plt.plot(epochs, test_metrics['accuracy'], 'g-', label='Test')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metrics['f2_score'], 'b-', label='Train')
    plt.plot(epochs, val_metrics['f2_score'], 'r-', label='Validation')
    plt.plot(epochs, test_metrics['f2_score'], 'g-', label='Test')
    plt.title('Model F2 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F2 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(config.results_plot_path)
    plt.close()

if __name__ == '__main__':    
    df = pd.read_csv(config.filtered_data_path)
    
    print('Creating graph...')
    G, node_mapping = create_graph(df)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    features = extract_features(df)
    labels = encode_labels(df)

    data = transform_to_pytorch_geometric(G, features, labels)
    print(f"PyTorch Geometric Data created with {data.num_nodes} nodes and {data.num_edges} edges")

    # Move data to GPU
    data = data.to(device)

    # Split the data
    num_nodes = data.num_nodes
    node_indices = torch.arange(num_nodes)
    train_indices, test_indices = train_test_split(node_indices, test_size=config.test_size, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=config.val_size, random_state=42)

    # Create DataLoaders
    train_loader = DataLoader(train_indices, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_indices, batch_size=config.batch_size)
    test_loader = DataLoader(test_indices, batch_size=config.batch_size)

    if config.model == 'SimpleGNN_SAGE':
        model = SimpleGNN_SAGE(num_features=data.num_node_features, num_classes=data.y.max().item() + 1).to(device)
    elif config.model == 'SimpleGNN_GAT':
        model = SimpleGNN(num_features=data.num_node_features, num_classes=data.y.max().item() + 1).to(device)
    elif config.model == 'SimpleGCN':
        model = SimpleGCN(num_features=data.num_node_features, num_classes=data.y.max().item() + 1).to(device)
    elif config.model == 'ComplexGNN':
        model = ComplexGNN(num_features=data.num_features, 
                        num_classes=data.y.max().item() + 1, 
                        num_layers=config.num_layers, 
                        hidden_dim=config.hidden_dim, 
                        heads=config.heads, 
                        dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    if config.train:
        print("Training started")
        best_val_acc = 0
        train_metrics = {'accuracy': [], 'f2_score': []}
        val_metrics = {'accuracy': [], 'f2_score': []}
        test_metrics = {'accuracy': [], 'f2_score': []}

        for epoch in range(config.n_epochs):
            loss = train(model, optimizer, criterion, train_loader, data)
            train_acc, train_f2 = evaluate(model, train_loader, data)
            val_acc, val_f2 = evaluate(model, val_loader, data)
            test_acc, test_f2 = evaluate(model, test_loader, data)
            
            train_metrics['accuracy'].append(train_acc)
            train_metrics['f2_score'].append(train_f2)
            val_metrics['accuracy'].append(val_acc)
            val_metrics['f2_score'].append(val_f2)
            test_metrics['accuracy'].append(test_acc)
            test_metrics['f2_score'].append(test_f2)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), config.model_path)
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        print("Training completed")
    
        # Plot and save metrics
        plot_metrics(train_metrics, val_metrics, test_metrics)
    
    
    model.load_state_dict(torch.load(config.model_path))
    best_test_acc, best_test_f2 = evaluate(model, test_loader, data)
    print(f"Best model performance - Test Accuracy: {best_test_acc:.4f}, Test F2 Score: {best_test_f2:.4f}")