batch_size = 2**14
n_epochs = 300
test_size = 0.2
val_size = 0.2
model = 'SimpleGNN_SAGE' # 'SimpleGNN_SAGE', 'SimpleGNN_GAT', 'SimpleGCN', 'ComplexGNN'
num_layers=6
hidden_dim=64 
heads=8
dropout=0.6
n_lines = 4000000

filtered_data_path = 'data/citation_filtered.csv'
results_plot_path = f'results/{model}.png'
model_path = f'model/{model}.pth'