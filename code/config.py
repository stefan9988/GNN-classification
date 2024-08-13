batch_size = 2**13
n_epochs = 300
test_size = 0.2
val_size = 0.2
model = 'ComplexGNN' # 'SimpleGNN_SAGE', 'SimpleGNN_GAT', 'SimpleGCN', 'ComplexGNN'
num_layers=4
hidden_dim=32 
heads=4
dropout=0.6
n_lines = 4000000
train = False

filtered_data_path = 'data/citation_filtered.csv'
results_plot_path = f'results/{model}.png'
model_path = f'model/{model}.pth'