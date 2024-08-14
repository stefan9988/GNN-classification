# Graph Neural Network for Citation Network Analysis

This project implements a Graph Neural Network (GNN) to analyze and classify papers in a citation network. It uses PyTorch Geometric to build and train the GNN on a dataset of academic papers.

## Project Structure

- `graph_nn.py`: The main script that orchestrates data loading, preprocessing, model training, and evaluation.
- `config.py`: Configuration file containing various settings and hyperparameters.
- `data_preprocessing.py`: Script for loading and preprocessing the raw citation data.

## Dependencies

- Python 3.7+
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- NetworkX
- scikit-learn
- sentence-transformers
- Matplotlib

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/stefan9988/GNN-classification.git
   cd <project-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the citation dataset and place it in the `data/` directory.

4. Update the `config.py` file with the appropriate paths and settings.

## Usage

1. Preprocess the data:
   ```
   python data_preprocessing.py
   ```

2. Train and evaluate the model:
   ```
   python graph_nn.py
   ```

## Model Architecture

The project includes several GNN architectures:

- SimpleGNN_SAGE: A simple GraphSAGE model
- SimpleGNN_GAT: A Graph Attention Network (GAT) model
- SimpleGCN: A Graph Convolutional Network (GCN) model
- ComplexGNN: A more sophisticated GNN with skip connections and additional MLPs

The model architecture can be selected in the `config.py` file.

## Data

The project uses a citation network dataset. Each paper is represented as a node in the graph, with edges representing citations between papers. The task is to classify papers into different fields of study (FoS).

## Features

- Graph creation from citation data
- Feature extraction using Sentence Transformers
- Multiple GNN architectures
- Training models
- Evaluation using accuracy and F2 score
- Visualization of training metrics

## Results

The training progress and final results are saved in the specified output directory. This includes:

- A plot of accuracy and F2 score over epochs
- The best model weights
- Evaluation metrics on the test set

