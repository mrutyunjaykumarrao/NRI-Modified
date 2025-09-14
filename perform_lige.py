import torch
import numpy as np
import argparse
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx

# Import necessary components from the project's own files
from modules import MLPEncoder
from utils import load_data, my_softmax, get_offdiag_indices, encode_onehot

def run_lige_and_visualize():
    """
    This function performs the LIGE (Latent Interaction Graph Extraction) step.
    It loads a trained encoder, feeds it a test simulation, and visualizes
    the predicted interaction graph alongside the ground truth.
    """
    
    # --- 1. SETUP: Manually specify the folder containing your best model ---
    
    # IMPORTANT: Paste the correct path you copied from the file explorer
    # It should be the folder that contains your encoder.pt and decoder.pt
    model_folder = "/Users/mrutyunjaykumarrao/Desktop/MAJOR_PROJECT/Simulations/NRI/logs/exp2025-09-10T06:54:55.801044" 
    
    print(f"Loading model from specified folder: {model_folder}")

    encoder_path = os.path.join(model_folder, 'encoder.pt')
    metadata_path = os.path.join(model_folder, 'metadata.pkl')

    if not os.path.exists(encoder_path) or not os.path.exists(metadata_path):
        print(f"ERROR: 'encoder.pt' or 'metadata.pkl' not found in {model_folder}")
        return

    # Load the metadata to get the same settings used for training
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        args = metadata['args']

    # --- 2. LOAD THE TRAINED LIGE MODEL (The Encoder) ---
    
    # Re-create the encoder with the saved settings
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types, args.encoder_dropout, args.factor)
    
    # Load the saved "brain" (the trained weights) into the model
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval() # Set the model to evaluation/inference mode
    
    print("Successfully loaded trained LIGE (encoder) model.")

    # --- 3. LOAD ONE PIECE OF DATA TO ANALYZE ---
    
    # Load the full test dataset using the correct suffix
    _, _, test_loader, _, _, _, _ = load_data(args.batch_size, args.suffix)
    
    # Get just the first batch of data and relations from the test set
    test_data_batch, test_relations_batch = next(iter(test_loader))
    
    # Get the very first simulation from that batch to analyze
    simulation_index = 0
    test_simulation = test_data_batch[simulation_index:simulation_index+1] # Slice to keep the batch dimension
    ground_truth_relations = test_relations_batch[simulation_index]

    print(f"Loaded Simulation #{simulation_index} from the test set to perform inference on.")
    # Trim the data to the length the model was trained on (args.timesteps)
    test_simulation = test_simulation[:, :, :args.timesteps, :]

    # --- 4. PERFORM LIGE (Run the Inference) ---
    
    # Prepare the relation senders/receivers matrix (same as in training)
    off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
    rel_rec = torch.FloatTensor(encode_onehot(np.where(off_diag)[1]))
    rel_send = torch.FloatTensor(encode_onehot(np.where(off_diag)[0]))

    # Feed the trajectory data into the encoder to get the predicted graph
    logits = encoder(test_simulation, rel_rec, rel_send)
    
    # Convert the model's raw output (logits) into a clear prediction
    edge_probabilities = my_softmax(logits, -1)
    predicted_edge_types = edge_probabilities.argmax(-1).squeeze().detach().numpy()

    # --- 5. VISUALIZE THE RESULTS ---
    
    print("\n--- LIGE RESULTS ---")
    print(f"Ground Truth Edges (The real answer):\n{ground_truth_relations.numpy()}")
    print(f"\nPredicted Edges (What the AI thinks):\n{predicted_edge_types}")
    
    visualize_graphs(ground_truth_relations.numpy(), predicted_edge_types, args.num_atoms)

def visualize_graphs(true_edges, pred_edges, num_nodes):
    """
    Helper function to draw the ground truth and predicted graphs,
    and also calculate and print the DCII ranking.
    """
    
    G_true = nx.DiGraph()
    G_pred = nx.DiGraph()
    
    offdiag_indices = get_offdiag_indices(num_nodes).numpy()

    for i, edge_type in enumerate(true_edges):
        if edge_type > 0:
            sender, receiver = to_2d_idx(offdiag_indices[i], num_nodes)
            G_true.add_edge(int(sender), int(receiver))

    for i, edge_type in enumerate(pred_edges):
        if edge_type > 0:
            sender, receiver = to_2d_idx(offdiag_indices[i], num_nodes)
            G_pred.add_edge(int(sender), int(receiver))

    # --- NEW: DCII MODULE IMPLEMENTATION ---
    print("\n--- DCII RESULTS (Influential Nodes Ranking) ---")
    
    # Networkx's degree function returns the total number of connections (in + out) for each node.
    # It returns a special "DegreeView" object, so we convert it to a standard list.
    degree_list = list(G_pred.degree())
    
    # Sort the list of drones by their degree (the second item in each tuple), from highest to lowest.
    ranked_drones = sorted(degree_list, key=lambda item: item[1], reverse=True)
    
    if not ranked_drones:
        print("The predicted graph has no edges.")
    else:
        print("Ranking of drones by number of connections:")
        for drone, degree in ranked_drones:
            print(f"  - Drone {drone}: {degree} connections")
    # --- END OF DCII MODULE ---


    # Plotting
    all_nodes = sorted(list(set(G_true.nodes()) | set(G_pred.nodes())))
    layout_graph = nx.DiGraph()
    layout_graph.add_nodes_from(all_nodes)
    pos = nx.spring_layout(layout_graph, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_title("Ground Truth Interaction Graph")
    nx.draw(G_true, pos, with_labels=True, node_color='lightblue', node_size=800, font_size=16, arrows=True, ax=axes[0], nodelist=all_nodes)
    
    axes[1].set_title("LIGE Predicted Interaction Graph")
    nx.draw(G_pred, pos, with_labels=True, node_color='lightgreen', node_size=800, font_size=16, arrows=True, ax=axes[1], nodelist=all_nodes)
    
    plt.suptitle("Comparison of Real vs. Predicted Drone Connections")
    plt.show()

def to_2d_idx(idx, num_cols):
    """Helper function to convert linear index to 2D matrix index."""
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx

if __name__ == '__main__':
    run_lige_and_visualize()