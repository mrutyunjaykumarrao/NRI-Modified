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

# Import FSCR components
from fscr import create_fscr_module
from fscr_integration import EnhancedNRIWithFSCR, FSCRVisualizer

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

    # --- 5. APPLY FSCR RERANKING ---
    
    print("\n--- APPLYING FSCR (Formation Stability Coefficient Reranking) ---")
    
    # Create FSCR module with custom configuration
    fscr_config = {
        'spatial_weight': 0.3,
        'velocity_weight': 0.3,
        'temporal_weight': 0.2,
        'interaction_weight': 0.2,
        'stability_threshold': 0.6
    }
    fscr = create_fscr_module(fscr_config)
    
    # Apply FSCR reranking
    reranked_interactions, reranked_probs, stability_info = fscr.rerank_interactions(
        test_simulation, 
        torch.tensor(predicted_edge_types),
        edge_probabilities.squeeze(),
        args.num_atoms
    )
    
    # --- 6. ENHANCED ANALYSIS WITH STABILITY METRICS ---
    
    # Create enhanced NRI model for detailed analysis
    enhanced_nri = EnhancedNRIWithFSCR(encoder, fscr_config=fscr_config)
    
    # Get detailed stability analysis
    with torch.no_grad():
        detailed_results = enhanced_nri.predict_with_stability_analysis(
            test_simulation, rel_rec, rel_send
        )
    
    # --- 7. VISUALIZE THE RESULTS ---
    
    print("\n--- LIGE RESULTS ---")
    print(f"Ground Truth Edges (The real answer):\n{ground_truth_relations.numpy()}")
    print(f"\nOriginal Predicted Edges:\n{predicted_edge_types}")
    print(f"\nFSCR Reranked Edges:\n{reranked_interactions.detach().numpy()}")
    
    # Print stability analysis
    if 'detailed_stability' in detailed_results:
        stability_metrics = detailed_results['detailed_stability']['overall_stability']
        print(f"\n--- FORMATION STABILITY ANALYSIS ---")
        print(f"Spatial Compactness: {stability_metrics['spatial_stability']:.4f}")
        print(f"Velocity Coherence: {stability_metrics['velocity_stability']:.4f}")
        print(f"Temporal Consistency: {stability_metrics['temporal_stability']:.4f}")
        print(f"Interaction Strength: {stability_metrics['interaction_stability']:.4f}")
        print(f"Overall Formation Stability Coefficient: {stability_metrics['formation_stability_coefficient']:.4f}")
        
        if stability_info['improvement'] > 0:
            print(f"\n✅ FSCR Improvement: +{stability_info['improvement']:.4f}")
            print(f"   Tested {stability_info['num_alternatives_tested']} alternative configurations")
        else:
            print(f"\n ℹ️  Original prediction was already optimal (no improvement from FSCR)")
    
    # Visualize graphs and stability analysis
    visualize_enhanced_results(
        ground_truth_relations.numpy(), 
        predicted_edge_types, 
        reranked_interactions.detach().numpy(),
        args.num_atoms,
        test_simulation,
        detailed_results
    )

def visualize_enhanced_results(true_edges, orig_pred_edges, reranked_edges, num_nodes, trajectory, detailed_results):
    """
    Enhanced visualization function that shows original predictions, FSCR reranked predictions,
    and comprehensive stability analysis.
    """
    
    # Create visualizer
    visualizer = FSCRVisualizer(figsize=(20, 12))
    
    # Plot comprehensive stability analysis
    if 'detailed_stability' in detailed_results:
        stability_fig = visualizer.plot_stability_analysis(
            detailed_results['detailed_stability'],
            "FSCR Formation Stability Analysis"
        )
        stability_fig.show()
    
    # Plot formation trajectory
    trajectory_fig = visualizer.plot_formation_trajectory(
        trajectory,
        torch.tensor(reranked_edges),
        "Agent Formation Trajectory with FSCR Analysis"
    )
    trajectory_fig.show()
    
    # Plot comparison of all three predictions
    visualize_comparison_graphs(true_edges, orig_pred_edges, reranked_edges, num_nodes)


def visualize_comparison_graphs(true_edges, orig_pred_edges, reranked_edges, num_nodes):
    """
    Visualize ground truth, original predictions, and FSCR reranked predictions side by side.
    """
    
    G_true = nx.DiGraph()
    G_orig = nx.DiGraph()
    G_reranked = nx.DiGraph()
    
    offdiag_indices = get_offdiag_indices(num_nodes).numpy()
    
    # Build graphs
    for i, edge_type in enumerate(true_edges):
        if edge_type > 0:
            sender, receiver = to_2d_idx(offdiag_indices[i], num_nodes)
            G_true.add_edge(int(sender), int(receiver))

    for i, edge_type in enumerate(orig_pred_edges):
        if edge_type > 0:
            sender, receiver = to_2d_idx(offdiag_indices[i], num_nodes)
            G_orig.add_edge(int(sender), int(receiver))
            
    for i, edge_type in enumerate(reranked_edges):
        if edge_type > 0:
            sender, receiver = to_2d_idx(offdiag_indices[i], num_nodes)
            G_reranked.add_edge(int(sender), int(receiver))

    # --- ENHANCED DCII MODULE IMPLEMENTATION ---
    print("\n--- ENHANCED DCII RESULTS ---")
    
    # Analyze all three graphs
    graphs = [
        ("Ground Truth", G_true),
        ("Original NRI", G_orig), 
        ("FSCR Reranked", G_reranked)
    ]
    
    for name, graph in graphs:
        print(f"\n{name} - Influential Nodes Ranking:")
        degree_list = list(graph.degree())
        ranked_drones = sorted(degree_list, key=lambda item: item[1], reverse=True)
        
        if not ranked_drones:
            print(f"  No edges detected in {name}")
        else:
            for drone, degree in ranked_drones:
                print(f"  - Drone {drone}: {degree} connections")
    
    # Plotting with enhanced layout
    all_nodes = sorted(list(set(G_true.nodes()) | set(G_orig.nodes()) | set(G_reranked.nodes())))
    layout_graph = nx.DiGraph()
    layout_graph.add_nodes_from(all_nodes)
    pos = nx.spring_layout(layout_graph, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Ground Truth
    axes[0].set_title("Ground Truth Interaction Graph", fontsize=14, fontweight='bold')
    nx.draw(G_true, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=12, font_weight='bold', 
            arrows=True, ax=axes[0], nodelist=all_nodes,
            edge_color='blue', width=2)
    
    # Original NRI Predictions
    axes[1].set_title("Original NRI Predictions", fontsize=14, fontweight='bold')
    nx.draw(G_orig, pos, with_labels=True, node_color='lightcoral', 
            node_size=1000, font_size=12, font_weight='bold',
            arrows=True, ax=axes[1], nodelist=all_nodes,
            edge_color='red', width=2)
    
    # FSCR Reranked
    axes[2].set_title("FSCR Reranked Predictions", fontsize=14, fontweight='bold') 
    nx.draw(G_reranked, pos, with_labels=True, node_color='lightgreen', 
            node_size=1000, font_size=12, font_weight='bold',
            arrows=True, ax=axes[2], nodelist=all_nodes,
            edge_color='green', width=2)
    
    plt.suptitle("Comparison: Ground Truth vs Original NRI vs FSCR Enhanced", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_graphs(true_edges, pred_edges, num_nodes):
    """
    Legacy function for backward compatibility.
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