"""
Enhanced FSCR Integration with NRI Framework

This module provides integration between the FSCR (Formation Stability Coefficient Reranking)
system and the existing Neural Relational Inference framework for improved interaction prediction
in multi-agent systems.

Key Features:
- Integration with existing NRI encoder/decoder architecture
- Enhanced stability metrics for drone formations
- Real-time reranking during inference
- Compatibility with existing training pipeline

Author: Mrutyunjay Kumar Rao, Nischay
Date: September 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

from fscr import FSCRModule
from utils import my_softmax, get_offdiag_indices


class EnhancedNRIWithFSCR(nn.Module):
    """
    Enhanced NRI model that integrates FSCR for improved interaction prediction.
    
    This class wraps the original NRI encoder/decoder with FSCR reranking capabilities.
    """
    
    def __init__(self, 
                 encoder: nn.Module,
                 decoder: Optional[nn.Module] = None,
                 fscr_config: Optional[Dict] = None,
                 use_fscr_training: bool = False):
        """
        Initialize enhanced NRI with FSCR.
        
        Args:
            encoder: Pre-trained NRI encoder
            decoder: Optional NRI decoder
            fscr_config: Configuration for FSCR module
            use_fscr_training: Whether to use FSCR during training
        """
        super(EnhancedNRIWithFSCR, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.use_fscr_training = use_fscr_training
        
        # Initialize FSCR module
        if fscr_config is None:
            fscr_config = {
                'spatial_weight': 0.25,
                'velocity_weight': 0.25,
                'temporal_weight': 0.25,
                'interaction_weight': 0.25,
                'stability_threshold': 0.6
            }
        
        self.fscr = FSCRModule(**fscr_config)
        
    def forward(self, 
                inputs: torch.Tensor,
                rel_rec: torch.Tensor,
                rel_send: torch.Tensor,
                use_fscr: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional FSCR reranking.
        
        Args:
            inputs: Input trajectory data [batch_size, num_atoms, timesteps, dims]
            rel_rec: Receiver relations matrix
            rel_send: Sender relations matrix  
            use_fscr: Whether to apply FSCR reranking
            
        Returns:
            Dictionary containing predictions and stability metrics
        """
        # Get original NRI predictions
        logits = self.encoder(inputs, rel_rec, rel_send)
        edge_probs = my_softmax(logits, -1)
        edge_types = edge_probs.argmax(-1)
        
        results = {
            'logits': logits,
            'edge_probabilities': edge_probs,
            'edge_types': edge_types,
            'original_predictions': edge_types.clone()
        }
        
        # Apply FSCR reranking if requested
        if use_fscr and not (self.training and not self.use_fscr_training):
            reranked_edges, reranked_probs, stability_info = self.fscr.rerank_interactions(
                inputs, edge_types, edge_probs, inputs.shape[1]
            )
            
            results.update({
                'reranked_edge_types': reranked_edges,
                'reranked_probabilities': reranked_probs,
                'stability_metrics': stability_info
            })
        
        return results
    
    def predict_with_stability_analysis(self, 
                                      inputs: torch.Tensor,
                                      rel_rec: torch.Tensor,
                                      rel_send: torch.Tensor) -> Dict[str, any]:
        """
        Make predictions with comprehensive stability analysis.
        
        Returns detailed stability metrics and confidence scores.
        """
        self.eval()
        
        with torch.no_grad():
            results = self.forward(inputs, rel_rec, rel_send, use_fscr=True)
            
            # Calculate detailed stability metrics
            if 'stability_metrics' in results:
                stability_analysis = self._detailed_stability_analysis(
                    inputs, results['edge_types'], results['edge_probabilities']
                )
                results['detailed_stability'] = stability_analysis
        
        return results
    
    def _detailed_stability_analysis(self, 
                                   trajectory: torch.Tensor,
                                   interactions: torch.Tensor,
                                   probs: torch.Tensor) -> Dict[str, any]:
        """Perform detailed stability analysis for research purposes."""
        
        # Basic stability metrics
        stability = self.fscr.calculate_formation_stability_coefficient(
            trajectory, interactions, probs
        )
        
        # Additional analysis
        num_agents = trajectory.shape[1]
        timesteps = trajectory.shape[2]
        
        # Calculate per-agent stability contributions
        agent_contributions = []
        for agent_id in range(num_agents):
            # Create modified trajectory excluding this agent
            modified_trajectory = torch.cat([
                trajectory[:, :agent_id], 
                trajectory[:, agent_id+1:]
            ], dim=1)
            
            if modified_trajectory.shape[1] > 0:  # Ensure we have remaining agents
                # Recalculate interactions for reduced system
                modified_interactions = self._get_interactions_excluding_agent(
                    interactions, agent_id, num_agents
                )
                modified_probs = self._get_probs_excluding_agent(
                    probs, agent_id, num_agents
                )
                
                if len(modified_interactions) > 0:
                    modified_stability = self.fscr.calculate_formation_stability_coefficient(
                        modified_trajectory, modified_interactions, modified_probs
                    )
                    
                    contribution = (stability['formation_stability_coefficient'] - 
                                  modified_stability['formation_stability_coefficient'])
                else:
                    contribution = 0.0
            else:
                contribution = stability['formation_stability_coefficient']  # Single agent case
            
            agent_contributions.append(contribution)
        
        # Calculate temporal stability evolution
        temporal_evolution = []
        window_size = min(5, timesteps // 2)
        
        for t in range(window_size, timesteps - window_size + 1):
            window_trajectory = trajectory[:, :, t-window_size:t+window_size, :]
            window_stability = self.fscr.calculate_formation_stability_coefficient(
                window_trajectory, interactions, probs
            )
            temporal_evolution.append({
                'timestep': t,
                'stability': window_stability['formation_stability_coefficient']
            })
        
        return {
            'overall_stability': stability,
            'agent_contributions': agent_contributions,
            'temporal_evolution': temporal_evolution,
            'stability_statistics': {
                'mean_agent_contribution': np.mean(agent_contributions),
                'std_agent_contribution': np.std(agent_contributions),
                'max_contributing_agent': np.argmax(agent_contributions),
                'min_contributing_agent': np.argmin(agent_contributions)
            }
        }
    
    def _get_interactions_excluding_agent(self, 
                                        interactions: torch.Tensor, 
                                        excluded_agent: int, 
                                        num_agents: int) -> torch.Tensor:
        """Get interaction matrix excluding a specific agent."""
        
        # Create mapping from original to reduced indices
        offdiag_indices = get_offdiag_indices(num_agents).numpy()
        
        valid_interactions = []
        for i, interaction in enumerate(interactions):
            sender_idx = offdiag_indices[i] // num_agents
            receiver_idx = offdiag_indices[i] % num_agents
            
            if sender_idx != excluded_agent and receiver_idx != excluded_agent:
                valid_interactions.append(interaction.item())
        
        return torch.tensor(valid_interactions) if valid_interactions else torch.tensor([])
    
    def _get_probs_excluding_agent(self, 
                                 probs: torch.Tensor, 
                                 excluded_agent: int, 
                                 num_agents: int) -> torch.Tensor:
        """Get probability matrix excluding a specific agent."""
        
        offdiag_indices = get_offdiag_indices(num_agents).numpy()
        
        valid_probs = []
        for i, prob_dist in enumerate(probs):
            sender_idx = offdiag_indices[i] // num_agents
            receiver_idx = offdiag_indices[i] % num_agents
            
            if sender_idx != excluded_agent and receiver_idx != excluded_agent:
                valid_probs.append(prob_dist)
        
        if valid_probs:
            return torch.stack(valid_probs)
        else:
            return torch.empty(0, probs.shape[-1])


class FSCRVisualizer:
    """
    Visualization tools for FSCR analysis and results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        
    def plot_stability_analysis(self, 
                              stability_results: Dict[str, any],
                              title: str = "FSCR Stability Analysis") -> plt.Figure:
        """
        Create comprehensive stability analysis visualization.
        
        Args:
            stability_results: Results from detailed stability analysis
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Overall stability metrics radar chart
        self._plot_stability_radar(axes[0, 0], stability_results['overall_stability'])
        
        # 2. Agent contribution bar chart  
        self._plot_agent_contributions(axes[0, 1], stability_results['agent_contributions'])
        
        # 3. Temporal stability evolution
        self._plot_temporal_evolution(axes[0, 2], stability_results['temporal_evolution'])
        
        # 4. Stability improvement comparison
        if 'stability_metrics' in stability_results:
            self._plot_improvement_comparison(axes[1, 0], stability_results['stability_metrics'])
        else:
            axes[1, 0].text(0.5, 0.5, 'No Reranking Applied', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Reranking Results')
        
        # 5. Agent contribution heatmap
        self._plot_agent_contribution_heatmap(axes[1, 1], stability_results['agent_contributions'])
        
        # 6. Statistics summary
        self._plot_statistics_summary(axes[1, 2], stability_results['stability_statistics'])
        
        plt.tight_layout()
        return fig
    
    def _plot_stability_radar(self, ax, stability_metrics):
        """Plot radar chart of stability metrics."""
        metrics = ['spatial_stability', 'velocity_stability', 'temporal_stability', 'interaction_stability']
        values = [stability_metrics[metric] for metric in metrics]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax = plt.subplot(2, 3, 1, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Stability Metrics', pad=20)
    
    def _plot_agent_contributions(self, ax, contributions):
        """Plot agent stability contributions."""
        agent_ids = list(range(len(contributions)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(contributions)))
        
        bars = ax.bar(agent_ids, contributions, color=colors)
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Stability Contribution')
        ax.set_title('Agent Stability Contributions')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, contrib in zip(bars, contributions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{contrib:.3f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_temporal_evolution(self, ax, temporal_data):
        """Plot temporal stability evolution."""
        if temporal_data:
            timesteps = [d['timestep'] for d in temporal_data]
            stabilities = [d['stability'] for d in temporal_data]
            
            ax.plot(timesteps, stabilities, 'o-', linewidth=2, color='green')
            ax.fill_between(timesteps, stabilities, alpha=0.3, color='green')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Formation Stability')
            ax.set_title('Temporal Stability Evolution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient temporal data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Temporal Evolution')
    
    def _plot_improvement_comparison(self, ax, stability_metrics):
        """Plot before/after stability comparison."""
        baseline = stability_metrics['baseline_stability']['formation_stability_coefficient']
        final = stability_metrics['final_stability']['formation_stability_coefficient']
        improvement = stability_metrics['improvement']
        
        categories = ['Baseline', 'After FSCR']
        values = [baseline, final]
        colors = ['red' if improvement <= 0 else 'orange', 'green' if improvement > 0 else 'red']
        
        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel('Formation Stability Coefficient')
        ax.set_title(f'FSCR Improvement: {improvement:+.4f}')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom')
    
    def _plot_agent_contribution_heatmap(self, ax, contributions):
        """Plot agent contribution as heatmap."""
        # Create a 2D representation for visualization
        n_agents = len(contributions)
        contrib_matrix = np.array(contributions).reshape(1, -1)
        
        im = ax.imshow(contrib_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(n_agents))
        ax.set_xticklabels([f'Agent {i}' for i in range(n_agents)])
        ax.set_yticks([])
        ax.set_title('Agent Contributions Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
    
    def _plot_statistics_summary(self, ax, stats):
        """Plot summary statistics."""
        ax.axis('off')
        
        stats_text = f"""
Stability Statistics:

Mean Contribution: {stats['mean_agent_contribution']:.4f}
Std Contribution: {stats['std_agent_contribution']:.4f}

Top Contributing Agent: {stats['max_contributing_agent']}
Least Contributing Agent: {stats['min_contributing_agent']}
        """.strip()
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Summary Statistics')
    
    def plot_formation_trajectory(self, 
                                trajectory: torch.Tensor,
                                interactions: torch.Tensor,
                                title: str = "Formation Trajectory") -> plt.Figure:
        """
        Plot the trajectory of agents in the formation.
        
        Args:
            trajectory: Agent trajectory tensor [batch, agents, time, features]
            interactions: Predicted interactions
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract positions (assuming first 2 features are x, y)
        positions = trajectory[0, :, :, :2].detach().numpy()  # [agents, time, 2]
        num_agents, timesteps, _ = positions.shape
        
        # Plot trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
        
        for agent_id in range(num_agents):
            x_traj = positions[agent_id, :, 0]
            y_traj = positions[agent_id, :, 1]
            
            # Plot trajectory
            ax.plot(x_traj, y_traj, 'o-', color=colors[agent_id], 
                   label=f'Agent {agent_id}', alpha=0.7, markersize=3)
            
            # Mark start and end
            ax.scatter(x_traj[0], y_traj[0], color=colors[agent_id], 
                      s=100, marker='s', edgecolor='black', linewidth=2)  # Start
            ax.scatter(x_traj[-1], y_traj[-1], color=colors[agent_id], 
                      s=100, marker='^', edgecolor='black', linewidth=2)  # End
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position') 
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig


def create_enhanced_nri(encoder_path: str, 
                       decoder_path: Optional[str] = None,
                       fscr_config: Optional[Dict] = None) -> EnhancedNRIWithFSCR:
    """
    Factory function to create enhanced NRI with FSCR from saved models.
    
    Args:
        encoder_path: Path to saved encoder model
        decoder_path: Optional path to saved decoder model
        fscr_config: FSCR configuration
        
    Returns:
        Enhanced NRI model with FSCR
    """
    from modules import MLPEncoder, MLPDecoder
    
    # Load encoder (you'll need to adapt this based on your model architecture)
    # This is a placeholder - adjust according to your saved model format
    encoder = torch.load(encoder_path)
    decoder = torch.load(decoder_path) if decoder_path else None
    
    return EnhancedNRIWithFSCR(encoder, decoder, fscr_config)