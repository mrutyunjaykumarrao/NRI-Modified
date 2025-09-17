"""
FSCR: Formation Stability Coefficient Reranking

This module implements Formation Stability Coefficient Reranking (FSCR) for 
improving Neural Relational Inference predictions by considering formation 
stability in multi-agent systems.

The FSCR algorithm evaluates the stability of agent formations and reranks
interaction predictions based on how they contribute to overall formation stability.

Key Components:
1. Formation Stability Metrics:
   - Spatial Compactness: How tightly grouped agents are
   - Velocity Coherence: How aligned agent velocities are
   - Temporal Consistency: How stable the formation is over time
   - Interaction Strength: How strong the predicted interactions are

2. Reranking Algorithm:
   - Calculates stability coefficients for each possible interaction
   - Reranks interactions based on their contribution to formation stability
   - Provides confidence scores for reranked predictions

Author: Mrutyunjay Kumar Rao, Nischay
Date: September 2025
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

# Use numpy for cosine similarity to avoid sklearn dependency issues
def cosine_similarity(X):
    """Compute cosine similarity using numpy to avoid sklearn dependency."""
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    return np.dot(X_normalized, X_normalized.T)


class FSCRModule:
    """
    Formation Stability Coefficient Reranking Module
    
    This class implements the FSCR algorithm for reranking NRI predictions
    based on formation stability analysis.
    """
    
    def __init__(self, 
                 spatial_weight: float = 0.3,
                 velocity_weight: float = 0.3, 
                 temporal_weight: float = 0.2,
                 interaction_weight: float = 0.2,
                 stability_threshold: float = 0.5):
        """
        Initialize FSCR module with configurable weights.
        
        Args:
            spatial_weight: Weight for spatial compactness metric
            velocity_weight: Weight for velocity coherence metric  
            temporal_weight: Weight for temporal consistency metric
            interaction_weight: Weight for interaction strength metric
            stability_threshold: Threshold for considering formations stable
        """
        self.spatial_weight = spatial_weight
        self.velocity_weight = velocity_weight
        self.temporal_weight = temporal_weight
        self.interaction_weight = interaction_weight
        self.stability_threshold = stability_threshold
        
        # Normalize weights to sum to 1
        total_weight = sum([spatial_weight, velocity_weight, temporal_weight, interaction_weight])
        self.spatial_weight /= total_weight
        self.velocity_weight /= total_weight
        self.temporal_weight /= total_weight
        self.interaction_weight /= total_weight
        
    def calculate_formation_stability_coefficient(self, 
                                                trajectory: torch.Tensor,
                                                interactions: torch.Tensor,
                                                interaction_probs: torch.Tensor) -> Dict[str, float]:
        """
        Calculate the Formation Stability Coefficient (FSC) for a given trajectory and interactions.
        
        Args:
            trajectory: Tensor of shape [batch_size, num_agents, timesteps, features]
            interactions: Predicted interaction matrix [num_agent_pairs]
            interaction_probs: Interaction probabilities [num_agent_pairs, num_edge_types]
            
        Returns:
            Dictionary containing individual stability metrics and overall FSC
        """
        # Extract positions and velocities
        positions = trajectory[:, :, :, :2]  # Assuming first 2 features are x, y positions
        velocities = trajectory[:, :, :, 2:4] if trajectory.shape[-1] >= 4 else self._compute_velocities(positions)
        
        # Calculate individual stability metrics
        spatial_stability = self._calculate_spatial_compactness(positions)
        velocity_stability = self._calculate_velocity_coherence(velocities)
        temporal_stability = self._calculate_temporal_consistency(positions, velocities)
        interaction_stability = self._calculate_interaction_strength(interaction_probs)
        
        # Compute overall Formation Stability Coefficient
        fsc = (self.spatial_weight * spatial_stability + 
               self.velocity_weight * velocity_stability +
               self.temporal_weight * temporal_stability +
               self.interaction_weight * interaction_stability)
        
        return {
            'spatial_stability': spatial_stability,
            'velocity_stability': velocity_stability, 
            'temporal_stability': temporal_stability,
            'interaction_stability': interaction_stability,
            'formation_stability_coefficient': fsc
        }
    
    def _compute_velocities(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute velocities from position trajectories."""
        velocities = torch.zeros_like(positions)
        velocities[:, :, 1:, :] = positions[:, :, 1:, :] - positions[:, :, :-1, :]
        return velocities
    
    def _calculate_spatial_compactness(self, positions: torch.Tensor) -> float:
        """
        Calculate spatial compactness of the formation.
        Higher values indicate more compact formations.
        """
        batch_size, num_agents, timesteps, _ = positions.shape
        compactness_scores = []
        
        for t in range(timesteps):
            # Get positions at time t
            pos_t = positions[0, :, t, :].detach().numpy()  # Take first batch
            
            if num_agents > 1:
                # Calculate pairwise distances
                distances = pdist(pos_t)
                
                # Compactness is inverse of average distance (normalized)
                avg_distance = np.mean(distances)
                compactness = 1.0 / (1.0 + avg_distance)  # Normalized to [0, 1]
            else:
                compactness = 1.0  # Single agent is perfectly compact
                
            compactness_scores.append(compactness)
        
        return np.mean(compactness_scores)
    
    def _calculate_velocity_coherence(self, velocities: torch.Tensor) -> float:
        """
        Calculate velocity coherence of the formation.
        Higher values indicate agents moving in similar directions.
        """
        batch_size, num_agents, timesteps, _ = velocities.shape
        coherence_scores = []
        
        for t in range(1, timesteps):  # Skip first timestep (zero velocity)
            # Get velocities at time t
            vel_t = velocities[0, :, t, :].detach().numpy()  # Take first batch
            
            if num_agents > 1:
                # Calculate cosine similarity between velocity vectors
                vel_norms = np.linalg.norm(vel_t, axis=1)
                
                # Only consider agents that are moving
                moving_agents = vel_norms > 1e-6
                
                if np.sum(moving_agents) > 1:
                    moving_velocities = vel_t[moving_agents]
                    similarity_matrix = cosine_similarity(moving_velocities)
                    
                    # Average similarity (excluding diagonal)
                    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
                    coherence = np.mean(similarity_matrix[mask])
                    coherence = (coherence + 1) / 2  # Normalize to [0, 1]
                else:
                    coherence = 1.0  # No moving agents or single moving agent
            else:
                coherence = 1.0  # Single agent is perfectly coherent
                
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _calculate_temporal_consistency(self, positions: torch.Tensor, velocities: torch.Tensor) -> float:
        """
        Calculate temporal consistency of the formation.
        Higher values indicate stable formation over time.
        """
        batch_size, num_agents, timesteps, _ = positions.shape
        
        if timesteps < 3:
            return 1.0  # Can't measure consistency with too few timesteps
            
        # Calculate formation center at each timestep
        centers = torch.mean(positions[0], dim=0)  # [timesteps, 2]
        
        # Calculate relative positions (agents relative to formation center)
        relative_positions = positions[0] - centers.unsqueeze(0)  # [num_agents, timesteps, 2]
        
        # Measure consistency as stability of relative positions
        consistency_scores = []
        
        for agent in range(num_agents):
            agent_rel_pos = relative_positions[agent]  # [timesteps, 2]
            
            # Calculate variance in relative position over time
            position_variance = torch.var(agent_rel_pos, dim=0).sum().item()
            
            # Consistency is inverse of variance (normalized)
            consistency = 1.0 / (1.0 + position_variance)
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores)
    
    def _calculate_interaction_strength(self, interaction_probs: torch.Tensor) -> float:
        """
        Calculate the strength of predicted interactions.
        Higher values indicate stronger, more confident interactions.
        """
        # Calculate entropy of interaction probabilities (lower entropy = more confident)
        epsilon = 1e-8
        entropy = -torch.sum(interaction_probs * torch.log(interaction_probs + epsilon), dim=-1)
        
        # Normalize entropy (max entropy for uniform distribution)
        max_entropy = np.log(interaction_probs.shape[-1])
        normalized_entropy = entropy / max_entropy
        
        # Interaction strength is inverse of entropy
        interaction_strength = 1.0 - torch.mean(normalized_entropy).item()
        
        return interaction_strength
    
    def rerank_interactions(self,
                          trajectory: torch.Tensor,
                          original_interactions: torch.Tensor,
                          interaction_probs: torch.Tensor,
                          num_agents: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Rerank interaction predictions based on formation stability.
        
        Args:
            trajectory: Agent trajectory data
            original_interactions: Original NRI interaction predictions
            interaction_probs: Interaction probability distributions
            num_agents: Number of agents in the system
            
        Returns:
            Tuple of (reranked_interactions, reranked_probs, stability_metrics)
        """
        # Calculate baseline stability with original interactions
        baseline_stability = self.calculate_formation_stability_coefficient(
            trajectory, original_interactions, interaction_probs
        )
        
        # Calculate stability for alternative interaction configurations
        interaction_stability_scores = []
        num_interactions = len(original_interactions)
        
        # Generate alternative interaction configurations by perturbing original
        alternative_configs = self._generate_alternative_configurations(
            original_interactions, interaction_probs, num_perturbations=10
        )
        
        for alt_interactions, alt_probs in alternative_configs:
            stability = self.calculate_formation_stability_coefficient(
                trajectory, alt_interactions, alt_probs
            )
            interaction_stability_scores.append(stability['formation_stability_coefficient'])
        
        # Find the configuration with highest stability
        best_idx = np.argmax(interaction_stability_scores)
        best_interactions, best_probs = alternative_configs[best_idx]
        best_stability = interaction_stability_scores[best_idx]
        
        # If no alternative is better, keep original
        if best_stability <= baseline_stability['formation_stability_coefficient']:
            reranked_interactions = original_interactions
            reranked_probs = interaction_probs
            final_stability = baseline_stability
        else:
            reranked_interactions = best_interactions
            reranked_probs = best_probs
            final_stability = self.calculate_formation_stability_coefficient(
                trajectory, best_interactions, best_probs
            )
        
        # Add reranking metadata
        rerank_info = {
            'baseline_stability': baseline_stability,
            'final_stability': final_stability,
            'improvement': final_stability['formation_stability_coefficient'] - 
                          baseline_stability['formation_stability_coefficient'],
            'num_alternatives_tested': len(alternative_configs)
        }
        
        return reranked_interactions, reranked_probs, rerank_info
    
    def _generate_alternative_configurations(self,
                                           original_interactions: torch.Tensor,
                                           original_probs: torch.Tensor,
                                           num_perturbations: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate alternative interaction configurations for comparison."""
        alternatives = []
        
        for _ in range(num_perturbations):
            # Create perturbations by slightly modifying probabilities
            noise_scale = 0.1
            noise = torch.randn_like(original_probs) * noise_scale
            perturbed_probs = F.softmax(torch.log(original_probs + 1e-8) + noise, dim=-1)
            
            # Get new interactions from perturbed probabilities
            perturbed_interactions = perturbed_probs.argmax(-1)
            
            alternatives.append((perturbed_interactions, perturbed_probs))
        
        return alternatives
    
    def get_stability_ranking(self, 
                            agents_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> List[Dict]:
        """
        Rank multiple agent configurations by their formation stability.
        
        Args:
            agents_data: List of (trajectory, interactions, probs) tuples
            
        Returns:
            List of dictionaries with stability metrics, sorted by FSC
        """
        rankings = []
        
        for i, (trajectory, interactions, probs) in enumerate(agents_data):
            stability = self.calculate_formation_stability_coefficient(trajectory, interactions, probs)
            stability['config_id'] = i
            rankings.append(stability)
        
        # Sort by formation stability coefficient (highest first)
        rankings.sort(key=lambda x: x['formation_stability_coefficient'], reverse=True)
        
        return rankings


def create_fscr_module(config: Optional[Dict] = None) -> FSCRModule:
    """
    Factory function to create FSCR module with optional configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FSCR module
    """
    if config is None:
        config = {}
        
    return FSCRModule(
        spatial_weight=config.get('spatial_weight', 0.3),
        velocity_weight=config.get('velocity_weight', 0.3),
        temporal_weight=config.get('temporal_weight', 0.2),
        interaction_weight=config.get('interaction_weight', 0.2),
        stability_threshold=config.get('stability_threshold', 0.5)
    )


# Example usage and testing functions
def example_usage():
    """Example of how to use the FSCR module."""
    
    # Create sample data
    batch_size, num_agents, timesteps, features = 1, 5, 20, 4
    trajectory = torch.randn(batch_size, num_agents, timesteps, features)
    
    num_interactions = num_agents * (num_agents - 1)  # Excluding self-interactions
    interactions = torch.randint(0, 3, (num_interactions,))  # 3 edge types
    interaction_probs = F.softmax(torch.randn(num_interactions, 3), dim=-1)
    
    # Initialize FSCR module
    fscr = create_fscr_module()
    
    # Calculate stability
    stability = fscr.calculate_formation_stability_coefficient(
        trajectory, interactions, interaction_probs
    )
    
    print("Formation Stability Analysis:")
    for metric, value in stability.items():
        print(f"  {metric}: {value:.4f}")
    
    # Perform reranking
    reranked_interactions, reranked_probs, rerank_info = fscr.rerank_interactions(
        trajectory, interactions, interaction_probs, num_agents
    )
    
    print(f"\nReranking Results:")
    print(f"  Stability improvement: {rerank_info['improvement']:.4f}")
    print(f"  Alternatives tested: {rerank_info['num_alternatives_tested']}")


if __name__ == "__main__":
    example_usage()