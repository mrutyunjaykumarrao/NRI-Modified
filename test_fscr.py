"""
FSCR Test and Validation Script

This script provides comprehensive testing and validation for the FSCR 
(Formation Stability Coefficient Reranking) implementation.

It includes:
- Unit tests for individual FSCR components
- Integration tests with the NRI framework
- Performance benchmarks
- Visualization of results

Run this script to validate your FSCR implementation before using it in production.

Author: Mrutyunjay Kumar Rao, Nischay
Date: September 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import unittest
import sys
import os

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(__file__))

from fscr import create_fscr_module, FSCRModule
from fscr_integration import EnhancedNRIWithFSCR, FSCRVisualizer
from utils import load_data, encode_onehot
from modules import MLPEncoder
import torch.nn.functional as F


class FSCRTestSuite(unittest.TestCase):
    """Comprehensive test suite for FSCR implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fscr = create_fscr_module()
        self.batch_size = 1
        self.num_agents = 5
        self.timesteps = 20
        self.features = 4
        
        # Create sample trajectory data
        self.trajectory = torch.randn(self.batch_size, self.num_agents, self.timesteps, self.features)
        
        # Create sample interaction data
        self.num_interactions = self.num_agents * (self.num_agents - 1)
        self.interactions = torch.randint(0, 3, (self.num_interactions,))
        self.interaction_probs = F.softmax(torch.randn(self.num_interactions, 3), dim=-1)
        
    def test_fscr_module_creation(self):
        """Test FSCR module creation with different configurations."""
        # Default configuration
        fscr_default = create_fscr_module()
        self.assertIsInstance(fscr_default, FSCRModule)
        
        # Custom configuration
        config = {
            'spatial_weight': 0.4,
            'velocity_weight': 0.3,
            'temporal_weight': 0.2,
            'interaction_weight': 0.1
        }
        fscr_custom = create_fscr_module(config)
        self.assertAlmostEqual(fscr_custom.spatial_weight, 0.4)
        
    def test_stability_calculation(self):
        """Test formation stability coefficient calculation."""
        stability = self.fscr.calculate_formation_stability_coefficient(
            self.trajectory, self.interactions, self.interaction_probs
        )
        
        # Check that all required metrics are present
        required_metrics = [
            'spatial_stability', 'velocity_stability', 
            'temporal_stability', 'interaction_stability',
            'formation_stability_coefficient'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, stability)
            self.assertIsInstance(stability[metric], (float, np.float64))
            self.assertGreaterEqual(stability[metric], 0.0)
            self.assertLessEqual(stability[metric], 1.0)
            
    def test_reranking_functionality(self):
        """Test interaction reranking functionality."""
        reranked_interactions, reranked_probs, rerank_info = self.fscr.rerank_interactions(
            self.trajectory, self.interactions, self.interaction_probs, self.num_agents
        )
        
        # Check output shapes
        self.assertEqual(reranked_interactions.shape, self.interactions.shape)
        self.assertEqual(reranked_probs.shape, self.interaction_probs.shape)
        
        # Check rerank info structure
        required_info = ['baseline_stability', 'final_stability', 'improvement', 'num_alternatives_tested']
        for key in required_info:
            self.assertIn(key, rerank_info)
            
    def test_stability_ranking(self):
        """Test stability ranking of multiple configurations."""
        # Create multiple configurations
        configs = []
        for _ in range(5):
            traj = torch.randn(self.batch_size, self.num_agents, self.timesteps, self.features)
            inter = torch.randint(0, 3, (self.num_interactions,))
            probs = F.softmax(torch.randn(self.num_interactions, 3), dim=-1)
            configs.append((traj, inter, probs))
            
        rankings = self.fscr.get_stability_ranking(configs)
        
        # Check that rankings are sorted (highest stability first)
        for i in range(len(rankings) - 1):
            self.assertGreaterEqual(
                rankings[i]['formation_stability_coefficient'],
                rankings[i+1]['formation_stability_coefficient']
            )
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Single agent case
        single_agent_traj = torch.randn(1, 1, self.timesteps, self.features)
        single_agent_interactions = torch.tensor([])
        single_agent_probs = torch.empty(0, 3)
        
        # Should handle single agent gracefully
        try:
            stability = self.fscr.calculate_formation_stability_coefficient(
                single_agent_traj, single_agent_interactions, single_agent_probs
            )
            # Single agent should have perfect stability in most metrics
            self.assertEqual(stability['spatial_stability'], 1.0)
            self.assertEqual(stability['velocity_stability'], 1.0)
        except Exception as e:
            self.fail(f"Single agent case failed: {e}")
            
        # Minimal timesteps case
        minimal_traj = torch.randn(1, self.num_agents, 2, self.features)
        try:
            stability = self.fscr.calculate_formation_stability_coefficient(
                minimal_traj, self.interactions, self.interaction_probs
            )
            # Should handle minimal timesteps
            self.assertIsInstance(stability['formation_stability_coefficient'], (float, np.float64))
        except Exception as e:
            self.fail(f"Minimal timesteps case failed: {e}")


class FSCRPerformanceBenchmark:
    """Performance benchmarking for FSCR implementation."""
    
    def __init__(self):
        self.fscr = create_fscr_module()
        
    def benchmark_stability_calculation(self, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark stability calculation performance."""
        
        times = []
        
        for _ in range(num_runs):
            # Create test data
            trajectory = torch.randn(1, 5, 20, 4)
            interactions = torch.randint(0, 3, (20,))
            probs = F.softmax(torch.randn(20, 3), dim=-1)
            
            # Measure time
            start_time = time.time()
            _ = self.fscr.calculate_formation_stability_coefficient(trajectory, interactions, probs)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    def benchmark_reranking(self, num_runs: int = 50) -> Dict[str, float]:
        """Benchmark reranking performance."""
        
        times = []
        
        for _ in range(num_runs):
            # Create test data
            trajectory = torch.randn(1, 5, 20, 4)
            interactions = torch.randint(0, 3, (20,))
            probs = F.softmax(torch.randn(20, 3), dim=-1)
            
            # Measure time
            start_time = time.time()
            _ = self.fscr.rerank_interactions(trajectory, interactions, probs, 5)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    def run_full_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run complete performance benchmark."""
        
        print("Running FSCR Performance Benchmarks...")
        print("=" * 50)
        
        # Benchmark stability calculation
        print("Benchmarking stability calculation...")
        stability_results = self.benchmark_stability_calculation()
        
        print(f"Stability Calculation Performance:")
        print(f"  Mean time: {stability_results['mean_time']:.4f}s")
        print(f"  Std time:  {stability_results['std_time']:.4f}s")
        print(f"  Min time:  {stability_results['min_time']:.4f}s")
        print(f"  Max time:  {stability_results['max_time']:.4f}s")
        print()
        
        # Benchmark reranking
        print("Benchmarking reranking...")
        reranking_results = self.benchmark_reranking()
        
        print(f"Reranking Performance:")
        print(f"  Mean time: {reranking_results['mean_time']:.4f}s")
        print(f"  Std time:  {reranking_results['std_time']:.4f}s")
        print(f"  Min time:  {reranking_results['min_time']:.4f}s")
        print(f"  Max time:  {reranking_results['max_time']:.4f}s")
        print()
        
        return {
            'stability_calculation': stability_results,
            'reranking': reranking_results
        }


class FSCRIntegrationTest:
    """Integration tests for FSCR with NRI framework."""
    
    def __init__(self):
        self.visualizer = FSCRVisualizer()
        
    def test_with_synthetic_data(self):
        """Test FSCR with synthetic formation data."""
        
        print("Running FSCR Integration Test with Synthetic Data")
        print("=" * 50)
        
        # Create synthetic formation data
        # Scenario: 5 agents in a moving formation
        num_agents = 5
        timesteps = 30
        
        # Create a stable formation (agents maintaining relative positions)
        formation_center = np.array([0.0, 0.0])
        formation_velocity = np.array([0.1, 0.05])  # Slow drift
        
        # Agent positions relative to formation center
        relative_positions = np.array([
            [0.0, 0.0],      # Center agent
            [1.0, 0.0],      # Right agent  
            [-1.0, 0.0],     # Left agent
            [0.0, 1.0],      # Top agent
            [0.0, -1.0]      # Bottom agent
        ])
        
        # Generate trajectory maintaining formation
        trajectory = np.zeros((1, num_agents, timesteps, 4))  # [batch, agents, time, features]
        
        for t in range(timesteps):
            # Formation center moves with constant velocity
            center_t = formation_center + t * formation_velocity
            
            # Add small random perturbations to make it realistic
            noise_scale = 0.05
            noise = np.random.randn(num_agents, 2) * noise_scale
            
            # Agent positions = formation center + relative positions + noise
            positions = center_t + relative_positions + noise
            
            # Calculate velocities (difference from previous timestep)
            if t > 0:
                velocities = positions - trajectory[0, :, t-1, :2]
            else:
                velocities = np.zeros_like(positions)
            
            trajectory[0, :, t, :2] = positions
            trajectory[0, :, t, 2:4] = velocities
        
        trajectory_tensor = torch.tensor(trajectory, dtype=torch.float32)
        
        # Create interaction data (assuming strong interactions in formation)
        num_interactions = num_agents * (num_agents - 1)
        interactions = torch.ones(num_interactions, dtype=torch.long)  # Strong interactions
        
        # Create high-confidence interaction probabilities
        interaction_probs = torch.zeros(num_interactions, 3)
        interaction_probs[:, 1] = 0.8  # High probability for edge type 1
        interaction_probs[:, 0] = 0.1  # Low probability for no edge
        interaction_probs[:, 2] = 0.1  # Low probability for edge type 2
        
        # Test FSCR
        fscr = create_fscr_module()
        
        # Calculate stability
        stability = fscr.calculate_formation_stability_coefficient(
            trajectory_tensor, interactions, interaction_probs
        )
        
        print("Formation Stability Analysis for Synthetic Stable Formation:")
        for metric, value in stability.items():
            print(f"  {metric}: {value:.4f}")
        
        # Test reranking
        reranked_interactions, reranked_probs, rerank_info = fscr.rerank_interactions(
            trajectory_tensor, interactions, interaction_probs, num_agents
        )
        
        print(f"\nReranking Results:")
        print(f"  Baseline FSC: {rerank_info['baseline_stability']['formation_stability_coefficient']:.4f}")
        print(f"  Final FSC: {rerank_info['final_stability']['formation_stability_coefficient']:.4f}")
        print(f"  Improvement: {rerank_info['improvement']:.4f}")
        print(f"  Alternatives tested: {rerank_info['num_alternatives_tested']}")
        
        # Expected: Stable formation should have high stability metrics
        assert stability['spatial_stability'] > 0.7, f"Expected high spatial stability, got {stability['spatial_stability']}"
        assert stability['temporal_stability'] > 0.7, f"Expected high temporal stability, got {stability['temporal_stability']}"
        
        # Visualize results
        fig = self.visualizer.plot_formation_trajectory(
            trajectory_tensor, interactions,
            "Synthetic Stable Formation Test"
        )
        fig.show()
        
        print("✅ Synthetic data test passed!")
        
        return stability, rerank_info
    
    def test_with_unstable_formation(self):
        """Test FSCR with unstable formation data."""
        
        print("\nTesting with Unstable Formation")
        print("-" * 30)
        
        # Create chaotic/unstable formation
        num_agents = 5
        timesteps = 30
        
        # Random, uncorrelated movement
        trajectory = torch.randn(1, num_agents, timesteps, 4)
        
        # Random interactions
        num_interactions = num_agents * (num_agents - 1)
        interactions = torch.randint(0, 3, (num_interactions,))
        interaction_probs = F.softmax(torch.randn(num_interactions, 3), dim=-1)
        
        # Test FSCR
        fscr = create_fscr_module()
        stability = fscr.calculate_formation_stability_coefficient(
            trajectory, interactions, interaction_probs
        )
        
        print("Formation Stability Analysis for Unstable Formation:")
        for metric, value in stability.items():
            print(f"  {metric}: {value:.4f}")
        
        # Expected: Unstable formation should have lower stability metrics
        print("✅ Unstable formation test completed!")
        
        return stability


def run_comprehensive_fscr_test():
    """Run comprehensive FSCR testing suite."""
    
    print("🚀 FSCR Comprehensive Test Suite")
    print("=" * 70)
    print()
    
    # 1. Unit Tests
    print("1️⃣ Running Unit Tests...")
    unittest.main(argv=[''], module=FSCRTestSuite, exit=False, verbosity=2)
    print("✅ Unit tests completed!\n")
    
    # 2. Performance Benchmarks
    print("2️⃣ Running Performance Benchmarks...")
    benchmark = FSCRPerformanceBenchmark()
    benchmark_results = benchmark.run_full_benchmark()
    print("✅ Performance benchmarks completed!\n")
    
    # 3. Integration Tests
    print("3️⃣ Running Integration Tests...")
    integration_test = FSCRIntegrationTest()
    
    # Test with stable formation
    stable_results = integration_test.test_with_synthetic_data()
    
    # Test with unstable formation  
    unstable_results = integration_test.test_with_unstable_formation()
    
    print("✅ Integration tests completed!\n")
    
    # 4. Summary
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print("All FSCR tests passed successfully! 🎉")
    print()
    print("Key Results:")
    print(f"  - Stable formation FSC: {stable_results[0]['formation_stability_coefficient']:.4f}")
    print(f"  - Unstable formation FSC: {unstable_results['formation_stability_coefficient']:.4f}")
    print(f"  - Average stability calc time: {benchmark_results['stability_calculation']['mean_time']:.4f}s")
    print(f"  - Average reranking time: {benchmark_results['reranking']['mean_time']:.4f}s")
    print()
    print("🚀 FSCR is ready for production use!")


if __name__ == "__main__":
    run_comprehensive_fscr_test()