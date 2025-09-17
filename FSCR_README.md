# FSCR: Formation Stability Coefficient Reranking

## Overview

Formation Stability Coefficient Reranking (FSCR) is an advanced enhancement to the Neural Relational Inference (NRI) framework that improves interaction prediction accuracy by considering the stability of agent formations in multi-agent systems.

## Key Features

### 🎯 Core Capabilities
- **Formation Stability Analysis**: Comprehensive metrics for evaluating agent formation stability
- **Intelligent Reranking**: Reorders NRI predictions based on formation stability principles
- **Multi-metric Evaluation**: Combines spatial, temporal, velocity, and interaction metrics
- **Real-time Processing**: Optimized for real-time inference and analysis

### 📊 Stability Metrics

FSCR evaluates four key stability dimensions:

1. **Spatial Compactness** (30% weight by default)
   - Measures how tightly grouped agents are in space
   - Higher values indicate more compact formations
   - Calculated using inverse average pairwise distances

2. **Velocity Coherence** (30% weight by default)
   - Evaluates how aligned agent velocities are
   - Uses cosine similarity between velocity vectors
   - Higher values indicate coordinated movement

3. **Temporal Consistency** (20% weight by default)
   - Assesses formation stability over time
   - Measures variance in relative agent positions
   - Higher values indicate stable formations

4. **Interaction Strength** (20% weight by default)
   - Evaluates confidence of predicted interactions
   - Based on entropy of interaction probabilities
   - Higher values indicate stronger, more confident predictions

## Usage

### Basic Usage

```python
from fscr import create_fscr_module

# Create FSCR module with default configuration
fscr = create_fscr_module()

# Calculate formation stability
stability = fscr.calculate_formation_stability_coefficient(
    trajectory, interactions, interaction_probs
)

# Apply reranking
reranked_interactions, reranked_probs, info = fscr.rerank_interactions(
    trajectory, interactions, interaction_probs, num_agents
)
```

### Advanced Configuration

```python
# Custom FSCR configuration
fscr_config = {
    'spatial_weight': 0.4,      # Emphasize spatial compactness
    'velocity_weight': 0.3,     # Standard velocity coherence
    'temporal_weight': 0.2,     # Standard temporal consistency  
    'interaction_weight': 0.1,  # De-emphasize interaction strength
    'stability_threshold': 0.7  # Higher stability threshold
}

fscr = create_fscr_module(fscr_config)
```

### Integration with NRI

```python
from fscr_integration import EnhancedNRIWithFSCR

# Create enhanced NRI with FSCR
enhanced_nri = EnhancedNRIWithFSCR(
    encoder=your_encoder,
    decoder=your_decoder,
    fscr_config=fscr_config
)

# Make predictions with FSCR reranking
results = enhanced_nri.predict_with_stability_analysis(
    inputs, rel_rec, rel_send
)
```

## Results Interpretation

### Formation Stability Coefficient (FSC)

The overall FSC is a weighted combination of all stability metrics, ranging from 0 to 1:

- **FSC > 0.8**: Highly stable formation
- **FSC 0.6-0.8**: Moderately stable formation  
- **FSC 0.4-0.6**: Somewhat unstable formation
- **FSC < 0.4**: Highly unstable formation

### Reranking Improvement

FSCR provides improvement metrics:

- **Positive improvement**: FSCR found a more stable configuration
- **Zero/negative improvement**: Original prediction was already optimal
- **Large improvements (>0.1)**: Significant stability enhancement

### Individual Metrics

Understanding each stability component:

```python
stability_results = {
    'spatial_stability': 0.85,      # Very compact formation
    'velocity_stability': 0.72,     # Good velocity alignment
    'temporal_stability': 0.68,     # Reasonably stable over time
    'interaction_stability': 0.91,  # High-confidence interactions
    'formation_stability_coefficient': 0.79  # Overall good stability
}
```

## Visualization

### Comprehensive Analysis

```python
from fscr_integration import FSCRVisualizer

visualizer = FSCRVisualizer()

# Plot complete stability analysis
fig = visualizer.plot_stability_analysis(
    detailed_results,
    "Formation Stability Analysis"
)
fig.show()
```

### Formation Trajectories

```python
# Plot agent trajectories with stability overlay
trajectory_fig = visualizer.plot_formation_trajectory(
    trajectory, interactions, "Agent Formation Analysis"
)
trajectory_fig.show()
```

## Performance Characteristics

### Computational Complexity

- **Stability Calculation**: O(n²t) where n = agents, t = timesteps
- **Reranking**: O(k×n²t) where k = alternatives tested (default: 10)
- **Memory Usage**: Linear with respect to trajectory length

### Typical Performance

On modern hardware with default settings:

- **Stability calculation**: ~0.01-0.05s for 5 agents, 20 timesteps
- **Full reranking**: ~0.1-0.5s for 5 agents, 20 timesteps
- **Memory footprint**: < 50MB for typical scenarios

## Best Practices

### Configuration Guidelines

1. **High-precision scenarios**: Increase `temporal_weight` and `interaction_weight`
2. **Real-time applications**: Reduce number of alternatives tested
3. **Formation-focused analysis**: Increase `spatial_weight` and `velocity_weight`
4. **Noisy data**: Lower `stability_threshold` to be less restrictive

### Integration Tips

1. **Training**: Use `use_fscr_training=False` for faster training
2. **Inference**: Enable FSCR for improved prediction accuracy
3. **Validation**: Compare baseline vs FSCR performance on your specific data
4. **Monitoring**: Track improvement statistics to tune configuration

## Troubleshooting

### Common Issues

**Issue**: Low stability scores across all metrics
- **Cause**: Highly chaotic or random agent behavior
- **Solution**: Check if data represents actual coordinated behavior

**Issue**: No improvement from reranking
- **Cause**: Original NRI predictions are already optimal
- **Solution**: This is expected for high-quality base predictions

**Issue**: High computational cost
- **Cause**: Large number of agents or long trajectories
- **Solution**: Reduce alternatives tested or use temporal windowing

### Performance Optimization

```python
# For real-time applications
fscr_config = {
    'spatial_weight': 0.4,
    'velocity_weight': 0.4,
    'temporal_weight': 0.2,
    'interaction_weight': 0.0,  # Skip expensive interaction analysis
    'stability_threshold': 0.5
}

# Reduce alternatives for speed
def fast_reranking(fscr, trajectory, interactions, probs, num_agents):
    return fscr.rerank_interactions(
        trajectory, interactions, probs, num_agents,
        num_perturbations=3  # Reduced from default 10
    )
```

## Research Applications

FSCR is particularly valuable for:

- **Drone swarm coordination analysis**
- **Multi-robot formation control**
- **Traffic flow optimization** 
- **Animal behavior studies**
- **Social dynamics modeling**

## Future Extensions

Potential enhancements to FSCR:

- **Adaptive weight learning**: Learn optimal metric weights from data
- **Hierarchical formations**: Support for nested formation structures
- **Dynamic stability**: Real-time stability tracking and prediction
- **Multi-objective optimization**: Balance stability with other objectives

## References

- Original NRI Paper: Kipf et al., "Neural Relational Inference for Interacting Systems" (2018)
- Formation Control Theory: Various robotics and control systems literature
- Multi-Agent Systems: Coordination and stability analysis methodologies