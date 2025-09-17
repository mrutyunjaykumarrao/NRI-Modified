# NRI-Modified: Neural Relational Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-orange.svg)](https://pytorch.org/)

A modified implementation of Neural Relational Inference (NRI) for dynamic system modeling and analysis. This repository contains enhancements and modifications to the original NRI framework for improved performance and additional functionality.

![Neural Relational Inference (NRI)](nri.png)

## 🚀 Features

- **Enhanced NRI Implementation**: Modified neural relational inference with improved training dynamics
- **🎯 FSCR Integration**: Formation Stability Coefficient Reranking for improved interaction prediction
- **📊 Advanced Stability Metrics**: Spatial compactness, velocity coherence, and temporal consistency analysis
- **🔄 Intelligent Reranking**: Reorder predictions based on formation stability principles  
- **Trajectory Visualization**: Advanced visualization tools for system trajectories with stability overlays
- **Custom Performance Analysis**: Tools for analyzing model performance and behavior
- **Flexible Dataset Generation**: Enhanced synthetic dataset generation capabilities
- **Comprehensive Logging**: Detailed experiment tracking and logging system

## 📋 Requirements

- Python 3.7+
- PyTorch 1.0+
- NumPy
- Matplotlib
- NetworkX
- Other dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/mrutyunjaykumarrao/NRI-Modified.git
cd NRI-Modified
```

2. Create a virtual environment:
```bash
python -m venv nri-env
source nri-env/bin/activate  # On Windows: nri-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Training the Model

```bash
python train.py --epochs 500 --lr 0.0005 --hidden 256
```

### Generating Synthetic Data

```bash
cd data
python generate_dataset.py
```
This generates the springs dataset, use `--simulation charged` for charged particles.

### Visualizing Trajectories

```bash
python visualize_trajectory.py --load-folder logs/your_experiment_folder
```

### Performance Analysis with FSCR

```bash
python perform_lige.py --model-path logs/your_experiment_folder
```

This now includes Formation Stability Coefficient Reranking (FSCR) analysis:
- Original NRI predictions
- FSCR reranked predictions  
- Comprehensive stability metrics
- Visual comparison of results

### FSCR Testing and Validation

```bash
python test_fscr.py
```

### Training Options

To train the encoder or decoder separately:

```bash
python train_enc.py  # Train encoder only
python train_dec.py  # Train decoder only
```

### LSTM Baseline

Run the LSTM baseline (denoted *LSTM (joint)* in the original paper):
```bash
python lstm_baseline.py
```

## 📁 Project Structure

```
NRI-Modified/
├── data/                      # Data generation and processing
│   ├── generate_dataset.py    # Synthetic dataset generation
│   └── synthetic_sim.py       # Simulation utilities
├── logs/                      # Experiment logs and saved models
├── modules.py                 # Core neural network modules
├── train.py                   # Main training script
├── utils.py                   # Utility functions
├── fscr.py                    # 🎯 Formation Stability Coefficient Reranking
├── fscr_integration.py        # FSCR integration with NRI framework
├── test_fscr.py              # FSCR testing and validation suite
├── visualize_trajectory.py    # Visualization tools
├── perform_lige.py           # Performance analysis with FSCR
├── FSCR_README.md            # Detailed FSCR documentation
└── README.md                 # This file
```

## 🔬 Experiments

The `logs/` directory contains various experiment runs with different configurations:
- Model checkpoints (encoder.pt, decoder.pt)
- Training logs
- Experiment metadata

## 👥 Collaborators

- [@mrutyunjaykumarrao](https://github.com/mrutyunjaykumarrao) - Project Lead
- [@Nischay23](https://github.com/Nischay23) - Collaborator

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work is based on the original Neural Relational Inference implementation:

**Neural relational inference for interacting systems.**  
Thomas Kipf*, Ethan Fetaya*, Kuan-Chieh Wang, Max Welling, Richard Zemel.  
https://arxiv.org/abs/1802.04687  (*: equal contribution)

## 📚 Citation

If you use this code in your research, please cite both this modified version and the original work:

```bibtex
@misc{nri-modified,
  title={NRI-Modified: Enhanced Neural Relational Inference},
  author={Mrutyunjay Kumar Rao and Nischay},
  year={2025},
  url={https://github.com/mrutyunjaykumarrao/NRI-Modified}
}

@article{kipf2018neural,
  title={Neural Relational Inference for Interacting Systems},
  author={Kipf, Thomas and Fetaya, Ethan and Wang, Kuan-Chieh and Welling, Max and Zemel, Richard},
  journal={arXiv preprint arXiv:1802.04687},
  year={2018}
}
```

---

**Note**: This is a modified version of the original NRI implementation. For the original work, please refer to the [original repository](https://github.com/ethanfetaya/NRI).
