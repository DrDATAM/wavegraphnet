## WaveGraphNet & Structural Health Monitoring Baselines

This repository contains an end-to-end deep learning framework designed for Guided Wave-based Structural Health Monitoring (SHM). It processes high-dimensional sensor network signals to perform damage localization and forward wave attenuation modeling.

The framework features WaveGraphNet, a physics-guided coupled inverse-forward Graph Neural Network (GNN), alongside multiple sequential and spatial deep learning baseline architectures.

## System Architecture Overview

The codebase is built around predicting damage localization coordinates (xd​,yd​) using structural grid data transformed into the frequency domain via Fast Fourier Transforms (FFT).

                      +-----------------------------+
                      |   Raw Signal pkl Data       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | scipy.fft.rfft Preprocess   |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+------------------+       +------------------+       +------------------+
|     1D CNN       |       |       LSTM       |       |    GNN Models    |
| (Baseline Spec)  |       | (Temporal/Freq)  |       | (Spatial Graph)  |
+------------------+       +------------------+       +--------+---------+
                                                               |
                                        +----------------------+----------------------+
                                        |                                             |
                                        v                                             v
                             +--------------------+                        +--------------------+
                             |    FlexibleGNN     |                        |    WaveGraphNet    |
                             | (MLP/Attention v1) |                        | (Coupled/Improved) |
                             +--------------------+                        +---------+----------+
                                                                                     |
                                                                       +-------------+-------------+
                                                                       |                           |
                                                                       v                           v
                                                             +-------------------+       +-------------------+
                                                             |  Inverse Branch   |       |  Forward Branch   |
                                                             | (Predicts Coords) |       | (Physics Target)  |
                                                             +-------------------+       +-------------------+

## Project Structure & Requirements
Key Dependencies

    Core: torch, torch_geometric (PyG)

    Signal Processing & Data Handling: scipy, numpy, tqdm, pickle

## Expected Directory Architecture
Plaintext

├── data/
│   └── processed/
│       └── ogw_data.pkl         # Main preprocessed signal dictionary map
├── models/
│   ├── cnn1d.py                 # 1D CNN baseline
│   ├── lstm.py                  # LSTM sequential baseline
│   ├── gnn_baselines.py         # Flexible GNN architectures
│   ├── wavegraphnet.py          # Original Hierarchical WaveGraphNet
│   └── wavegraphnet_new.py      # WaveGraphNet with Dynamic Weighted Loss
└── utils/
    ├── data_loader.py           # Datasets (Cnn1DDataset, LstmDataset, StandardGraphDataset, CoupledModelDataset)
    ├── splits.py                # Split utilities (A/B cross-validation splits)
    ├── checkpointer.py          # Checkpoint saving utilities
    └── logger.py                # Results tracking and evaluation metric logs

## Supported Models & Execution

Each execution script reads configuration flags via argparse. Training processes include dynamic learning rate scheduling via ReduceLROnPlateau and automatic performance checkpoint saving.
1. Baseline 1D CNN (main_cnn.py)

    Processes raw fast Fourier amplitude/phase features stacked globally across sensor channels.
    
Bash

    python main_cnn.py --split A --epochs 150 --batch_size 32 --lr 0.001 --num_fft_bins 251

2. Sequential LSTM (main_lstm.py)

    Evaluates the sequential relationship across fixed FFT bins extracted from the physical sensor pairs.

Bash

    python main_lstm.py --split A --epochs 150 --batch_size 4 --lr 0.001 --lstm_hidden_dim 128 --num_lstm_layers 2

3. Spatial Graph Neural Networks (main_gnn_baselines.py)

    Maps the sensor network structure explicitly into a graph dataset (StandardGraphDataset). Supports simple MLP spatial layers or Multi-Head Graph Attention (GAT) structural encoders.

Bash

# Train using standard graph message-passing
python main_gnn_baselines.py --model simple_mlp --split A --epochs 150 --hidden_dim 128

# Train using multi-head GAT attention nodes
python main_gnn_baselines.py --model attention --split A --epochs 150

4. WaveGraphNet Frameworks (main_wavegraphnet.py / main_wavegraphnet_new.py)

A dual-branch architecture that combines structural inverse coordinate predictions with physical forward attenuation verification modeling.

Coupled Mode: Interlinks inverse localization loss with a physics-informed forward path calculation.

Inverse Only Mode: Skips the secondary structural calculation pathway during inference optimizations.

Bash

# Standard Hierarchical Attention WaveGraphNet
python main_wavegraphnet.py --mode coupled --split A --epochs 150 --inv_hidden_dim 128 --fwd_hidden_dim 128

# Improved WaveGraphNet (featuring learned Dynamic Weighted Loss optimization)
python main_wavegraphnet_new.py --mode coupled --split B --epochs 150 --batch_size 32

**Configuration Parameters Summary
CLI Argument	Type	Default	Description
--split	str	"A" / "B"	Validation fold partitioning pattern
--epochs	int	150	Maximum tracking iterations per training runtime
--batch_size	int	32 / 4	Minibatch sizes optimized per model architecture constraint
--lr	float	0.001	Primary optimization step size (Adam optimizer)
--num_fft_bins	int	251	Target bins for rFFT lookback length window configuration
--mode	str	"coupled"	Runtime pairing routine for WaveGraphNet targets (coupled, inverse_only)

**Checkpoints & Serialization 

Every script relies on centralized utilities to guarantee clean experimentation tracking:

    Saving: High-performance weight states are serialized via save_checkpoint capturing parameters, configs, and final validation criteria evaluations.

    Logging: Validated testing losses are logged cleanly back into a performance matrix using the structural configuration profiles under results.json.


## Citation

If you use this code in your reseracg, cite this as:

@Vsharma_wavegraphnet{sharma2026wavegraphnetphysicsconsistentguidedwavedamage,
      title={WaveGraphNet: Physics-Consistent Guided-Wave Damage Localization through Coupled Inverse-Forward Graph Learning}, 
      author={Vinay Sharma and Aditya Bharade and Olga Fink},
      year={2026},
      eprint={2605.20311},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.20311}, 
}