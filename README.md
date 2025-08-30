# QSIRDV Epidemiological Model : Milestone 1 :  Baseline QSIRDV & Data

## Executive Summary

A comprehensive implementation of the QSIRDV (Quarantined-Susceptible-Infected-Recovered-Dead-Vaccinated) compartmental epidemiological model, designed for analyzing disease dynamics with quarantine and vaccination interventions. This implementation provides synthetic data generation capabilities for training and validating Neural Ordinary Differential Equations (Neural ODEs) and Universal Differential Equations (UDEs).

## Table of Contents

- [Model Architecture](#model-architecture)
- [Mathematical Framework](#mathematical-framework)
- [Implementation Features](#implementation-features)
- [Experimental Scenarios](#experimental-scenarios)
- [Results and Analysis](#results-and-analysis)
- [Installation and Setup](#installation-and-setup)
- [Usage Guide](#usage-guide)
- [Data Specifications](#data-specifications)
- [Outputs and Artifacts](#outputs-and-artifacts)

## Model Architecture

The QSIRDV model extends classical SIR epidemiological models by incorporating:
- **Quarantine dynamics** for isolated individuals
- **Vaccination programs** for susceptible populations
- **Mortality tracking** for comprehensive epidemic assessment
- **Differential recovery and death rates** for quarantined vs. infected individuals

## Mathematical Framework

### System of Ordinary Differential Equations

The model dynamics are governed by the following system of ODEs:

```
dQ/dt = κ·I - γq·Q - δq·Q
dS/dt = -β·S·I/N - ν·S
dI/dt = β·S·I/N - κ·I - γ·I - δ·I
dR/dt = γ·I + γq·Q
dD/dt = δ·I + δq·Q
dV/dt = ν·S
```

### Parameter Definitions

| Parameter | Description |
|-----------|-------------|
| β | Transmission rate coefficient |
| κ | Quarantine/isolation rate |
| γ | Recovery rate (infected) |
| γq | Recovery rate (quarantined) |
| δ | Death rate (infected) |
| δq | Death rate (quarantined) |
| ν | Vaccination rate |
| N | Total living population |

## Implementation Features

### Core Capabilities
- High-precision ODE integration using Tsit5 solver
- Reproducible stochastic noise generation
- Automated train/validation dataset splitting
- Multi-scenario parallel simulation support
- Comprehensive visualization suite

### Technical Specifications
- **Language**: Julia 1.9+
- **Key Dependencies**: DifferentialEquations.jl, Plots.jl, DataFrames.jl
- **Numerical Precision**: 64-bit floating point
- **Integration Tolerances**: reltol=1e-8, abstol=1e-10

## Experimental Scenarios

Three calibrated scenarios explore different epidemic dynamics:

### Scenario 1: Baseline
Standard epidemic progression with moderate interventions
- Transmission rate (β): 0.3
- Quarantine rate (κ): 0.05
- Vaccination rate (ν): 0.02

### Scenario 2: High Transmission
Aggressive disease spread with enhanced quarantine measures
- Transmission rate (β): 0.5
- Quarantine rate (κ): 0.10
- Vaccination rate (ν): 0.01

### Scenario 3: Strong Vaccination
Reduced transmission through intensive vaccination campaign
- Transmission rate (β): 0.2
- Quarantine rate (κ): 0.05
- Vaccination rate (ν): 0.05

## Results and Analysis

### Key Metrics Summary

| Scenario | Peak Infected | Peak Day | Total Deaths | Final Recovered | Final Vaccinated |
|----------|--------------|----------|--------------|-----------------|------------------|
| Baseline | 42.1 | 22 | 19.3 | 218.8 | 736.2 |
| High Transmission | 153.9 | 17 | 98.0 | 594.9 | 268.4 |
| Strong Vaccination | 10.8 | 4 | 2.8 | 32.2 | 964.7 |

### Key Insights
- Vaccination reduces peak infection by 74% compared to baseline
- High transmission scenarios show 5× mortality increase
- Early quarantine implementation critical for peak reduction


### Environment Setup
```julia
# Create and activate project environment
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Install required packages
Pkg.add([
    "DifferentialEquations",
    "Plots",
    "CSV",
    "DataFrames",
    "JLD2",
    "Random",
    "Statistics",
    "Printf",
    "Dates"
])
```

## Usage Guide

### Basic Execution
```bash
julia qsirdv_model.jl
```

### Data Loading
```julia
using JLD2, CSV, DataFrames

# Load complete dataset bundle
@load "qsirdv_datasets.jld2" datasets_noiseless datasets_noisy train_data val_data

# Load specific CSV files
train_df = CSV.read("qsirdv_train_data.csv", DataFrame)
val_df = CSV.read("qsirdv_val_data.csv", DataFrame)
```

### Custom Scenario Definition
```julia
# Define custom parameters
custom_params = [β, κ, γ, γq, δ, δq, ν]
custom_scenario = (
    name="Custom",
    params=custom_params,
    description="User-defined scenario"
)
```

## Data Specifications

### Initial Conditions
- Population: 1000 individuals
- Initial infected: 10
- Initial susceptible: 990
- All other compartments: 0

### Dataset Structure
- **Temporal Resolution**: Daily observations
- **Simulation Duration**: 160 days
- **Training Set**: Days 0-99 (100 observations)
- **Validation Set**: Days 100-160 (61 observations)
- **Noise Model**: 5% relative Gaussian noise on I and R compartments

## Outputs and Artifacts

### Generated Data Files
| File | Description | Format |
|------|-------------|--------|
| `qsirdv_scenario[1-3]_noiseless.csv` | Clean trajectory data | CSV |
| `qsirdv_scenario[1-3]_noisy.csv` | Noisy observations | CSV |
| `qsirdv_train_data.csv` | Training dataset | CSV |
| `qsirdv_val_data.csv` | Validation dataset | CSV |
| `qsirdv_datasets.jld2` | Complete data bundle | JLD2 |

### Visualization Outputs
| File | Description |
|------|-------------|
| `qsirdv_timeseries.png` | Temporal evolution of all compartments |
| `qsirdv_stackedbar.png` | Population distribution snapshots |
| `qsirdv_infected_comparison.png` | Cross-scenario infection dynamics |
| `qsirdv_noise_comparison.png` | Signal vs. noise visualization |

## Performance Considerations

- **Computational Requirements**: ~2GB RAM, <1 minute runtime
- **Numerical Stability**: Guaranteed non-negative populations

## QSIRDV Epidemiological Model : Milestone 2 : Neural ODE for SIQRDV Epidemic Model

A Julia implementation of Neural Ordinary Differential Equations (Neural ODEs) for learning epidemic dynamics using the SIQRDV (Susceptible-Infected-Quarantined-Recovered-Deaths-Vaccinated) compartmental model.

## Overview

This project implements a data-driven approach to learn epidemic dynamics using neural networks embedded within differential equations. The model learns the underlying dynamics from time-series data and can forecast future epidemic trajectories.

## Features

- **SIQRDV Compartmental Model**: Complete implementation with 6 compartments
- **Neural ODE Architecture**: 3-layer neural network (6→32→32→6)
- **Two-Phase Training**: ADAM optimizer for global search followed by BFGS for local refinement
- **Comprehensive Checkpointing**: Automatic saving of model progress at regular intervals
- **Multiple Forecasting Methods**: Both multi-step and one-step ahead predictions
- **Automated Visualization**: Generates publication-ready plots automatically
- **Performance Metrics**: MSE, MAE, and R² score tracking

## Requirements

### Julia Version
- Julia 1.8 or higher

### Required Packages
```julia
Lux
DiffEqFlux
DifferentialEquations
Optimization
OptimizationOptimJL
OptimizationOptimisers
Random
Plots
ComponentArrays
CSV
DataFrames
Statistics
Dates
```

## Installation

1. Clone or download the repository
2. Navigate to the project directory
3. Install required packages:

```julia
using Pkg
Pkg.add(["Lux", "DiffEqFlux", "DifferentialEquations", "Optimization", 
         "OptimizationOptimJL", "OptimizationOptimisers", "ComponentArrays",
         "CSV", "DataFrames", "Plots"])
```

## Usage

### Quick Start

Simply run the script from the command line:

```bash
julia neural_ode_siqrdv.jl
```

Or from Julia REPL:

```julia
include("neural_ode_siqrdv.jl")
```

The script will automatically:
1. Generate synthetic epidemic data
2. Train the Neural ODE model
3. Evaluate performance on test data
4. Save checkpoints and results
5. Generate visualization plots

### Expected Runtime

- **Total Training Time**: ~2-5 minutes (depending on hardware)
- **Epochs**: 7 total (5 ADAM + 2 BFGS)
- **Iterations**: 700 total

## Model Architecture

### SIQRDV Compartments
- **S**: Susceptible population
- **I**: Infected individuals
- **Q**: Quarantined patients
- **R**: Recovered individuals
- **D**: Deaths
- **V**: Vaccinated population

### Neural Network
- **Input Layer**: 6 nodes (compartment states)
- **Hidden Layer 1**: 32 nodes with tanh activation
- **Hidden Layer 2**: 32 nodes with tanh activation
- **Output Layer**: 6 nodes (derivative predictions)
- **Total Parameters**: ~1,350

### Training Strategy
1. **Phase 1 (Epochs 1-5)**: ADAM optimizer with learning rate 0.01
2. **Phase 2 (Epochs 6-7)**: BFGS optimizer for fine-tuning
3. **Loss Function**: Weighted MSE with emphasis on Infected (5x) and Quarantined (2x) compartments

## Output Files

The script generates 6 output files:

### CSV Files
1. **`training_checkpoints.csv`**: Complete training history with metrics at each checkpoint
   - Epoch number, iteration, loss, MSE, MAE, R², learning rate, optimizer, timestamp

2. **`best_checkpoint.csv`**: Information about the best performing model
   - Best epoch, iteration, loss, and performance metrics

3. **`final_checkpoint.csv`**: Summary of final model performance
   - All final metrics, training time, data statistics

4. **`predictions.csv`**: Complete prediction data
   - Time points, true values, predicted values for all compartments

### Visualization Files
5. **`results.png`**: 6-panel plot showing all compartment dynamics
   - Ground truth vs predictions
   - Train/test split marker
   - Multi-step and 1-step forecasts

6. **`loss.png`**: Training convergence plot
   - Loss curve over iterations
   - Checkpoint markers
   - Epoch boundaries
   - Optimizer transition point

## Performance Metrics

The model evaluates performance using:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score (Coefficient of Determination)**

Metrics are calculated for:
- Training set fit
- Test set multi-step forecast
- Test set one-step ahead forecast

## Example Results

Typical performance after training:
- **Training R²**: >0.99
- **Test R² (1-step)**: >0.95
- **Test R² (multi-step)**: >0.90

## Customization

### Modifying Hyperparameters

You can adjust key parameters in the script:

```julia
# Network architecture
hidden_dim = 32  # Number of hidden neurons

# Training parameters
adam_epochs = 5  # Number of ADAM epochs
bfgs_epochs = 2  # Number of BFGS epochs
adam_lr = 0.01   # ADAM learning rate

# Loss weights for [S, I, Q, R, D, V]
weights = [1.0, 5.0, 2.0, 1.0, 1.0, 1.0]

# Data split
n_train = 121  # 75% of 161 points
```

### Changing Initial Conditions

Modify the initial population distribution:

```julia
u0 = [990.0, 10.0, 0.0, 0.0, 0.0, 0.0]  # [S, I, Q, R, D, V]
```

### Adjusting Model Parameters

Change the true parameters for data generation:

```julia
p_true = [0.3, 0.05, 0.1, 0.08, 0.01, 0.005, 0.02]  # [β, κ, γ, γq, δ, δq, ν]
```







