# QSIRDV Epidemiological Model : Milestone 1 

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


