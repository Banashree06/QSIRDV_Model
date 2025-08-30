# QSIRDV Model Implementation Code

## Milestone 1

```julia
# qsirdv_model.jl - Milestone 1: Baseline QSIRDV & Data Generation
# QSIRDV epidemiological model implementation with data generation and visualization
# Generates synthetic datasets for training Neural ODEs/UDEs

using DifferentialEquations
using Plots
using Random, Statistics
using CSV, DataFrames, JLD2
using Printf
using Dates

println("Starting QSIRDV Model Implementation")
println("="^50)

# Define QSIRDV epidemiological model
function qsirdv!(du, u, p, t)
    """
    QSIRDV Model Compartments:
    Q - Quarantined (exposed but not yet infectious)
    S - Susceptible
    I - Infected (infectious)
    R - Recovered
    D - Dead
    V - Vaccinated
    
    Parameters: 
    β - transmission rate
    κ - quarantine/isolation rate 
    γ - recovery rate (infected)
    γq - recovery rate (quarantined)
    δ - death rate (infected)
    δq - death rate (quarantined)
    ν - vaccination rate
    """
    Q, S, I, R, D, V = u
    β, κ, γ, γq, δ, δq, ν = p
    N = Q + S + I + R + V  # total living population
    
    # Corrected differential equations
    du[1] = κ * I - γq * Q - δq * Q              # dQ/dt (Quarantined)
    du[2] = -β * S * I / N - ν * S               # dS/dt (Susceptible)
    du[3] = β * S * I / N - κ * I - γ * I - δ * I # dI/dt (Infected)
    du[4] = γ * I + γq * Q                       # dR/dt (Recovered)
    du[5] = δ * I + δq * Q                       # dD/dt (Dead)
    du[6] = ν * S                                # dV/dt (Vaccinated)
end

# Model configuration
# Initial conditions (Q, S, I, R, D, V)
u0 = [0.0, 990.0, 10.0, 0.0, 0.0, 0.0]  
tspan = (0.0, 160.0)  # Simulation time span (160 days)
compartment_names = ["Q", "S", "I", "R", "D", "V"]
compartment_labels = ["Quarantined", "Susceptible", "Infected", "Recovered", "Dead", "Vaccinated"]

# Define scenarios with different parameter combinations
# Parameters: (β, κ, γ, γq, δ, δq, ν)
scenarios = [
    (name="Baseline", 
     params=[0.3, 0.05, 0.1, 0.08, 0.01, 0.005, 0.02],
     description="Standard epidemic parameters"),
    
    (name="High_Transmission", 
     params=[0.5, 0.10, 0.1, 0.08, 0.02, 0.01, 0.01],
     description="Higher transmission and quarantine rates"),
    
    (name="Strong_Vaccination", 
     params=[0.2, 0.05, 0.1, 0.08, 0.01, 0.005, 0.05],
     description="Lower transmission with strong vaccination")
]

println("Configured scenarios:")
for (i, scenario) in enumerate(scenarios)
    β, κ, γ, γq, δ, δq, ν = scenario.params
    println("  $i. $(scenario.name): $(scenario.description)")
    println("β=$β, κ=$κ, γ=$γ, γq=$γq, δ=$δ, δq=$δq, ν=$ν")
end

# Task 2: Simulate trajectories over multiple scenarios
println("\nRunning ODE simulations...")
Random.seed!(1234)  # Set seed for reproducibility

sols = []
datasets_noiseless = []

for (i, scenario) in enumerate(scenarios)
    println("  Solving scenario $i: $(scenario.name)")
    
    try
        prob = ODEProblem(qsirdv!, u0, tspan, scenario.params)
        sol = solve(prob, Tsit5(), saveat=1.0, reltol=1e-8, abstol=1e-10)
        
        # Check for negative values (should not happen with proper parameters)
        if any(sol.u) do u
            any(x -> x < -1e-10, u)
        end
            @warn "Negative values detected in scenario $(scenario.name)"
        end
        
        push!(sols, sol)
        push!(datasets_noiseless, Array(sol))
        
        println("Simulation completed successfully")
        println("Final state: Q=$(round(sol[end][1], digits=2)), S=$(round(sol[end][2], digits=2)), I=$(round(sol[end][3], digits=2)), R=$(round(sol[end][4], digits=2)), D=$(round(sol[end][5], digits=2)), V=$(round(sol[end][6], digits=2))")
        
    catch e
        println("ERROR in simulation: $e")
        rethrow(e)
    end
end

# Task 3: Generate noisy datasets
println("\nGenerating noisy datasets...")
datasets_noisy = []

for (i, data) in enumerate(datasets_noiseless)
    println("Adding noise to scenario $i: $(scenarios[i].name)")
    
    noisy_data = copy(data)
    
    # Add small Gaussian noise to I (Infected) and R (Recovered) as specified
    noise_level = 0.05  # 5% relative noise
    
    # Add noise to Infected compartment (index 3)
    I_noise = noise_level .* abs.(data[3, :]) .* randn(size(data[3, :]))
    noisy_data[3, :] .+= I_noise
    
    # Add noise to Recovered compartment (index 4)
    R_noise = noise_level .* abs.(data[4, :]) .* randn(size(data[4, :]))
    noisy_data[4, :] .+= R_noise
    
    # Ensure non-negative values
    noisy_data = max.(noisy_data, 0.0)
    
    push!(datasets_noisy, noisy_data)
    
    println("    Added $(round(noise_level*100, digits=1))% relative noise to I and R compartments")
end

# Create train/validation split
println("\nCreating train/validation split...")
train_split = 100  # Use first 100 days for training
train_data = datasets_noisy[1][:, 1:train_split]     
val_data   = datasets_noisy[1][:, (train_split+1):end]   

println("  Training data: $(size(train_data, 2)) time points (days 0-$(train_split-1))")
println("  Validation data: $(size(val_data, 2)) time points (days $train_split-160)")

# Export data files
println("\nExporting datasets...")

# Create data directory if it doesn't exist
if !isdir("data")
    mkdir("data")
    println("Created 'data' directory")
end

# Save CSV files for each scenario (noisy data)
for (i, data) in enumerate(datasets_noisy)
    filename = "data/qsirdv_scenario$(i)_noisy.csv"
    println("  Saving $filename")
    
    df = DataFrame(
        time = 0:160,
        Q = data[1, :],
        S = data[2, :],
        I = data[3, :],
        R = data[4, :],
        D = data[5, :],
        V = data[6, :]
    )
    
    CSV.write(filename, df)
end

# Save noiseless data for comparison
for (i, data) in enumerate(datasets_noiseless)
    filename = "data/qsirdv_scenario$(i)_noiseless.csv"
    println("Saving $filename")
    
    df = DataFrame(
        time = 0:160,
        Q = data[1, :],
        S = data[2, :],
        I = data[3, :],
        R = data[4, :],
        D = data[5, :],
        V = data[6, :]
    )
    
    CSV.write(filename, df)
end

# Save train and validation datasets
println("  Saving data/qsirdv_train_data.csv")
train_df = DataFrame(
    time = 0:(train_split-1),
    Q = train_data[1, :],
    S = train_data[2, :],
    I = train_data[3, :],
    R = train_data[4, :],
    D = train_data[5, :],
    V = train_data[6, :]
)
CSV.write("data/qsirdv_train_data.csv", train_df)

println("  Saving data/qsirdv_val_data.csv")
val_df = DataFrame(
    time = train_split:160,
    Q = val_data[1, :],
    S = val_data[2, :],
    I = val_data[3, :],
    R = val_data[4, :],
    D = val_data[5, :],
    V = val_data[6, :]
)
CSV.write("data/qsirdv_val_data.csv", val_df)

# Save complete dataset bundle in JLD2 format
println("  Saving data/qsirdv_datasets.jld2")
@save "data/qsirdv_datasets.jld2" datasets_noiseless datasets_noisy train_data val_data scenarios u0 tspan compartment_names

# Task 4: Create basic visualization scripts
println("\nCreating visualizations...")

# Create figures directory if it doesn't exist
if !isdir("figures")
    mkdir("figures")
    println("  Created 'figures' directory")
end

# 1. Time series plot for all compartments (baseline scenario)
println(" Creating compartment curves...")
try
    p1 = plot(
        title="QSIRDV Model Dynamics - $(scenarios[1].name)",
        xlabel="Time (days)",
        ylabel="Population",
        linewidth=2.5,
        size=(1000, 600),
        legend=:outertopright,
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3
    )
    
    colors = [:purple, :blue, :red, :green, :black, :orange]
    for (i, (name, label)) in enumerate(zip(compartment_names, compartment_labels))
        plot!(p1, 0:160, datasets_noiseless[1][i, :], 
              label="$name - $label", 
              color=colors[i], 
              linewidth=2.5)
    end
    
    savefig(p1, "figures/qsirdv_timeseries.png")
    println("Saved compartment curves")
catch e
    println("ERROR creating time series plot: $e")
end

# 2. Stacked bar chart showing compartment distribution
println("  Creating stacked bar chart...")
try
    sample_times = collect(0:20:160)
    sample_indices = sample_times .+ 1
    data_sampled = datasets_noiseless[1][:, sample_indices]
    
    p2 = plot(
        title="Population Distribution Over Time",
        xlabel="Time (days)",
        ylabel="Population",
        size=(1000, 600),
        legend=:outertopright
    )
    
    colors = [:purple, :blue, :red, :green, :black, :orange]
    bar!(p2, sample_times, data_sampled', 
         label=permutedims(compartment_labels),
         color=permutedims(colors),
         bar_position=:stack,
         bar_width=15)
    
    savefig(p2, "figures/qsirdv_stackedbar.png")
    println("Saved stacked bar chart")
catch e
    println("ERROR creating stacked bar chart: $e")
end

# 3. Scenario comparison plot (Infected population)
println("  Creating scenario comparison plot...")
try
    p3 = plot(
        title="Infected Population Across Scenarios",
        xlabel="Time (days)",
        ylabel="Infected Population",
        linewidth=2.5,
        size=(1000, 600),
        legend=:topright,
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3
    )
    
    scenario_colors = [:red, :blue, :green]
    for (i, scenario) in enumerate(scenarios)
        plot!(p3, 0:160, datasets_noiseless[i][3, :], 
              label=scenario.name, 
              linewidth=2.5,
              color=scenario_colors[i])
    end
    
    savefig(p3, "figures/qsirdv_infected_comparison.png")
    println("Saved scenario comparison")
catch e
    println("ERROR creating comparison plot: $e")
end

# 4. Noisy vs Noiseless comparison
println("  Creating noise comparison plot...")
try
    p4 = plot(
        title="Noisy vs Noiseless Data - Infected & Recovered",
        xlabel="Time (days)",
        ylabel="Population",
        linewidth=2,
        size=(1000, 600),
        legend=:topright,
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3
    )
    
    # Plot infected compartment
    plot!(p4, 0:160, datasets_noiseless[1][3, :], 
          label="I (Noiseless)", 
          color=:red, 
          linewidth=2.5)
    plot!(p4, 0:160, datasets_noisy[1][3, :], 
          label="I (Noisy)", 
          color=:red, 
          alpha=0.6, 
          linestyle=:dash,
          linewidth=2)
    
    # Plot recovered compartment
    plot!(p4, 0:160, datasets_noiseless[1][4, :], 
          label="R (Noiseless)", 
          color=:green, 
          linewidth=2.5)
    plot!(p4, 0:160, datasets_noisy[1][4, :], 
          label="R (Noisy)", 
          color=:green, 
          alpha=0.6, 
          linestyle=:dash,
          linewidth=2)
    
    savefig(p4, "figures/qsirdv_noise_comparison.png")
    println("Saved noise comparison")
catch e
    println("ERROR creating noise comparison plot: $e")
end

# Generate summary statistics
println("\nCalculating summary statistics...")

peak_infected = [maximum(data[3, :]) for data in datasets_noiseless]
peak_time = [argmax(data[3, :]) - 1 for data in datasets_noiseless]
total_deaths = [data[5, end] for data in datasets_noiseless]
final_recovered = [data[4, end] for data in datasets_noiseless]
final_vaccinated = [data[6, end] for data in datasets_noiseless]
```

## Running the Code

Save the above code as `qsirdv_model.jl` and execute:

```bash
julia qsirdv_model.jl
```

## Expected Output

The script will generate:
- 6 CSV files with scenario data (3 noiseless, 3 noisy)
- 2 CSV files for train/validation split
- 1 JLD2 bundle with all datasets
- 4 PNG visualization files
- Console output showing simulation progress and final statistics

## Milestone 2

# ============================================
# NEURAL ODE FOR SIQRDV MODEL
# ============================================
# 
# SCRIPTED RUN - ONE COMMAND EXECUTION:
# --------------------------------------
# To run this entire script and generate all outputs, simply execute:
#
#   julia neural_ode_siqrdv.jl
#
# Or from Julia REPL:
#
#   include("neural_ode_siqrdv.jl")
#   run_neural_ode_training()
#
# This will automatically:
# 1. Install required packages (if needed)
# 2. Generate synthetic SIQRDV data
# 3. Train the Neural ODE with checkpointing
# 4. Evaluate model performance
# 5. Generate all CSV checkpoints
# 6. Create visualization plots
#
# Output files generated:
# - training_checkpoints.csv
# - best_checkpoint.csv
# - final_checkpoint.csv
# - predictions.csv
# - results.png
# - loss.png
#
# ============================================

# Package installation check
import Pkg

function check_and_install_packages()
    required_packages = [
        "Lux", "DiffEqFlux", "DifferentialEquations", 
        "Optimization", "OptimizationOptimJL", "OptimizationOptimisers",
        "Random", "Plots", "ComponentArrays",
        "CSV", "DataFrames", "Statistics", "Dates"
    ]
    
    println("Checking required packages...")
    for pkg in required_packages
        if !haskey(Pkg.project().dependencies, pkg)
            println("  Installing $pkg...")
            Pkg.add(pkg)
        end
    end
    println("All packages ready! ✓\n")
end

# Load packages
using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, Plots, ComponentArrays
using CSV, DataFrames, Statistics
using Dates

# ============================================
# Configuration Structure
# ============================================
struct TrainingConfig
    # Data parameters
    u0::Vector{Float64}
    tspan::Tuple{Float64, Float64}
    n_points::Int
    train_split::Float64
    
    # Model parameters
    p_true::Vector{Float64}
    
    # Network architecture
    hidden_dim::Int
    n_hidden_layers::Int
    activation::Function
    
    # Training parameters
    adam_epochs::Int
    bfgs_epochs::Int
    iters_per_epoch::Int
    adam_lr::Float64
    bfgs_stepnorm::Float64
    checkpoint_interval::Int
    
    # Loss weights
    compartment_weights::Vector{Float64}
    
    # Random seed
    seed::Int
end

# Default configuration
function get_default_config()
    return TrainingConfig(
        # Data
        [990.0, 10.0, 0.0, 0.0, 0.0, 0.0],  # S, I, Q, R, D, V
        (0.0, 160.0),
        161,
        0.75,
        # Model parameters: β, κ, γ, γq, δ, δq, ν
        [0.3, 0.05, 0.1, 0.08, 0.01, 0.005, 0.02],
        # Network
        32,
        2,
        tanh,
        # Training
        5,
        2,
        100,
        0.01,
        0.001,
        50,
        # Loss weights for S, I, Q, R, D, V
        [1.0, 5.0, 2.0, 1.0, 1.0, 1.0],
        # Seed
        1234
    )
end

# ============================================
# Model Definition
# ============================================
function siqrdv!(du, u, p, t)
    S, I, Q, R, D, V = u
    β, κ, γ, γq, δ, δq, ν = p
    N = S + I + Q + R + V  # Total living population
    
    du[1] = -β * S * I / N - ν * S                    # Susceptible
    du[2] = β * S * I / N - (γ + δ + κ) * I          # Infected
    du[3] = κ * I - (γq + δq) * Q                     # Quarantined
    du[4] = γ * I + γq * Q                            # Recovered
    du[5] = δ * I + δq * Q                            # Deaths
    du[6] = ν * S                                     # Vaccinated
end

# ============================================
# Data Generation Module
# ============================================
function generate_data(config::TrainingConfig)
    println("1. Generating synthetic SIQRDV data...")
    
    t = range(config.tspan[1], config.tspan[2], length=config.n_points)
    
    # Generate ground truth
    prob = ODEProblem(siqrdv!, config.u0, config.tspan, config.p_true)
    ode_data = Array(solve(prob, Tsit5(), saveat=t))
    
    # Split data
    n_train = Int(floor(config.n_points * config.train_split))
    t_train = t[1:n_train]
    t_test = t[n_train+1:end]
    train_data = ode_data[:, 1:n_train]
    test_data = ode_data[:, n_train+1:end]
    
    println("  Generated $(size(ode_data, 2)) time points")
    println("  Training: $(n_train) points (t=$(t_train[1]) to t=$(t_train[end]))")
    println("  Testing: $(length(t_test)) points (t=$(t_test[1]) to t=$(t_test[end]))")
    
    return t, ode_data, t_train, t_test, train_data, test_data
end

# ============================================
# Neural Network Setup Module
# ============================================
function setup_neural_ode(config::TrainingConfig, rng)
    println("\n2. Setting up Neural ODE architecture...")
    
    # Build network layers
    layers = []
    push!(layers, Lux.Dense(6, config.hidden_dim, config.activation))
    for _ in 1:config.n_hidden_layers-1
        push!(layers, Lux.Dense(config.hidden_dim, config.hidden_dim, config.activation))
    end
    push!(layers, Lux.Dense(config.hidden_dim, 6))
    
    dudt_nn = Lux.Chain(layers...)
    
    # Initialize
    p_nn, st_nn = Lux.setup(rng, dudt_nn)
    
    println("  Network architecture: 6 → $(join(fill(config.hidden_dim, config.n_hidden_layers), " → ")) → 6")
    println("  Activation: $(config.activation)")
    println("  Total parameters: $(sum(length, Lux.parameterlength(dudt_nn)))")
    
    return dudt_nn, p_nn, st_nn
end

# ============================================
# Metrics Calculation
# ============================================
function calc_metrics(true_data, pred_data)
    mse = sum(abs2, true_data .- pred_data) / length(true_data)
    mae = sum(abs, true_data .- pred_data) / length(true_data)
    
    # R² score
    ss_tot = sum(abs2, true_data .- mean(true_data))
    ss_res = sum(abs2, true_data .- pred_data)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, mae, r2
end

# ============================================
# Training Module
# ============================================
function train_model(config::TrainingConfig, dudt_nn, p_nn, st_nn, 
                    train_data, test_data, t_train, t_test, u0)
    println("\n3. Setting up checkpoint system...")
    println("\n4. Training Neural ODE with checkpointing...")
    println("  Optimization strategy: ADAM ($(config.adam_epochs) epochs) → BFGS ($(config.bfgs_epochs) epochs)")
    println("  Checkpoints will be saved every $(config.checkpoint_interval) iterations")
    
    # Setup Neural ODE problem
    prob_train = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5(), saveat=t_train)
    
    # Prediction function
    predict_train = p -> Array(prob_train(u0, p, st_nn)[1])
    
    # Loss function
    function loss_train(p)
        pred = predict_train(p)
        loss = 0.0
        for i in 1:6
            loss += config.compartment_weights[i] * sum(abs2, train_data[i,:] .- pred[i,:])
        end
        return loss / length(train_data)
    end
    
    # Initialize checkpoint system
    checkpoints_df = DataFrame(
        epoch = Int[],
        iteration = Int[],
        loss = Float64[],
        train_mse = Float64[],
        train_mae = Float64[],
        train_r2 = Float64[],
        learning_rate = Float64[],
        optimizer = String[],
        time_elapsed = Float64[],
        timestamp = String[]
    )
    
    # Checkpoint saving function
    function save_checkpoint(epoch, iteration, loss, params, optimizer_name, lr, elapsed_time)
        try
            train_pred = predict_train(params)
            train_mse, train_mae, train_r2 = calc_metrics(train_data, train_pred)
            
            push!(checkpoints_df, (
                epoch, iteration, loss, train_mse, train_mae, train_r2,
                lr, optimizer_name, elapsed_time, string(Dates.now())
            ))
            
            CSV.write("training_checkpoints.csv", checkpoints_df)
            
            if loss == minimum(checkpoints_df.loss)
                best_checkpoint = DataFrame(
                    metric = ["best_epoch", "best_iteration", "best_loss", 
                             "best_train_mse", "best_train_mae", "best_train_r2", "time_to_best"],
                    value = [epoch, iteration, loss, train_mse, train_mae, train_r2, elapsed_time]
                )
                CSV.write("best_checkpoint.csv", best_checkpoint)
            end
            return true
        catch e
            println("    Warning: Could not save checkpoint: $e")
            return false
        end
    end
    
    # Training tracking
    losses = Float64[]
    train_times = Float64[]
    start_time = time()
    current_epoch = 1
    
    # Callback
    training_callback = function(state, l)
        push!(losses, l)
        push!(train_times, time() - start_time)
        iteration = length(losses)
        
        if iteration % config.checkpoint_interval == 0 || iteration == 1
            current_lr = current_epoch <= config.adam_epochs ? config.adam_lr : config.bfgs_stepnorm
            optimizer_name = current_epoch <= config.adam_epochs ? "ADAM" : "BFGS"
            
            if save_checkpoint(current_epoch, iteration, l, state.u, 
                              optimizer_name, current_lr, train_times[end])
                println("  ✓ Checkpoint saved at iteration $iteration | Loss = $(round(l, digits=6)) | Time = $(round(train_times[end], digits=2))s")
            end
        end
        
        if iteration % 100 == 0
            println("  Iteration $iteration: Loss = $(round(l, digits=6)) | Time = $(round(train_times[end], digits=2))s")
        end
        
        return false
    end
    
    # Setup optimization
    pinit = ComponentArray(p_nn)
    optf = Optimization.OptimizationFunction((x, p) -> loss_train(x), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, pinit)
    
    # Phase 1: ADAM
    println("\n  Phase 1: ADAM Optimization (Epochs 1-$(config.adam_epochs))")
    println("  " * "─"^40)
    
    local result1
    for epoch in 1:config.adam_epochs
        current_epoch = epoch
        println("  Epoch $epoch:")
        
        if epoch == 1
            result1 = Optimization.solve(optprob, ADAM(config.adam_lr), 
                                        callback=training_callback, 
                                        maxiters=config.iters_per_epoch)
        else
            optprob_cont = remake(optprob, u0=result1.u)
            result1 = Optimization.solve(optprob_cont, ADAM(config.adam_lr), 
                                        callback=training_callback, 
                                        maxiters=config.iters_per_epoch)
        end
        
        println("    Epoch $epoch complete: Loss = $(round(losses[end], digits=6))")
    end
    
    # Phase 2: BFGS
    println("\n  Phase 2: BFGS Refinement (Epochs $(config.adam_epochs+1)-$(config.adam_epochs+config.bfgs_epochs))")
    println("  " * "─"^40)
    
    optprob2 = remake(optprob, u0=result1.u)
    local result2
    
    for epoch in (config.adam_epochs+1):(config.adam_epochs+config.bfgs_epochs)
        current_epoch = epoch
        println("  Epoch $epoch:")
        
        if epoch == config.adam_epochs + 1
            result2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=config.bfgs_stepnorm),
                                       callback=training_callback, 
                                       maxiters=config.iters_per_epoch)
        else
            optprob_cont = remake(optprob2, u0=result2.u)
            result2 = Optimization.solve(optprob_cont, Optim.BFGS(initial_stepnorm=config.bfgs_stepnorm),
                                       callback=training_callback, 
                                       maxiters=config.iters_per_epoch)
        end
        
        println("    Epoch $epoch complete: Loss = $(round(losses[end], digits=6))")
    end
    
    total_time = time() - start_time
    
    println("\n  Training completed!")
    println("  Total epochs: $current_epoch")
    println("  Total iterations: $(length(losses))")
    println("  Final loss: $(round(losses[end], digits=8))")
    println("  Total time: $(round(total_time, digits=2)) seconds")
    
    return result2.u, losses, train_times, checkpoints_df, total_time, current_epoch, predict_train
end

# ============================================
# Evaluation Module
# ============================================
function evaluate_model(final_params, dudt_nn, st_nn, train_data, test_data, 
                       t_train, t_test, predict_train)
    println("\n5. Evaluating model performance...")
    
    # Training predictions
    train_pred = predict_train(final_params)
    
    # Multi-step forecast
    u_last = train_data[:, end]
    prob_test = NeuralODE(dudt_nn, (t_test[1], t_test[end]), Tsit5(), saveat=t_test)
    test_forecast = Array(prob_test(u_last, final_params, st_nn)[1])
    
    # One-step forecast
    function forecast_1step()
        pred = zeros(6, length(t_test))
        pred[:, 1] = test_data[:, 1]
        
        for i in 2:length(t_test)
            u_curr = test_data[:, i-1]
            t_span = (t_test[i-1], t_test[i])
            prob_1s = NeuralODE(dudt_nn, t_span, Tsit5(), saveat=[t_test[i]])
            pred[:, i] = Array(prob_1s(u_curr, final_params, st_nn)[1])[:, end]
        end
        return pred
    end
    
    test_1step = forecast_1step()
    
    # Calculate metrics
    println("\n6. Computing final performance metrics...")
    
    train_mse, train_mae, train_r2 = calc_metrics(train_data, train_pred)
    test_mse_multi, test_mae_multi, test_r2_multi = calc_metrics(test_data, test_forecast)
    test_mse_1step, test_mae_1step, test_r2_1step = calc_metrics(test_data, test_1step)
    
    # Print summary
    println("\nPerformance Summary:")
    println("─"^40)
    println("Training Set:")
    println("  MSE: $(round(train_mse, digits=6))")
    println("  MAE: $(round(train_mae, digits=6))")
    println("  R²:  $(round(train_r2, digits=4))")
    println("\nTest Set (Multi-step):")
    println("  MSE: $(round(test_mse_multi, digits=6))")
    println("  MAE: $(round(test_mae_multi, digits=6))")
    println("  R²:  $(round(test_r2_multi, digits=4))")
    println("\nTest Set (1-step):")
    println("  MSE: $(round(test_mse_1step, digits=6))")
    println("  MAE: $(round(test_mae_1step, digits=6))")
    println("  R²:  $(round(test_r2_1step, digits=4))")
    
    metrics = Dict(
        "train" => (train_mse, train_mae, train_r2),
        "test_multi" => (test_mse_multi, test_mae_multi, test_r2_multi),
        "test_1step" => (test_mse_1step, test_mae_1step, test_r2_1step)
    )
    
    return train_pred, test_forecast, test_1step, metrics
end

# ============================================
# Checkpoint Saving Module
# ============================================
function save_final_checkpoints(config, metrics, losses, total_time, current_epoch, 
                               t_train, t_test, ode_data, train_pred, test_forecast)
    println("\n7. Saving final checkpoint summary...")
    
    # Final checkpoint
    final_checkpoint_df = DataFrame(
        metric = String[],
        value = Float64[]
    )
    
    push!(final_checkpoint_df, ("final_epoch", Float64(current_epoch)))
    push!(final_checkpoint_df, ("total_iterations", Float64(length(losses))))
    push!(final_checkpoint_df, ("final_loss", losses[end]))
    push!(final_checkpoint_df, ("min_loss", minimum(losses)))
    push!(final_checkpoint_df, ("min_loss_iteration", Float64(argmin(losses))))
    
    # Add metrics
    for (key, (mse, mae, r2)) in metrics
        push!(final_checkpoint_df, ("$(key)_mse", mse))
        push!(final_checkpoint_df, ("$(key)_mae", mae))
        push!(final_checkpoint_df, ("$(key)_r2", r2))
    end
    
    push!(final_checkpoint_df, ("training_time_seconds", total_time))
    push!(final_checkpoint_df, ("training_time_minutes", total_time/60))
    push!(final_checkpoint_df, ("n_train_points", Float64(length(t_train))))
    push!(final_checkpoint_df, ("n_test_points", Float64(length(t_test))))
    
    CSV.write("final_checkpoint.csv", final_checkpoint_df)
    println("  Saved: final_checkpoint.csv")
    
    # Predictions
    n_train = length(t_train)
    predictions_df = DataFrame(
        time = vcat(t_train, t_test),
        S_true = ode_data[1, :],
        I_true = ode_data[2, :],
        Q_true = ode_data[3, :],
        R_true = ode_data[4, :],
        D_true = ode_data[5, :],
        V_true = ode_data[6, :],
        S_pred = vcat(train_pred[1, :], test_forecast[1, :]),
        I_pred = vcat(train_pred[2, :], test_forecast[2, :]),
        Q_pred = vcat(train_pred[3, :], test_forecast[3, :]),
        R_pred = vcat(train_pred[4, :], test_forecast[4, :]),
        D_pred = vcat(train_pred[5, :], test_forecast[5, :]),
        V_pred = vcat(train_pred[6, :], test_forecast[6, :]),
        dataset = vcat(fill("train", n_train), fill("test", length(t_test)))
    )
    
    CSV.write("predictions.csv", predictions_df)
    println("  Saved: predictions.csv")
end

# ============================================
# Visualization Module
# ============================================
function create_visualizations(t, ode_data, t_train, t_test, train_pred, 
                              test_forecast, test_1step, losses, checkpoints_df, 
                              config, current_epoch)
    println("\n8. Creating visualizations...")
    
    # Plot 1: Compartment dynamics
    plt_results = plot(layout=(3,2), size=(1200, 900), dpi=100)
    compartment_names = ["Susceptible (S)", "Infected (I)", "Quarantined (Q)", 
                        "Recovered (R)", "Deaths (D)", "Vaccinated (V)"]
    colors = [:blue, :red, :orange, :green, :purple, :brown]
    
    for i in 1:6
        plot!(plt_results[i], t, ode_data[i,:], 
              label="Ground Truth", lw=3, color=:black, alpha=0.7)
        plot!(plt_results[i], t_train, train_pred[i,:], 
              label="Neural ODE (Train)", lw=2, color=colors[i])
        plot!(plt_results[i], t_test, test_forecast[i,:], 
              label="Multi-step Forecast", lw=2, color=colors[i], ls=:dash)
        plot!(plt_results[i], t_test, test_1step[i,:], 
              label="1-step Forecast", lw=2, color=colors[i], ls=:dot, alpha=0.7)
        vline!(plt_results[i], [t_train[end]], color=:gray, ls=:dash, 
               alpha=0.5, label="Train/Test Split", lw=1)
        
        title!(plt_results[i], compartment_names[i], titlefontsize=10)
        xlabel!(plt_results[i], "Time (days)", guidefontsize=8)
        ylabel!(plt_results[i], "Population", guidefontsize=8)
        
        if i == 1
            plot!(plt_results[i], legend=:topright, legendfontsize=6)
        else
            plot!(plt_results[i], legend=false)
        end
        
        plot!(plt_results[i], grid=true, gridalpha=0.3, minorgrid=true, minorgridalpha=0.1)
    end
    
    best_loss_iter = argmin(losses)
    plot!(plt_results, plot_title="Neural ODE: SIQRDV Model | Final Loss: $(round(losses[end], digits=6)) | Best Loss: $(round(minimum(losses), digits=6)) @ Iter $best_loss_iter", 
          plot_titlefontsize=12)
    
    savefig(plt_results, "results.png")
    println("  Saved: results.png")
    
    # Plot 2: Training loss curve
    plt_loss = plot(size=(1000, 600), dpi=100)
    
    plot!(plt_loss, 1:length(losses), losses, 
          label="Training Loss", lw=2, color=:blue, yscale=:log10)
    
    # Checkpoints
    checkpoint_iters = config.checkpoint_interval:config.checkpoint_interval:length(losses)
    checkpoint_losses = [losses[min(i, length(losses))] for i in checkpoint_iters]
    scatter!(plt_loss, checkpoint_iters, checkpoint_losses, 
             label="Checkpoints", color=:red, markersize=4, alpha=0.7)
    
    # Epoch boundaries
    epoch_boundaries = [i * config.iters_per_epoch for i in 1:(config.adam_epochs + config.bfgs_epochs - 1)]
    for (i, boundary) in enumerate(epoch_boundaries)
        if boundary <= length(losses)
            vline!(plt_loss, [boundary], color=:gray, ls=:dash, alpha=0.3, 
                   label= i == 1 ? "Epoch Boundaries" : false, lw=1)
            annotate!(plt_loss, boundary, minimum(losses) * 1.5, 
                     text("Epoch $(i+1)", 8, :gray, rotation=90))
        end
    end
    
    # Optimizer transition
    adam_end = config.adam_epochs * config.iters_per_epoch
    vline!(plt_loss, [adam_end], color=:red, ls=:dash, alpha=0.5, 
           label="ADAM → BFGS", lw=2)
    
    # Smoothed trend
    if length(losses) > 20
        window = 20
        smoothed = [mean(losses[max(1,i-window):min(length(losses),i+window)]) 
                    for i in 1:length(losses)]
        plot!(plt_loss, 1:length(losses), smoothed, 
              label="Smoothed (window=$window)", lw=2, color=:orange, alpha=0.7)
    end
    
    # Best loss
    best_iter = argmin(losses)
    best_loss = minimum(losses)
    scatter!(plt_loss, [best_iter], [best_loss], 
             label="Best Loss: $(round(best_loss, digits=6)) @ Iter $best_iter", 
             color=:green, markersize=8, markershape=:star)
    
    xlabel!(plt_loss, "Iteration")
    ylabel!(plt_loss, "Loss (log scale)")
    title!(plt_loss, "Neural ODE Training Convergence with Checkpoints\n$(nrow(checkpoints_df)) Checkpoints Saved | $current_epoch Epochs Total")
    plot!(plt_loss, legend=:topright, grid=true, gridalpha=0.3, 
          minorgrid=true, minorgridalpha=0.1)
    
    savefig(plt_loss, "loss.png")
    println("  Saved: loss.png")
end

# ============================================
# Summary Report Module
# ============================================
function print_summary_report(checkpoints_df, losses, total_time, current_epoch, metrics)
    println("\n9. Training Summary Report:")
    println("─"^40)
    
    # Checkpoint analysis
    println("\nCheckpoint Statistics:")
    println("  Total checkpoints saved: $(nrow(checkpoints_df))")
    
    if nrow(checkpoints_df) > 0
        best_idx = argmin(checkpoints_df.loss)
        best_cp = checkpoints_df[best_idx, :]
        println("  Best checkpoint:")
        println("    - Epoch: $(best_cp.epoch)")
        println("    - Iteration: $(best_cp.iteration)")
        println("    - Loss: $(round(best_cp.loss, digits=8))")
        println("    - Train R²: $(round(best_cp.train_r2, digits=4))")
        println("    - Time: $(round(best_cp.time_elapsed, digits=2))s")
        
        # Loss progression
        println("\nLoss Progression by Epoch:")
        for epoch in 1:current_epoch
            epoch_checkpoints = filter(row -> row.epoch == epoch, checkpoints_df)
            if nrow(epoch_checkpoints) > 0
                epoch_start_loss = epoch_checkpoints[1, :loss]
                epoch_end_loss = epoch_checkpoints[end, :loss]
                improvement = (epoch_start_loss - epoch_end_loss) / epoch_start_loss * 100
                println("  Epoch $epoch: $(round(epoch_start_loss, digits=6)) → $(round(epoch_end_loss, digits=6)) ($(round(improvement, digits=2))% improvement)")
            end
        end
    end
    
    # Convergence analysis
    recent_losses = losses[max(1, end-19):end]
    loss_std = std(recent_losses)
    loss_trend = (recent_losses[end] - recent_losses[1]) / length(recent_losses)
    
    println("\nConvergence Analysis (last 20 iterations):")
    println("  Mean loss: $(round(mean(recent_losses), digits=8))")
    println("  Std dev:   $(round(loss_std, digits=8))")
    println("  Trend:     $(round(loss_trend, digits=10))")
    println("  Stability: $(loss_std < 1e-4 ? "Excellent ✓" : loss_std < 1e-3 ? "Good ✓" : "Moderate")")
    
    # Final summary
    println("\n" * "="^60)
    println("NEURAL ODE TRAINING COMPLETE")
    println("="^60)
    println("\nOutput files:")
    println("  1. training_checkpoints.csv - All checkpoints with epoch & loss info")
    println("  2. best_checkpoint.csv      - Best model checkpoint details")
    println("  3. final_checkpoint.csv     - Final summary metrics")
    println("  4. predictions.csv          - Full prediction data")
    println("  5. results.png              - Compartment dynamics visualization")
    println("  6. loss.png                 - Training convergence with checkpoints")
    println("\nKey Results:")
    println("  • Total Epochs: $current_epoch")
    println("  • Best Loss: $(round(minimum(losses), digits=8)) at iteration $(argmin(losses))")
    println("  • Final Loss: $(round(losses[end], digits=8))")
    
    test_r2_1step = metrics["test_1step"][3]
    println("  • Best test R² (1-step): $(round(test_r2_1step, digits=4))")
    println("  • Training time: $(round(total_time/60, digits=2)) minutes")
    println("  • Checkpoints saved: $(nrow(checkpoints_df))")
    println("\nCompleted at: $(Dates.now())")
    println("="^60)
end

# ============================================
# MAIN EXECUTION FUNCTION
# ============================================
function run_neural_ode_training(config::TrainingConfig = get_default_config())
    """
    Main function to run the complete Neural ODE training pipeline.
    
    Usage:
        run_neural_ode_training()                    # Run with default config
        run_neural_ode_training(custom_config)       # Run with custom config
    
    This function will:
    1. Generate synthetic SIQRDV data
    2. Setup and train Neural ODE with checkpointing
    3. Evaluate model performance
    4. Save all checkpoints and predictions
    5. Generate visualization plots
    
    Returns:
        Dictionary with results including final parameters, metrics, and file paths
    """
    
    println("NEURAL ODE FOR SIQRDV MODEL")
    println("="^60)
    println("Started at: $(Dates.now())")
    println("\nConfiguration:")
    println("  Random seed: $(config.seed)")
    println("  Network: 6 → $(config.hidden_dim) → $(config.hidden_dim) → 6")
    println("  Training: $(config.adam_epochs) ADAM + $(config.bfgs_epochs) BFGS epochs")
    println("  $(config.iters_per_epoch) iterations per epoch")
    println()
    
    # Set random seed
    Random.seed!(config.seed)
    rng = Random.default_rng()
    
    # Step 1: Generate data
    t, ode_data, t_train, t_test, train_data, test_data = generate_data(config)
    
    # Step 2: Setup neural network
    dudt_nn, p_nn, st_nn = setup_neural_ode(config, rng)
    
    # Step 3: Train model
    final_params, losses, train_times, checkpoints_df, total_time, current_epoch, predict_train = 
        train_model(config, dudt_nn, p_nn, st_nn, train_data, test_data, 
                   t_train, t_test, config.u0)
    
    # Step 4: Evaluate model
    train_pred, test_forecast, test_1step, metrics = 
        evaluate_model(final_params, dudt_nn, st_nn, train_data, test_data, 
                      t_train, t_test, predict_train)
    
    # Step 5: Save checkpoints
    save_final_checkpoints(config, metrics, losses, total_time, current_epoch,
                          t_train, t_test, ode_data, train_pred, test_forecast)
    
    # Step 6: Create visualizations
    create_visualizations(t, ode_data, t_train, t_test, train_pred, 
                         test_forecast, test_1step, losses, checkpoints_df, 
                         config, current_epoch)
    
    # Step 7: Print summary
    print_summary_report(checkpoints_df, losses, total_time, current_epoch, metrics)
    
    # Return results
    return Dict(
        "final_params" => final_params,
        "metrics" => metrics,
        "losses" => losses,
        "checkpoints" => checkpoints_df,
        "files" => [
            "training_checkpoints.csv",
            "best_checkpoint.csv",
            "final_checkpoint.csv",
            "predictions.csv",
            "results.png",
            "loss.png"
        ]
    )
end

# ============================================
# AUTOMATED EXECUTION
# ============================================
# This block runs automatically when the script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("AUTOMATED NEURAL ODE TRAINING SCRIPT")
    println("="^60)
    println("\nStarting automated training pipeline...")
    println("This will generate all outputs automatically.\n")
    
    # Optional: Check and install packages
    # Uncomment the next line if you want automatic package installation
    # check_and_install_packages()
    
    # Run the complete training pipeline
    results = run_neural_ode_training()
    
    println("\n✓ All outputs successfully generated!")
    println("\nGenerated files:")
    for file in results["files"]
        if isfile(file)
            size_kb = round(filesize(file) / 1024, digits=2)
            println("  ✓ $file ($(size_kb) KB)")
        end
    end
    
    println("\nTraining pipeline completed successfully!")
end
