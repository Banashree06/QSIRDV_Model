
# Milestone 1 :  QSIRDV Model Implementation Code



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

  # neural_ode_siqrdv.jl - Milestone 2: Neural ODE for SIQRDV Model
# Learns epidemic dynamics using neural networks embedded in differential equations
# Trains on synthetic data and evaluates forecasting performance

using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, Plots, ComponentArrays
using CSV, DataFrames, Statistics
using Dates

println("Starting Neural ODE for SIQRDV Model")
println("="^50)

# Set random seed for reproducibility
Random.seed!(1234)
rng = Random.default_rng()

# Model configuration
# Initial conditions (S, I, Q, R, D, V)
u0 = [990.0, 10.0, 0.0, 0.0, 0.0, 0.0]
tspan = (0.0, 160.0)  # Simulation time span (160 days)
t = range(0, 160, length=161)

# True parameters for data generation
# Parameters: (β, κ, γ, γq, δ, δq, ν)
p_true = [0.3, 0.05, 0.1, 0.08, 0.01, 0.005, 0.02]

println("Configuration:")
println("  Initial state: S=990, I=10, Q=0, R=0, D=0, V=0")
println("  Time span: 0 to 160 days")
println("  Parameters: β=0.3, κ=0.05, γ=0.1, γq=0.08, δ=0.01, δq=0.005, ν=0.02")

# Define SIQRDV epidemiological model
function siqrdv!(du, u, p, t)
    """
    SIQRDV Model Compartments:
    S - Susceptible
    I - Infected (infectious)
    Q - Quarantined (isolated infected)
    R - Recovered
    D - Dead
    V - Vaccinated
    
    Parameters:
    β - transmission rate
    κ - quarantine rate
    γ - recovery rate (infected)
    γq - recovery rate (quarantined)
    δ - death rate (infected)
    δq - death rate (quarantined)
    ν - vaccination rate
    """
    S, I, Q, R, D, V = u
    β, κ, γ, γq, δ, δq, ν = p
    N = S + I + Q + R + V  # Total living population
    
    du[1] = -β * S * I / N - ν * S           # dS/dt (Susceptible)
    du[2] = β * S * I / N - (γ + δ + κ) * I  # dI/dt (Infected)
    du[3] = κ * I - (γq + δq) * Q           # dQ/dt (Quarantined)
    du[4] = γ * I + γq * Q                  # dR/dt (Recovered)
    du[5] = δ * I + δq * Q                  # dD/dt (Dead)
    du[6] = ν * S                           # dV/dt (Vaccinated)
end

# Task 1: Generate synthetic training data
println("\nGenerating synthetic SIQRDV data...")

prob = ODEProblem(siqrdv!, u0, tspan, p_true)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

# Create train/test split (75/25)
n_train = 121  # 75% of 161 points
t_train = t[1:n_train]
t_test = t[n_train+1:end]
train_data = ode_data[:, 1:n_train]
test_data = ode_data[:, n_train+1:end]

println("  Generated $(size(ode_data, 2)) time points")
println("  Training data: $(n_train) points (days 0-$(t_train[end]))")
println("  Testing data: $(length(t_test)) points (days $(t_test[1])-160)")

# Task 2: Setup Neural ODE architecture
println("\nSetting up Neural ODE architecture...")

# Define 3-layer neural network
dudt_nn = Lux.Chain(
    Lux.Dense(6, 32, tanh),     # Input layer: 6 → 32
    Lux.Dense(32, 32, tanh),    # Hidden layer: 32 → 32
    Lux.Dense(32, 6)            # Output layer: 32 → 6
)

# Initialize network parameters
p_nn, st_nn = Lux.setup(rng, dudt_nn)

println("  Network architecture: 6 → 32 → 32 → 6")
println("  Activation function: tanh")
println("  Total parameters: $(sum(length, Lux.parameterlength(dudt_nn)))")

# Create Neural ODE problem for training
prob_train = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5(), saveat=t_train)

# Define prediction function
function predict_train(p)
    Array(prob_train(u0, p, st_nn)[1])
end

# Define weighted loss function
function loss_train(p)
    pred = predict_train(p)
    weights = [1.0, 5.0, 2.0, 1.0, 1.0, 1.0]  # S, I, Q, R, D, V weights
    loss = 0.0
    
    for i in 1:6
        compartment_loss = sum(abs2, train_data[i,:] .- pred[i,:])
        loss += weights[i] * compartment_loss
    end
    
    return loss / length(train_data)
end

# Task 3: Setup checkpoint system
println("\nSetting up checkpoint system...")

# Initialize tracking DataFrames
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

# Metrics calculation function
function calc_metrics(true_data, pred_data)
    mse = sum(abs2, true_data .- pred_data) / length(true_data)
    mae = sum(abs, true_data .- pred_data) / length(true_data)
    
    # R² score
    ss_tot = sum(abs2, true_data .- mean(true_data))
    ss_res = sum(abs2, true_data .- pred_data)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, mae, r2
end

# Checkpoint saving function
function save_checkpoint(epoch, iteration, loss, params, optimizer_name, lr, elapsed_time)
    try
        train_pred = predict_train(params)
        train_mse, train_mae, train_r2 = calc_metrics(train_data, train_pred)
        
        push!(checkpoints_df, (
            epoch,
            iteration,
            loss,
            train_mse,
            train_mae,
            train_r2,
            lr,
            optimizer_name,
            elapsed_time,
            string(Dates.now())
        ))
        
        CSV.write("training_checkpoints.csv", checkpoints_df)
        
        if loss == minimum(checkpoints_df.loss)
            best_checkpoint = DataFrame(
                metric = ["best_epoch", "best_iteration", "best_loss", "best_train_mse", 
                         "best_train_mae", "best_train_r2", "time_to_best"],
                value = [epoch, iteration, loss, train_mse, train_mae, train_r2, elapsed_time]
            )
            CSV.write("best_checkpoint.csv", best_checkpoint)
        end
        
        return true
    catch e
        println("    Warning: Could not save checkpoint at iteration $iteration: $e")
        return false
    end
end

# Task 4: Training with two-phase optimization
println("\nTraining Neural ODE...")
println("  Optimization strategy: ADAM → BFGS")
println("  Checkpoints saved every 50 iterations")

# Initialize tracking variables
losses = Float64[]
train_times = Float64[]
start_time = time()
current_epoch = 1
checkpoint_interval = 50

# Callback function for training
training_callback = function(state, l)
    push!(losses, l)
    push!(train_times, time() - start_time)
    iteration = length(losses)
    
    # Save checkpoint at intervals
    if iteration % checkpoint_interval == 0 || iteration == 1
        current_lr = iteration <= 500 ? 0.01 : 0.001
        optimizer_name = iteration <= 500 ? "ADAM" : "BFGS"
        
        if save_checkpoint(current_epoch, iteration, l, state.u, 
                          optimizer_name, current_lr, train_times[end])
            println("  Checkpoint saved: iteration $iteration, loss=$(round(l, digits=6))")
        end
    end
    
    # Progress update
    if iteration % 100 == 0
        println("  Iteration $iteration: Loss=$(round(l, digits=6)), Time=$(round(train_times[end], digits=1))s")
    end
    
    return false
end

# Initialize optimization
pinit = ComponentArray(p_nn)
optf = Optimization.OptimizationFunction((x, p) -> loss_train(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, pinit)

# Phase 1: ADAM optimization (500 iterations across 5 epochs)
println("\n  Phase 1: ADAM Optimization")
adam_iters_per_epoch = 100

for epoch in 1:5
    current_epoch = epoch
    println("    Epoch $epoch starting...")
    
    if epoch == 1
        global result1 = Optimization.solve(optprob, ADAM(0.01), 
                                           callback=training_callback, 
                                           maxiters=adam_iters_per_epoch)
    else
        optprob_cont = remake(optprob, u0=result1.u)
        global result1 = Optimization.solve(optprob_cont, ADAM(0.01), 
                                           callback=training_callback, 
                                           maxiters=adam_iters_per_epoch)
    end
    
    println("    Epoch $epoch complete: Loss=$(round(losses[end], digits=6))")
end

# Phase 2: BFGS refinement (200 iterations across 2 epochs)
println("\n  Phase 2: BFGS Refinement")
optprob2 = remake(optprob, u0=result1.u)
bfgs_iters_per_epoch = 100

for epoch in 6:7
    current_epoch = epoch
    println("    Epoch $epoch starting...")
    
    if epoch == 6
        global result2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.001),
                                           callback=training_callback, 
                                           maxiters=bfgs_iters_per_epoch)
    else
        optprob_cont = remake(optprob2, u0=result2.u)
        global result2 = Optimization.solve(optprob_cont, Optim.BFGS(initial_stepnorm=0.001),
                                           callback=training_callback, 
                                           maxiters=bfgs_iters_per_epoch)
    end
    
    println("    Epoch $epoch complete: Loss=$(round(losses[end], digits=6))")
end

final_params = result2.u
total_time = time() - start_time

println("\nTraining completed successfully")
println("  Total epochs: $current_epoch")
println("  Total iterations: $(length(losses))")
println("  Final loss: $(round(losses[end], digits=8))")
println("  Training time: $(round(total_time, digits=1)) seconds")

# Task 5: Model evaluation and forecasting
println("\nEvaluating model performance...")

# Training set predictions
train_pred = predict_train(final_params)

# Multi-step forecast
u_last = train_data[:, end]
prob_test = NeuralODE(dudt_nn, (t_test[1], t_test[end]), Tsit5(), saveat=t_test)
test_forecast = Array(prob_test(u_last, final_params, st_nn)[1])

# One-step ahead forecast
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

# Calculate performance metrics
train_mse, train_mae, train_r2 = calc_metrics(train_data, train_pred)
test_mse_multi, test_mae_multi, test_r2_multi = calc_metrics(test_data, test_forecast)
test_mse_1step, test_mae_1step, test_r2_1step = calc_metrics(test_data, test_1step)

println("\nPerformance metrics:")
println("  Training set:")
println("    MSE: $(round(train_mse, digits=6))")
println("    MAE: $(round(train_mae, digits=6))")
println("    R²:  $(round(train_r2, digits=4))")
println("  Test set (multi-step):")
println("    MSE: $(round(test_mse_multi, digits=6))")
println("    MAE: $(round(test_mae_multi, digits=6))")
println("    R²:  $(round(test_r2_multi, digits=4))")
println("  Test set (1-step):")
println("    MSE: $(round(test_mse_1step, digits=6))")
println("    MAE: $(round(test_mae_1step, digits=6))")
println("    R²:  $(round(test_r2_1step, digits=4))")

# Task 6: Export results
println("\nExporting results...")

# Create data directory if needed
if !isdir("data")
    mkdir("data")
    println("  Created 'data' directory")
end

# Save final checkpoint summary
println("  Saving data/final_checkpoint.csv")
final_checkpoint_df = DataFrame(
    metric = String[],
    value = Float64[]
)

push!(final_checkpoint_df, ("final_epoch", Float64(current_epoch)))
push!(final_checkpoint_df, ("total_iterations", Float64(length(losses))))
push!(final_checkpoint_df, ("final_loss", losses[end]))
push!(final_checkpoint_df, ("min_loss", minimum(losses)))
push!(final_checkpoint_df, ("min_loss_iteration", Float64(argmin(losses))))
push!(final_checkpoint_df, ("train_mse", train_mse))
push!(final_checkpoint_df, ("train_mae", train_mae))
push!(final_checkpoint_df, ("train_r2", train_r2))
push!(final_checkpoint_df, ("test_mse_multi", test_mse_multi))
push!(final_checkpoint_df, ("test_mae_multi", test_mae_multi))
push!(final_checkpoint_df, ("test_r2_multi", test_r2_multi))
push!(final_checkpoint_df, ("test_mse_1step", test_mse_1step))
push!(final_checkpoint_df, ("test_mae_1step", test_mae_1step))
push!(final_checkpoint_df, ("test_r2_1step", test_r2_1step))
push!(final_checkpoint_df, ("training_time_seconds", total_time))
push!(final_checkpoint_df, ("training_time_minutes", total_time/60))
push!(final_checkpoint_df, ("n_train_points", Float64(n_train)))
push!(final_checkpoint_df, ("n_test_points", Float64(length(t_test))))

CSV.write("data/final_checkpoint.csv", final_checkpoint_df)

# Save predictions
println("  Saving data/predictions.csv")
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

CSV.write("data/predictions.csv", predictions_df)

# Task 7: Create visualizations
println("\nCreating visualizations...")

# Create figures directory if needed
if !isdir("figures")
    mkdir("figures")
    println("  Created 'figures' directory")
end

# 1. Compartment dynamics plot
println("  Creating compartment dynamics plot...")
try
    plt_results = plot(layout=(3,2), size=(1200, 900), dpi=100)
    compartment_names = ["Susceptible (S)", "Infected (I)", "Quarantined (Q)", 
                        "Recovered (R)", "Deaths (D)", "Vaccinated (V)"]
    colors = [:blue, :red, :orange, :green, :purple, :brown]
    
    for i in 1:6
        # Ground truth
        plot!(plt_results[i], t, ode_data[i,:], 
              label="Ground Truth", lw=3, color=:black, alpha=0.7)
        
        # Training fit
        plot!(plt_results[i], t_train, train_pred[i,:], 
              label="Neural ODE (Train)", lw=2, color=colors[i])
        
        # Multi-step forecast
        plot!(plt_results[i], t_test, test_forecast[i,:], 
              label="Multi-step Forecast", lw=2, color=colors[i], ls=:dash)
        
        # 1-step forecast
        plot!(plt_results[i], t_test, test_1step[i,:], 
              label="1-step Forecast", lw=2, color=colors[i], ls=:dot, alpha=0.7)
        
        # Train/test split line
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
        
        plot!(plt_results[i], grid=true, gridalpha=0.3)
    end
    
    savefig(plt_results, "figures/neural_ode_results.png")
    println("  Saved compartment dynamics plot")
catch e
    println("  ERROR creating results plot: $e")
end

# 2. Training loss curve
println("  Creating loss curve plot...")
try
    plt_loss = plot(size=(1000, 600), dpi=100)
    
    # Main loss curve
    plot!(plt_loss, 1:length(losses), losses, 
          label="Training Loss", lw=2, color=:blue, yscale=:log10)
    
    # Mark checkpoints
    checkpoint_iters = checkpoint_interval:checkpoint_interval:length(losses)
    checkpoint_losses = [losses[min(i, length(losses))] for i in checkpoint_iters]
    scatter!(plt_loss, checkpoint_iters, checkpoint_losses, 
             label="Checkpoints", color=:red, markersize=4, alpha=0.7)
    
    # Mark optimizer transition
    vline!(plt_loss, [500], color=:red, ls=:dash, alpha=0.5, 
           label="ADAM → BFGS", lw=2)
    
    # Add smoothed trend
    if length(losses) > 20
        window = 20
        smoothed = [mean(losses[max(1,i-window):min(length(losses),i+window)]) 
                    for i in 1:length(losses)]
        plot!(plt_loss, 1:length(losses), smoothed, 
              label="Smoothed", lw=2, color=:orange, alpha=0.7)
    end
    
    xlabel!(plt_loss, "Iteration")
    ylabel!(plt_loss, "Loss (log scale)")
    title!(plt_loss, "Neural ODE Training Convergence")
    plot!(plt_loss, legend=:topright, grid=true, gridalpha=0.3)
    
    savefig(plt_loss, "figures/training_loss.png")
    println("  Saved loss curve plot")
catch e
    println("  ERROR creating loss plot: $e")
end

# Generate summary statistics
println("\nCalculating summary statistics...")

best_loss_iter = argmin(losses)
println("  Best loss: $(round(minimum(losses), digits=8)) at iteration $best_loss_iter")
println("  Final loss: $(round(losses[end], digits=8))")
println("  Loss reduction: $(round((1 - losses[end]/losses[1])*100, digits=2))%")
println("  Average training speed: $(round(length(losses)/total_time, digits=1)) iterations/second")

println("\n" * "="^50)
println("Neural ODE training completed successfully!")
println("Generated files:")
println("  - data/final_checkpoint.csv")
println("  - data/predictions.csv")
println("  - training_checkpoints.csv")
println("  - best_checkpoint.csv")
println("  - figures/neural_ode_results.png")
println("  - figures/training_loss.png")
- 4 PNG visualization files

