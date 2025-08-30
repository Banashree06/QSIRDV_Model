# ============================================================================
# Milestone 1 :  QSIRDV Model Implementation Code
# ============================================================================


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

  # Neural ODE for SIQRDV Epidemic Model - Concise Version
# Learns epidemic dynamics using neural networks embedded in differential equations

using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, Plots, ComponentArrays, CSV, DataFrames, Statistics, Dates

Random.seed!(1234)
rng = Random.default_rng()

println("NEURAL ODE FOR SIQRDV MODEL")
println("="^60)
println("Started: $(Dates.now())")

# ============================================
# 1. Data Generation
# ============================================
u0 = [990.0, 10.0, 0.0, 0.0, 0.0, 0.0]  # [S, I, Q, R, D, V]
tspan = (0.0, 160.0)
t = range(0, 160, length=161)
p_true = [0.3, 0.05, 0.1, 0.08, 0.01, 0.005, 0.02]  # [β, κ, γ, γq, δ, δq, ν]

function siqrdv!(du, u, p, t)
    S, I, Q, R, D, V = u
    β, κ, γ, γq, δ, δq, ν = p
    N = S + I + Q + R + V
    
    du[1] = -β * S * I / N - ν * S
    du[2] = β * S * I / N - (γ + δ + κ) * I
    du[3] = κ * I - (γq + δq) * Q
    du[4] = γ * I + γq * Q
    du[5] = δ * I + δq * Q
    du[6] = ν * S
end

# Generate data
prob = ODEProblem(siqrdv!, u0, tspan, p_true)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

# Split data (75/25)
n_train = 121
t_train, t_test = t[1:n_train], t[n_train+1:end]
train_data, test_data = ode_data[:, 1:n_train], ode_data[:, n_train+1:end]

println("✓ Data: $(n_train) train, $(length(t_test)) test points")

# ============================================
# 2. Neural Network Setup
# ============================================
dudt_nn = Lux.Chain(
    Lux.Dense(6, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 6)
)

p_nn, st_nn = Lux.setup(rng, dudt_nn)
prob_train = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5(), saveat=t_train)

predict_train = p -> Array(prob_train(u0, p, st_nn)[1])

function loss_train(p)
    pred = predict_train(p)
    weights = [1.0, 5.0, 2.0, 1.0, 1.0, 1.0]  # Emphasize I and Q
    sum(weights[i] * sum(abs2, train_data[i,:] .- pred[i,:]) for i in 1:6) / length(train_data)
end

println("✓ Network: 6 → 32 → 32 → 6 ($(sum(length, Lux.parameterlength(dudt_nn))) params)")

# ============================================
# 3. Training with Checkpoints
# ============================================
losses = Float64[]
checkpoints = DataFrame()
start_time = time()

function save_checkpoint(iter, loss, params, opt_name)
    pred = predict_train(params)
    mse = sum(abs2, train_data .- pred) / length(train_data)
    r2 = 1 - sum(abs2, train_data .- pred) / sum(abs2, train_data .- mean(train_data))
    
    push!(checkpoints, (iter=iter, loss=loss, mse=mse, r2=r2, optimizer=opt_name))
    
    if iter % 50 == 0
        CSV.write("checkpoints.csv", checkpoints)
        println("  Iter $iter: Loss=$(round(loss,digits=5)), R²=$(round(r2,digits=3))")
    end
end

callback = function(state, l)
    push!(losses, l)
    iter = length(losses)
    opt = iter <= 500 ? "ADAM" : "BFGS"
    save_checkpoint(iter, l, state.u, opt)
    return false
end

# Training
println("\nTraining...")
pinit = ComponentArray(p_nn)
optf = Optimization.OptimizationFunction((x, p) -> loss_train(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, pinit)

# Phase 1: ADAM (500 iterations)
result1 = Optimization.solve(optprob, ADAM(0.01), callback=callback, maxiters=500)

# Phase 2: BFGS (200 iterations)
optprob2 = remake(optprob, u0=result1.u)
result2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.001), 
                            callback=callback, maxiters=200)

final_params = result2.u
println("✓ Training complete in $(round(time()-start_time, digits=1))s")

# ============================================
# 4. Evaluation
# ============================================
# Training predictions
train_pred = predict_train(final_params)

# Multi-step forecast
prob_test = NeuralODE(dudt_nn, (t_test[1], t_test[end]), Tsit5(), saveat=t_test)
test_forecast = Array(prob_test(train_data[:, end], final_params, st_nn)[1])

# One-step forecast
test_1step = zeros(6, length(t_test))
test_1step[:, 1] = test_data[:, 1]
for i in 2:length(t_test)
    prob_1s = NeuralODE(dudt_nn, (t_test[i-1], t_test[i]), Tsit5(), saveat=[t_test[i]])
    test_1step[:, i] = Array(prob_1s(test_data[:, i-1], final_params, st_nn)[1])[:, end]
end

# Metrics
calc_r2 = (true_d, pred_d) -> 1 - sum(abs2, true_d .- pred_d) / sum(abs2, true_d .- mean(true_d))

println("\nResults:")
println("  Train R²: $(round(calc_r2(train_data, train_pred), digits=4))")
println("  Test R² (multi): $(round(calc_r2(test_data, test_forecast), digits=4))")
println("  Test R² (1-step): $(round(calc_r2(test_data, test_1step), digits=4))")

# ============================================
# 5. Save Results
# ============================================
# Save predictions
predictions_df = DataFrame(
    time = vcat(t_train, t_test),
    S_true = ode_data[1, :], I_true = ode_data[2, :], Q_true = ode_data[3, :],
    R_true = ode_data[4, :], D_true = ode_data[5, :], V_true = ode_data[6, :],
    S_pred = vcat(train_pred[1, :], test_forecast[1, :]),
    I_pred = vcat(train_pred[2, :], test_forecast[2, :]),
    Q_pred = vcat(train_pred[3, :], test_forecast[3, :]),
    R_pred = vcat(train_pred[4, :], test_forecast[4, :]),
    D_pred = vcat(train_pred[5, :], test_forecast[5, :]),
    V_pred = vcat(train_pred[6, :], test_forecast[6, :]),
    dataset = vcat(fill("train", n_train), fill("test", length(t_test)))
)
CSV.write("predictions.csv", predictions_df)

# Save summary
summary = DataFrame(
    metric = ["train_r2", "test_r2_multi", "test_r2_1step", "final_loss", 
              "best_loss", "iterations", "time_seconds"],
    value = [calc_r2(train_data, train_pred), calc_r2(test_data, test_forecast),
             calc_r2(test_data, test_1step), losses[end], minimum(losses),
             length(losses), time()-start_time]
)
CSV.write("summary.csv", summary)

# ============================================
# 6. Visualization
# ============================================
# Plot 1: Results
plt = plot(layout=(3,2), size=(1000, 750))
names = ["S", "I", "Q", "R", "D", "V"]
colors = [:blue, :red, :orange, :green, :purple, :brown]

for i in 1:6
    plot!(plt[i], t, ode_data[i,:], label="Truth", lw=2.5, color=:black)
    plot!(plt[i], t_train, train_pred[i,:], label="Train", lw=2, color=colors[i])
    plot!(plt[i], t_test, test_forecast[i,:], label="Forecast", lw=2, 
          color=colors[i], ls=:dash)
    vline!(plt[i], [t_train[end]], color=:gray, ls=:dash, alpha=0.5, label=false)
    title!(plt[i], names[i])
    plot!(plt[i], legend=(i==1 ? :best : false))
end
savefig(plt, "results.png")

# Plot 2: Loss
plt_loss = plot(losses, xlabel="Iteration", ylabel="Loss", 
                title="Training ($(length(losses)) iterations)", 
                lw=2, yscale=:log10, label="Loss")
vline!([500], color=:red, ls=:dash, label="ADAM→BFGS")
savefig(plt_loss, "loss.png")

println("\n✓ Saved: predictions.csv, summary.csv, checkpoints.csv, results.png, loss.png")
println("="^60)
println("Completed: $(Dates.now())")
- Console output showing simulation progress and final statistics

