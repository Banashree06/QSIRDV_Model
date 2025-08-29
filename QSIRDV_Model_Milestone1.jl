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
    Q, S, I, R, D, V = u
    β, κ, γ, γq, δ, δq, ν = p
    N = Q + S + I + R + V  # total living population
    
    # Differential equations
    du[1] = κ * I - γq * Q - δq * Q              # dQ/dt (Quarantined)
    du[2] = -β * S * I / N - ν * S               # dS/dt (Susceptible)
    du[3] = β * S * I / N - κ * I - γ * I - δ * I # dI/dt (Infected)
    du[4] = γ * I + γq * Q                       # dR/dt (Recovered)
    du[5] = δ * I + δq * Q                       # dD/dt (Dead)
    du[6] = ν * S                                # dV/dt (Vaccinated)
end

# Model configuration
u0 = [0.0, 990.0, 10.0, 0.0, 0.0, 0.0]  # Initial conditions (Q, S, I, R, D, V)
tspan = (0.0, 160.0)  # Simulation time span (160 days)
compartment_names = ["Q", "S", "I", "R", "D", "V"]
compartment_labels = ["Quarantined", "Susceptible", "Infected", "Recovered", "Dead", "Vaccinated"]

# Define scenarios with different parameter combinations
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
    println("     β=$β, κ=$κ, γ=$γ, γq=$γq, δ=$δ, δq=$δq, ν=$ν")
end

# Simulate trajectories over multiple scenarios
println("\nRunning ODE simulations...")
Random.seed!(1234)  # Set seed for reproducibility

sols = []
datasets_noiseless = []

for (i, scenario) in enumerate(scenarios)
    println("  Solving scenario $i: $(scenario.name)")
    
    prob = ODEProblem(qsirdv!, u0, tspan, scenario.params)
    sol = solve(prob, Tsit5(), saveat=1.0, reltol=1e-8, abstol=1e-10)
    
    push!(sols, sol)
    push!(datasets_noiseless, Array(sol))
    
    println("    Simulation completed successfully")
    println("    Final state: Q=$(round(sol[end][1], digits=2)), S=$(round(sol[end][2], digits=2)), I=$(round(sol[end][3], digits=2)), R=$(round(sol[end][4], digits=2)), D=$(round(sol[end][5], digits=2)), V=$(round(sol[end][6], digits=2))")
end

# Generate noisy datasets
println("\nGenerating noisy datasets...")
datasets_noisy = []

for (i, data) in enumerate(datasets_noiseless)
    println("  Adding noise to scenario $i: $(scenarios[i].name)")
    
    noisy_data = copy(data)
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

# Save CSV files for each scenario (noisy data)
for (i, data) in enumerate(datasets_noisy)
    filename = "qsirdv_scenario$(i)_noisy.csv"
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
    filename = "qsirdv_scenario$(i)_noiseless.csv"
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

# Save train and validation datasets
println("  Saving qsirdv_train_data.csv")
train_df = DataFrame(
    time = 0:(train_split-1),
    Q = train_data[1, :],
    S = train_data[2, :],
    I = train_data[3, :],
    R = train_data[4, :],
    D = train_data[5, :],
    V = train_data[6, :]
)
CSV.write("qsirdv_train_data.csv", train_df)

println("  Saving qsirdv_val_data.csv")
val_df = DataFrame(
    time = train_split:160,
    Q = val_data[1, :],
    S = val_data[2, :],
    I = val_data[3, :],
    R = val_data[4, :],
    D = val_data[5, :],
    V = val_data[6, :]
)
CSV.write("qsirdv_val_data.csv", val_df)

# Save complete dataset bundle in JLD2 format
println("  Saving qsirdv_datasets.jld2")
@save "qsirdv_datasets.jld2" datasets_noiseless datasets_noisy train_data val_data scenarios u0 tspan compartment_names

# Create basic visualization scripts
println("\nCreating visualizations...")

# Time series plot for all compartments (baseline scenario)
println("  Creating compartment curves...")
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
savefig(p1, "qsirdv_timeseries.png")
println("    Saved compartment curves")

# Stacked bar chart showing compartment distribution
println("  Creating stacked bar chart...")
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

bar!(p2, sample_times, data_sampled', 
     label=permutedims(compartment_labels),
     color=permutedims(colors),
     bar_position=:stack,
     bar_width=15)
savefig(p2, "qsirdv_stackedbar.png")
println("    Saved stacked bar chart")

# Scenario comparison plot (Infected population)
println("  Creating scenario comparison plot...")
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
savefig(p3, "qsirdv_infected_comparison.png")
println("    Saved scenario comparison")

# Noisy vs Noiseless comparison
println("  Creating noise comparison plot...")
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

savefig(p4, "qsirdv_noise_comparison.png")
println("    Saved noise comparison")

# Generate summary statistics
println("\nCalculating summary statistics...")

peak_infected = [maximum(data[3, :]) for data in datasets_noiseless]
peak_time = [argmax(data[3, :]) - 1 for data in datasets_noiseless]
total_deaths = [data[5, end] for data in datasets_noiseless]
final_recovered = [data[4, end] for data in datasets_noiseless]
final_vaccinated = [data[6, end] for data in datasets_noiseless]

println("\nScenario Results Summary:")
for (i, scenario) in enumerate(scenarios)
    println("\n  $(scenario.name):")
    println("    Peak Infected: $(round(peak_infected[i], digits=1)) (day $(peak_time[i]))")
    println("    Total Deaths: $(round(total_deaths[i], digits=1))")
    println("    Final Recovered: $(round(final_recovered[i], digits=1))")
    println("    Final Vaccinated: $(round(final_vaccinated[i], digits=1))")
end

println("\n" * "="^50)
println("MILESTONE 1: COMPLETE")
println("="^50)