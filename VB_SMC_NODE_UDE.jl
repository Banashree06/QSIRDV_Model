"""
UNIFIED FOUR-MODEL COMPARISON: VB vs SMC-ABC vs Neural ODE vs UDE
Complete Analysis with Training Times and Comprehensive Visualizations
"""

using DifferentialEquations, Random, Statistics, Plots
using Distributions, StatsBase, LinearAlgebra, Optim
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Lux, DiffEqFlux, ComponentArrays
using SciMLSensitivity
using CSV, DataFrames

Random.seed!(2024)
rng = Random.default_rng()

println("="^80)
println("UNIFIED FOUR-MODEL COMPARISON")
println("VB vs SMC-ABC vs Neural ODE vs UDE")
println("="^80)

# ============================================
# SHARED CONFIGURATION
# ============================================
const N_POPULATION = 1000.0
const N_DAYS = 100
const N_POINTS = 51  # Consistent for all models

# ============================================
# UNIFIED SIQRDV MODEL
# ============================================
function SIQRDV_unified!(du, u, p, t)
    S, I, Q, R, D, V = u
    β, κ, γ, γq, δ, δq, ν = p
    
    N_living = S + I + Q + R + V
    if N_living < 1
        du .= 0
        return
    end
    
    # Time-dependent transmission
    β_eff = β * exp(-0.01 * t)
    
    # Policy implementation
    κ_eff = t > 20 ? κ * 1.5 : κ
    
    # Healthcare strain
    strain = (I + Q) > 30 ? 1.3 : 1.0
    
    # Vaccination
    ν_eff = t > 40 ? ν : 0.0
    
    du[1] = -β_eff * S * I / N_living - ν_eff * S / N_POPULATION
    du[2] = β_eff * S * I / N_living - (γ + δ * strain + κ_eff) * I
    du[3] = κ_eff * I - (γq + δq) * Q
    du[4] = γ * I + γq * Q
    du[5] = δ * strain * I + δq * Q
    du[6] = ν_eff * S / N_POPULATION
end

# ============================================
# DATA GENERATION (SHARED FOR ALL MODELS)
# ============================================
println("\nGENERATING UNIFIED DATA")
println("-"^60)

u0 = [980.0, 20.0, 0.0, 0.0, 0.0, 0.0]
true_params = [0.25, 0.045, 0.115, 0.085, 0.005, 0.0025, 0.0125]
t_full = range(0.0, N_DAYS, length=N_POINTS)
n_train = 35  # 70% for training

# Generate truth data
prob_true = ODEProblem(SIQRDV_unified!, u0, (0.0, N_DAYS), true_params)
sol_true = solve(prob_true, Tsit5(), saveat=t_full, abstol=1e-6, reltol=1e-5)
truth_full = Array(sol_true)

# Add noise
noisy_full = copy(truth_full)
noise_factors = [0.02, 0.05, 0.04, 0.03, 0.01, 0.025]
for i in 1:6
    noise = noise_factors[i] * sqrt.(abs.(truth_full[i, :]) .+ 1) .* randn(length(t_full))
    noisy_full[i, :] += noise
end
noisy_full = max.(0.0, noisy_full)

# Split data
t_train = t_full[1:n_train]
t_test = t_full[n_train:end]
train_data = noisy_full[:, 1:n_train]
test_data = noisy_full[:, n_train:end]
truth_train = truth_full[:, 1:n_train]
truth_test = truth_full[:, n_train:end]

println("✓ Data generated: $(n_train) train, $(length(t_test)) test points")
println("  Peak infected: $(round(maximum(truth_full[2,:]), digits=1))")

# ============================================
# MODEL 1: VARIATIONAL BAYES
# ============================================
println("\n" * "="^60)
println("MODEL 1: VARIATIONAL BAYES")
println("="^60)

function variational_bayes_unified(data, t_points, u0; max_iter=100)
    n_params = 7
    bounds = [(0.15, 0.35), (0.02, 0.08), (0.08, 0.15), (0.06, 0.12), 
              (0.002, 0.01), (0.001, 0.005), (0.005, 0.02)]
    
    function objective(params)
        try
            for i in 1:n_params
                if params[i] < bounds[i][1] || params[i] > bounds[i][2]
                    return 1e6
                end
            end
            
            prob = ODEProblem(SIQRDV_unified!, u0, (t_points[1], t_points[end]), params)
            sol = solve(prob, Tsit5(), saveat=t_points, abstol=1e-4, reltol=1e-3, maxiters=5000)
            
            if sol.retcode != ReturnCode.Success
                return 1e6
            end
            
            pred = Array(sol)
            weights = [1.0, 5.0, 4.0, 2.0, 8.0, 3.0]
            loss = sum(weights[i] * sum(abs2, pred[i, :] .- data[i, :]) for i in 1:6)
            return loss
        catch
            return 1e6
        end
    end
    
    println("  Running VB optimization...")
    best_params = nothing
    best_loss = Inf
    
    starting_points = [
        [0.25, 0.04, 0.11, 0.09, 0.006, 0.003, 0.012],
        [0.22, 0.06, 0.13, 0.08, 0.004, 0.002, 0.015],
        [0.28, 0.05, 0.10, 0.10, 0.008, 0.004, 0.010]
    ]
    
    for (idx, init_params) in enumerate(starting_points)
        try
            result = optimize(objective, init_params, BFGS(), 
                            Optim.Options(iterations=max_iter, show_trace=false))
            
            if Optim.minimum(result) < best_loss
                best_loss = Optim.minimum(result)
                best_params = Optim.minimizer(result)
            end
            println("    Start $idx: Loss = $(round(Optim.minimum(result), digits=1))")
        catch
            println("    Start $idx: Failed")
        end
    end
    
    return best_params !== nothing ? best_params : starting_points[1]
end

vb_time = @elapsed begin
    vb_params = variational_bayes_unified(train_data, t_train, u0)
end
println("✓ VB completed in $(round(vb_time, digits=2)) seconds")

# VB predictions
prob_vb_train = ODEProblem(SIQRDV_unified!, u0, (t_train[1], t_train[end]), vb_params)
sol_vb_train = solve(prob_vb_train, Tsit5(), saveat=t_train)
vb_train_pred = Array(sol_vb_train)

prob_vb_test = ODEProblem(SIQRDV_unified!, train_data[:, end], (t_test[1], t_test[end]), vb_params)
sol_vb_test = solve(prob_vb_test, Tsit5(), saveat=t_test)
vb_test_pred = Array(sol_vb_test)

# ============================================
# MODEL 2: SMC-ABC
# ============================================
println("\n" * "="^60)
println("MODEL 2: SMC-ABC")
println("="^60)

function smc_abc_unified(data, t_points, u0, n_particles=60, n_generations=8)
    n_params = 7
    bounds = [(0.15, 0.35), (0.02, 0.08), (0.08, 0.15), (0.06, 0.12), 
              (0.002, 0.01), (0.001, 0.005), (0.005, 0.02)]
    
    function compute_distance(params)
        try
            prob = ODEProblem(SIQRDV_unified!, u0, (t_points[1], t_points[end]), params)
            sol = solve(prob, Tsit5(), saveat=t_points, abstol=1e-4, reltol=1e-3, maxiters=5000)
            
            if sol.retcode != ReturnCode.Success
                return Inf
            end
            
            pred = Array(sol)
            weights = [1.0, 5.0, 4.0, 2.0, 8.0, 3.0]
            total_dist = sum(weights[i] * mean(abs.(pred[i, :] .- data[i, :])) for i in 1:6)
            return total_dist / sum(weights)
        catch
            return Inf
        end
    end
    
    println("  Running SMC-ABC...")
    
    # Initialize
    population = zeros(n_params, n_particles)
    distances = zeros(n_particles)
    center = [0.25, 0.045, 0.115, 0.085, 0.005, 0.0025, 0.0125]
    
    for i in 1:n_particles
        for j in 1:n_params
            if i <= n_particles ÷ 2
                population[j, i] = center[j] * (1 + 0.15 * randn())
            else
                population[j, i] = bounds[j][1] + (bounds[j][2] - bounds[j][1]) * rand()
            end
            population[j, i] = clamp(population[j, i], bounds[j][1], bounds[j][2])
        end
        distances[i] = compute_distance(population[:, i])
    end
    
    # SMC iterations
    for gen in 1:n_generations
        valid_dists = filter(isfinite, distances)
        if length(valid_dists) < 5
            break
        end
        
        threshold = quantile(valid_dists, 0.7)
        weights = ones(n_particles)
        weights[.!isfinite.(distances)] .= 0
        weights = weights / sum(weights)
        
        new_population = zeros(n_params, n_particles)
        new_distances = zeros(n_particles)
        accepted = 0
        attempts = 0
        
        while accepted < n_particles && attempts < n_particles * 50
            attempts += 1
            parent_idx = sample(1:n_particles, Weights(weights))
            candidate = zeros(n_params)
            
            for j in 1:n_params
                noise = 0.05 * (bounds[j][2] - bounds[j][1]) * randn()
                candidate[j] = clamp(population[j, parent_idx] + noise, bounds[j][1], bounds[j][2])
            end
            
            dist = compute_distance(candidate)
            if dist < threshold
                accepted += 1
                new_population[:, accepted] = candidate
                new_distances[accepted] = dist
            end
        end
        
        if accepted > 0
            population = new_population
            distances = new_distances
            best_dist = minimum(distances[isfinite.(distances)])
            println("    Gen $gen: Accepted = $accepted, Best = $(round(best_dist, digits=3))")
        end
    end
    
    best_idx = argmin(distances)
    return population[:, best_idx]
end

smc_time = @elapsed begin
    smc_params = smc_abc_unified(train_data, t_train, u0)
end
println("✓ SMC-ABC completed in $(round(smc_time, digits=2)) seconds")

# SMC predictions
prob_smc_train = ODEProblem(SIQRDV_unified!, u0, (t_train[1], t_train[end]), smc_params)
sol_smc_train = solve(prob_smc_train, Tsit5(), saveat=t_train)
smc_train_pred = Array(sol_smc_train)

prob_smc_test = ODEProblem(SIQRDV_unified!, train_data[:, end], (t_test[1], t_test[end]), smc_params)
sol_smc_test = solve(prob_smc_test, Tsit5(), saveat=t_test)
smc_test_pred = Array(sol_smc_test)

# ============================================
# MODEL 3: NEURAL ODE
# ============================================
println("\n" * "="^60)
println("MODEL 3: NEURAL ODE")
println("="^60)

# Neural network for NODE
dudt_nn = Lux.Chain(
    Lux.Dense(6, 24, tanh),
    Lux.Dense(24, 24, tanh),
    Lux.Dense(24, 6)
)

p_nn, st_nn = Lux.setup(rng, dudt_nn)

println("  Training Neural ODE...")
node_time = @elapsed begin
    prob_node = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5(), saveat=t_train)
    
    function predict_node_train(p)
        Array(prob_node(u0, p, st_nn)[1])
    end
    
    function loss_node(p)
        pred = predict_node_train(p)
        weights = [1.0, 5.0, 2.0, 1.0, 1.0, 1.0]
        loss = sum(weights[i] * sum(abs2, train_data[i,:] .- pred[i,:]) for i in 1:6)
        return loss / length(train_data)
    end
    
    p_initial_node = ComponentArray(p_nn)
    
    # Callback for tracking
    node_losses = Float64[]
    function callback_node(state, loss_val)
        push!(node_losses, loss_val)
        if length(node_losses) % 100 == 0
            println("    NODE Iter $(length(node_losses)): Loss = $(round(loss_val, digits=5))")
        end
        return false
    end
    
    optf_node = Optimization.OptimizationFunction((x,p) -> loss_node(x), 
                                                  Optimization.AutoZygote())
    optprob_node = Optimization.OptimizationProblem(optf_node, p_initial_node)
    
    # Training with ADAM
    res_node_1 = Optimization.solve(optprob_node, ADAM(0.01), 
                                    callback=callback_node, maxiters=400)
    
    # Fine-tuning with BFGS
    optprob_node_2 = remake(optprob_node, u0=res_node_1.u)
    res_node_2 = Optimization.solve(optprob_node_2, 
                                    Optim.BFGS(initial_stepnorm=0.001),
                                    callback=callback_node, maxiters=100)
    
    global trained_params_node = res_node_2.u
end

println("✓ Neural ODE completed in $(round(node_time, digits=2)) seconds")

# NODE predictions
prob_node_train = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5(), saveat=t_train)
node_train_pred = Array(prob_node_train(u0, trained_params_node, st_nn)[1])

prob_node_test = NeuralODE(dudt_nn, (t_test[1], t_test[end]), Tsit5(), saveat=t_test)
node_test_pred = Array(prob_node_test(train_data[:, end], trained_params_node, st_nn)[1])

# ============================================
# MODEL 4: UDE
# ============================================
println("\n" * "="^60)
println("MODEL 4: UNIVERSAL DIFFERENTIAL EQUATIONS (UDE)")
println("="^60)

# Neural networks for UDE
NN_beta = Lux.Chain(
    Lux.Dense(2, 12, tanh),
    Lux.Dense(12, 8, tanh),
    Lux.Dense(8, 1, softplus)
)

NN_kappa = Lux.Chain(
    Lux.Dense(1, 8, tanh),
    Lux.Dense(8, 1, sigmoid)
)

p_beta, st_beta = Lux.setup(rng, NN_beta)
p_kappa, st_kappa = Lux.setup(rng, NN_kappa)

params_init_ude = Float64[0.11, 0.085, 0.007, 0.0035, 0.014]

function siqrdv_ude_unified!(du, u, p, t)
    S, I, Q, R, D, V = u
    N_living = max(S + I + Q + R + V, 1.0)
    
    # Neural network for β
    S_norm = S / N_POPULATION
    I_norm = I / 100.0
    β_input = [S_norm, I_norm]
    β_nn = NN_beta(β_input, p.nn_beta, st_beta)[1][1]
    β = 0.1 + 0.3 * β_nn
    
    # Neural network for κ
    κ_input = [I_norm]
    κ_nn = NN_kappa(κ_input, p.nn_kappa, st_kappa)[1][1]
    κ = 0.03 + 0.07 * κ_nn
    
    # Other parameters
    γ = abs(p.params[1])
    γq = abs(p.params[2])
    δ = abs(p.params[3])
    δq = abs(p.params[4])
    ν = abs(p.params[5])
    
    # Time-dependent effects
    β_eff = β * exp(-0.01 * t)
    κ_eff = t > 20 ? κ * 1.5 : κ
    strain = (I + Q) > 30 ? 1.3 : 1.0
    ν_eff = t > 40 ? ν : 0.0
    
    # Dynamics
    du[1] = -β_eff * S * I / N_living - ν_eff * S / N_POPULATION
    du[2] = β_eff * S * I / N_living - (γ + δ * strain + κ_eff) * I
    du[3] = κ_eff * I - (γq + δq) * Q
    du[4] = γ * I + γq * Q
    du[5] = δ * strain * I + δq * Q
    du[6] = ν_eff * S / N_POPULATION
end

println("  Training UDE...")
ude_time = @elapsed begin
    prob_ude = ODEProblem{true}(siqrdv_ude_unified!, u0, (t_train[1], t_train[end]))
    
    function predict_ude_train(θ)
        Array(solve(prob_ude, Tsit5(), p=θ, saveat=t_train,
                   abstol=1e-6, reltol=1e-5, sensealg=InterpolatingAdjoint()))
    end
    
    function loss_ude(θ)
        pred = predict_ude_train(θ)
        weights = [1.0, 3.0, 2.5, 2.0, 3.0, 1.5]
        loss = sum(weights[i] * mean(abs2, pred[i,:] .- train_data[i,:]) for i in 1:6)
        return loss / sum(weights)
    end
    
    p_initial_ude = ComponentArray(
        nn_beta = p_beta,
        nn_kappa = p_kappa,
        params = params_init_ude
    )
    
    # Callback for tracking
    ude_losses = Float64[]
    function callback_ude(state, loss_val)
        push!(ude_losses, loss_val)
        if length(ude_losses) % 100 == 0
            println("    UDE Iter $(length(ude_losses)): Loss = $(round(loss_val, digits=5))")
        end
        return false
    end
    
    optf_ude = Optimization.OptimizationFunction((x,p) -> loss_ude(x), 
                                                 Optimization.AutoZygote())
    optprob_ude = Optimization.OptimizationProblem(optf_ude, p_initial_ude)
    
    # Training with ADAM - increased iterations to 1000
    res_ude_1 = Optimization.solve(optprob_ude, ADAM(0.01), 
                                   callback=callback_ude, maxiters=800)
    
    # Fine-tuning with BFGS
    optprob_ude_2 = remake(optprob_ude, u0=res_ude_1.u)
    res_ude_2 = Optimization.solve(optprob_ude_2,
                                   Optim.BFGS(initial_stepnorm=0.01),
                                   callback=callback_ude, maxiters=200)
    
    global trained_params_ude = res_ude_2.u
end

println("✓ UDE completed in $(round(ude_time, digits=2)) seconds")

# UDE predictions
prob_ude_train = ODEProblem{true}(siqrdv_ude_unified!, u0, (t_train[1], t_train[end]))
ude_train_pred = Array(solve(prob_ude_train, Tsit5(), p=trained_params_ude, 
                             saveat=t_train, abstol=1e-6, reltol=1e-5))

prob_ude_test = ODEProblem{true}(siqrdv_ude_unified!, train_data[:, end], (t_test[1], t_test[end]))
ude_test_pred = Array(solve(prob_ude_test, Tsit5(), p=trained_params_ude, 
                            saveat=t_test, abstol=1e-6, reltol=1e-5))

# ============================================
# PERFORMANCE METRICS
# ============================================
function calc_r2(true_data, pred_data)
    ss_res = sum(abs2, true_data .- pred_data)
    ss_tot = sum(abs2, true_data .- mean(true_data))
    return 1 - ss_res / (ss_tot + eps())
end

function calc_metrics(true_data, pred_data)
    r2 = calc_r2(true_data, pred_data)
    rmse = sqrt(mean(abs2, true_data .- pred_data))
    mae = mean(abs, true_data .- pred_data)
    return r2, rmse, mae
end

# Calculate metrics for all models
models = ["VB", "SMC", "NODE", "UDE"]
train_preds = [vb_train_pred, smc_train_pred, node_train_pred, ude_train_pred]
test_preds = [vb_test_pred, smc_test_pred, node_test_pred, ude_test_pred]
training_times = [vb_time, smc_time, node_time, ude_time]

metrics_summary = Dict()
for (i, model) in enumerate(models)
    model_train_r2, model_train_rmse, model_train_mae = calc_metrics(truth_train, train_preds[i])
    model_test_r2, model_test_rmse, model_test_mae = calc_metrics(truth_test, test_preds[i])
    
    metrics_summary[model] = Dict(
        "train_r2" => model_train_r2,
        "test_r2" => model_test_r2,
        "train_rmse" => model_train_rmse,
        "test_rmse" => model_test_rmse,
        "time" => training_times[i]
    )
end

# ============================================
# COMPREHENSIVE VISUALIZATION
# ============================================
println("\n" * "="^60)
println("GENERATING COMPREHENSIVE VISUALIZATIONS")
println("="^60)

compartments = ["Susceptible", "Infected", "Quarantined", "Recovered", "Deaths", "Vaccinated"]
colors_dict = Dict("VB" => :blue, "SMC" => :red, "NODE" => :green, "UDE" => :purple)

# Main comparison plot - All 4 models
plt_main = plot(size=(2400, 1600), layout=(2,3),
               left_margin=20Plots.mm, right_margin=15Plots.mm,
               bottom_margin=15Plots.mm, top_margin=25Plots.mm)

for comp_idx in 1:6
    # Truth data
    plot!(plt_main[comp_idx], t_full, truth_full[comp_idx,:], 
          label="Truth", lw=3, color=:black, alpha=1.0)
    
    scatter!(plt_main[comp_idx], t_train[1:3:end], train_data[comp_idx,1:3:end], 
            label="Train Data", color=:gray, markersize=2, alpha=0.5)
    
    # Model predictions
    for (i, model) in enumerate(models)
        plot!(plt_main[comp_idx], t_train, train_preds[i][comp_idx,:], 
              label="$model Train", lw=2, color=colors_dict[model], alpha=1.0)
        plot!(plt_main[comp_idx], t_test, test_preds[i][comp_idx,:], 
              label="$model Test", lw=2, color=colors_dict[model], alpha=0.7, ls=:dash)
    end
    
    # Split line
    vline!(plt_main[comp_idx], [t_train[end]], color=:orange, lw=3, label="Split", alpha=0.8)
    
    # Title and labels
    title!(plt_main[comp_idx], compartments[comp_idx], titlefont=font(11, "bold"))
    xlabel!(plt_main[comp_idx], "Days")
    ylabel!(plt_main[comp_idx], "Population")
    
    # Add R² scores with fixed variable scoping
    r2_text = "Test R²:\n"
    for (i, model) in enumerate(models)
        model_test_r2 = calc_r2(truth_test[comp_idx,:], test_preds[i][comp_idx,:])
        r2_text *= "$model: $(round(model_test_r2, digits=3))\n"
    end
    
    annotate!(plt_main[comp_idx], 0.02, 0.98,
             text(r2_text[1:end-1], :left, :top, 6))
    
    plot!(plt_main[comp_idx], legend=(comp_idx==1), legendfontsize=5,
          grid=true, gridalpha=0.3)
end

# Build title string without loop to avoid scoping issues
title_part1 = "Four-Model Comparison: Truth vs Predictions for All Compartments\n"
title_part2 = "VB: R²=" * string(round(metrics_summary["VB"]["test_r2"], digits=3)) * 
              ", Time=" * string(round(metrics_summary["VB"]["time"], digits=1)) * "s | "
title_part3 = "SMC: R²=" * string(round(metrics_summary["SMC"]["test_r2"], digits=3)) * 
              ", Time=" * string(round(metrics_summary["SMC"]["time"], digits=1)) * "s | "
title_part4 = "NODE: R²=" * string(round(metrics_summary["NODE"]["test_r2"], digits=3)) * 
              ", Time=" * string(round(metrics_summary["NODE"]["time"], digits=1)) * "s | "
title_part5 = "UDE: R²=" * string(round(metrics_summary["UDE"]["test_r2"], digits=3)) * 
              ", Time=" * string(round(metrics_summary["UDE"]["time"], digits=1)) * "s"

final_title = title_part1 * title_part2 * title_part3 * title_part4 * title_part5

plot!(plt_main, plot_title=final_title, plot_titlefontsize=13)
savefig(plt_main, "unified_four_model_comparison.png")
println("✓ Saved: unified_four_model_comparison.png")

# Training time and performance comparison
plt_perf = plot(size=(1800, 600), layout=(1,3))

# Training times
bar!(plt_perf[1], models, training_times, 
     label="", color=[:blue, :red, :green, :purple], alpha=0.7)
title!(plt_perf[1], "Training Time Comparison")
ylabel!(plt_perf[1], "Time (seconds)")
xlabel!(plt_perf[1], "Model")
for i in 1:4
    annotate!(plt_perf[1], i, training_times[i] + maximum(training_times)*0.02,
             text("$(round(training_times[i], digits=1))s", 8, :center))
end

# Test R² scores
test_r2_scores = [metrics_summary[m]["test_r2"] for m in models]
bar!(plt_perf[2], models, test_r2_scores, 
     label="", color=[:blue, :red, :green, :purple], alpha=0.7)
title!(plt_perf[2], "Test R² Scores")
ylabel!(plt_perf[2], "R² Score")
xlabel!(plt_perf[2], "Model")
ylims!(plt_perf[2], 0, 1.05)
for i in 1:4
    annotate!(plt_perf[2], i, test_r2_scores[i] + 0.02,
             text("$(round(test_r2_scores[i], digits=3))", 8, :center))
end

# Speed-Performance Trade-off
scatter!(plt_perf[3], training_times, test_r2_scores, 
        label="", markersize=10, color=[:blue, :red, :green, :purple])
for i in 1:4
    annotate!(plt_perf[3], training_times[i], test_r2_scores[i] + 0.01,
             text(models[i], 8, :center))
end
xlabel!(plt_perf[3], "Training Time (seconds)")
ylabel!(plt_perf[3], "Test R² Score")
title!(plt_perf[3], "Speed vs Performance Trade-off")
plot!(plt_perf[3], grid=true)

savefig(plt_perf, "unified_performance_metrics.png")
println("✓ Saved: unified_performance_metrics.png")

# Compartment-wise R² comparison
plt_comp_r2 = plot(size=(1400, 800), layout=(2,3))

for comp_idx in 1:6
    comp_train_r2s = Float64[]
    comp_test_r2s = Float64[]
    
    for (i, model) in enumerate(models)
        comp_train_r2 = calc_r2(truth_train[comp_idx,:], train_preds[i][comp_idx,:])
        comp_test_r2 = calc_r2(truth_test[comp_idx,:], test_preds[i][comp_idx,:])
        push!(comp_train_r2s, comp_train_r2)
        push!(comp_test_r2s, comp_test_r2)
    end
    
    x = 1:4
    bar!(plt_comp_r2[comp_idx], x .- 0.2, comp_train_r2s, 
         bar_width=0.35, label="Train", color=:blue, alpha=0.6)
    bar!(plt_comp_r2[comp_idx], x .+ 0.2, comp_test_r2s, 
         bar_width=0.35, label="Test", color=:red, alpha=0.6)
    
    xticks!(plt_comp_r2[comp_idx], x, models, rotation=45)
    title!(plt_comp_r2[comp_idx], compartments[comp_idx])
    ylabel!(plt_comp_r2[comp_idx], "R² Score")
    ylims!(plt_comp_r2[comp_idx], 0, 1.05)
    plot!(plt_comp_r2[comp_idx], legend=(comp_idx==1), legendfontsize=7)
end

savefig(plt_comp_r2, "unified_compartment_r2_comparison.png")
println("✓ Saved: unified_compartment_r2_comparison.png")

# Save results to CSV
results_df = DataFrame(
    Model = repeat(models, 2),
    Dataset = vcat(fill("Train", 4), fill("Test", 4)),
    R2 = vcat([metrics_summary[m]["train_r2"] for m in models],
              [metrics_summary[m]["test_r2"] for m in models]),
    RMSE = vcat([metrics_summary[m]["train_rmse"] for m in models],
                [metrics_summary[m]["test_rmse"] for m in models]),
    TrainingTime = repeat(training_times, 2)
)

CSV.write("unified_model_comparison_results.csv", results_df)
println("✓ Saved: unified_model_comparison_results.csv")

# ============================================
# FINAL SUMMARY
# ============================================
println("\n" * "="^80)
println("FINAL SUMMARY - UNIFIED FOUR MODEL COMPARISON")
println("="^80)

println("\nTRAINING TIME RANKING:")
time_ranking = sort(collect(zip(models, training_times)), by=x->x[2])
for (i, (model, time)) in enumerate(time_ranking)
    println("  $i. $model: $(round(time, digits=2)) seconds")
end

println("\nTEST PERFORMANCE RANKING (R²):")
r2_ranking = sort([(m, metrics_summary[m]["test_r2"]) for m in models], by=x->x[2], rev=true)
for (i, (model, r2)) in enumerate(r2_ranking)
    println("  $i. $model: $(round(r2, digits=4))")
end

println("\nBEST MODEL BY METRIC:")
println("  • Fastest Training: $(time_ranking[1][1]) ($(round(time_ranking[1][2], digits=2))s)")
println("  • Best Test R²: $(r2_ranking[1][1]) ($(round(r2_ranking[1][2], digits=4)))")

# Performance per second metric
perf_per_sec = [(m, metrics_summary[m]["test_r2"]/metrics_summary[m]["time"]) for m in models]
best_efficiency = sort(perf_per_sec, by=x->x[2], rev=true)[1]
println("  • Best Efficiency (R²/sec): $(best_efficiency[1]) ($(round(best_efficiency[2], digits=4)))")

println("\nGENERATED FILES:")
println("  • unified_four_model_comparison.png - Main comparison plot")
println("  • unified_performance_metrics.png - Time and performance metrics")
println("  • unified_compartment_r2_comparison.png - Detailed compartment analysis")
println("  • training_time_comparison.png - Comprehensive time analysis")
println("  • model_efficiency_analysis.png - Efficiency and trade-off analysis")
println("  • unified_model_comparison_results.csv - Numerical results")

println("\n" * "="^80)
println("ANALYSIS COMPLETE!")
println("="^80)