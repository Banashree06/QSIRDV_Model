"""
COMPLETE SIQRDV MODEL - CLEANED VERSION
UDE vs Neural ODE Comparison with All Visualizations - 1000 Points
"""

using DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Lux, DiffEqFlux, ComponentArrays, Random, Statistics
using Plots
using SciMLSensitivity
using CSV, DataFrames

# Set random seed
Random.seed!(1234)
rng = Random.default_rng()

println("="^60)
println("SIQRDV MODEL: UDE vs Neural ODE Comparison - 1000 Points")
println("="^60)

# ============================================
# CONFIGURATION
# ============================================
FloatType = Float64
N_days = 160
N0_population = FloatType(1000.0)

function regular_sample_times(N_days, n_points=1000)  # Changed to 1000 points
    return range(FloatType(0), FloatType(N_days), length=n_points)
end

# ============================================
# STEP 1: DATA GENERATION
# ============================================
println("\nSTEP 1: GENERATING DATA")
println("-"^50)

u0 = FloatType[990, 10, 0, 0, 0, 0]  # [S, I, Q, R, D, V]
p_static = FloatType[0.25, 0.06, 0.12, 0.09, 0.008, 0.004, 0.015]  # [β, κ, γ, γq, δ, δq, ν]
tspan = (FloatType(0), FloatType(N_days))
t_points = regular_sample_times(N_days, 1000)  # Changed to 1000 points

function SIQRDV_static!(du, u, p, t)
    S, I, Q, R, D, V = u
    β, κ, γ, γq, δ, δq, ν = p
    
    N_living = S + I + Q + R + V
    if N_living < 1
        du .= 0
        return
    end
    
    inv_N = 1.0 / N_living
    βSI_N = β * S * I * inv_N
    
    du[1] = -βSI_N - ν * S              # dS/dt
    du[2] = βSI_N - (γ + δ + κ) * I     # dI/dt
    du[3] = κ * I - (γq + δq) * Q       # dQ/dt
    du[4] = γ * I + γq * Q              # dR/dt
    du[5] = δ * I + δq * Q              # dD/dt
    du[6] = ν * S                        # dV/dt
end

# Generate ground truth data
prob_true = ODEProblem(SIQRDV_static!, u0, tspan, p_static)
sol_true = solve(prob_true, Tsit5(), saveat=t_points, abstol=1e-8, reltol=1e-7)
ode_data = Array(sol_true)

# Add noise
noise_level = FloatType(0.01)
ode_data_noisy = ode_data .+ noise_level * randn(FloatType, size(ode_data)) .* sqrt.(abs.(ode_data) .+ 1)
ode_data = max.(FloatType(0), ode_data_noisy)

println("✓ Data generated: Peak infected = $(round(maximum(ode_data[2,:]), digits=1))")

# ============================================
# STEP 2: TRAIN/TEST SPLIT
# ============================================
n_train = Int(round(length(t_points) * 0.7))
train_idx = 1:n_train
test_idx = n_train:length(t_points)

t_train = t_points[train_idx]
t_test = t_points[test_idx]
train_data = ode_data[:, train_idx]
test_data = ode_data[:, test_idx]

println("✓ Split: $(length(t_train)) training, $(length(t_test)) testing points")

# ============================================
# PART A: UDE MODEL
# ============================================
println("\nPART A: TRAINING UDE MODEL")
println("-"^50)

# Neural networks for UDE
NN_beta_static = Lux.Chain(
    Lux.Dense(2, 12, tanh),
    Lux.Dense(12, 8, tanh),
    Lux.Dense(8, 1, softplus)
)

NN_kappa_static = Lux.Chain(
    Lux.Dense(1, 8, tanh),
    Lux.Dense(8, 1, sigmoid)
)

p_beta_static, st_beta_static = Lux.setup(rng, NN_beta_static)
p_kappa_static, st_kappa_static = Lux.setup(rng, NN_kappa_static)

params_init_static = FloatType[0.11, 0.085, 0.007, 0.0035, 0.014]

function siqrdv_ude_static!(du, u, p, t)
    S, I, Q, R, D, V = u
    N_living = max(S + I + Q + R + V, FloatType(1))
    inv_N = FloatType(1) / N_living
    
    S_norm = S / N0_population
    I_norm = I / FloatType(100)
    
    β_input = [S_norm, I_norm]
    β_nn = NN_beta_static(β_input, p.nn_beta, st_beta_static)[1][1]
    β = FloatType(0.1) + FloatType(0.3) * β_nn
    
    κ_input = [I_norm]
    κ_nn = NN_kappa_static(κ_input, p.nn_kappa, st_kappa_static)[1][1]
    κ = FloatType(0.03) + FloatType(0.07) * κ_nn
    
    γ = abs(p.params[1])
    γq = abs(p.params[2])
    δ = abs(p.params[3])
    δq = abs(p.params[4])
    ν = abs(p.params[5])
    
    βSI_N = β * S * I * inv_N
    
    du[1] = -βSI_N - ν * S
    du[2] = βSI_N - (γ + δ + κ) * I
    du[3] = κ * I - (γq + δq) * Q
    du[4] = γ * I + γq * Q
    du[5] = δ * I + δq * Q
    du[6] = ν * S
end

# Training setup
prob_ude_static = ODEProblem{true}(siqrdv_ude_static!, u0, (t_train[1], t_train[end]))

function predict_ude_train_static(θ)
    Array(solve(prob_ude_static, Tsit5(), p=θ, saveat=t_train,
               abstol=1e-6, reltol=1e-5, sensealg=InterpolatingAdjoint()))
end

function loss_ude_static(θ)
    pred = predict_ude_train_static(θ)
    weights = FloatType[1.0, 3.0, 2.5, 2.0, 3.0, 1.5]
    loss = FloatType(0)
    for i in 1:6
        loss += weights[i] * mean(abs2, pred[i,:] .- train_data[i,:])
    end
    return loss / sum(weights)
end

# Train UDE
p_initial_ude_static = ComponentArray(
    nn_beta = p_beta_static,
    nn_kappa = p_kappa_static,
    params = params_init_static
)

ude_losses = Float64[]
function callback_ude(state, loss_val)
    push!(ude_losses, loss_val)
    if length(ude_losses) % 100 == 0
        println("  UDE Iter $(length(ude_losses)): Loss = $(round(loss_val, digits=5))")
    end
    return false
end

# Time UDE training
ude_train_time = @elapsed begin
    optf_ude = Optimization.OptimizationFunction((x,p) -> loss_ude_static(x), 
                                                  Optimization.AutoZygote())
    optprob_ude = Optimization.OptimizationProblem(optf_ude, p_initial_ude_static)

    res_ude_1 = Optimization.solve(optprob_ude, ADAM(0.01), 
                                    callback=callback_ude, maxiters=500)

    optprob_ude_2 = remake(optprob_ude, u0=res_ude_1.u)
    res_ude_2 = Optimization.solve(optprob_ude_2, 
                                    Optim.BFGS(initial_stepnorm=0.01),
                                    callback=callback_ude, maxiters=300)

    trained_params_ude = res_ude_2.u
end

println("  Training time: $(round(ude_train_time, digits=2)) seconds")

# Generate UDE predictions
ude_train_pred = predict_ude_train_static(trained_params_ude)
prob_ude_test = remake(prob_ude_static, tspan=(t_test[1], t_test[end]), 
                       u0=train_data[:, end])
ude_test_pred = Array(solve(prob_ude_test, Tsit5(), p=trained_params_ude, 
                            saveat=t_test, abstol=1e-6, reltol=1e-5))

println("✓ UDE training complete")

# ============================================
# PART B: NEURAL ODE MODEL
# ============================================
println("\nPART B: TRAINING NEURAL ODE")
println("-"^50)

dudt_nn = Lux.Chain(
    Lux.Dense(6, 24, tanh),
    Lux.Dense(24, 24, tanh),
    Lux.Dense(24, 6)
)

p_nn, st_nn = Lux.setup(rng, dudt_nn)
prob_node = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5(), saveat=t_train)

function predict_node_train(p)
    Array(prob_node(u0, p, st_nn)[1])
end

function loss_node(p)
    pred = predict_node_train(p)
    weights = [1.0, 5.0, 2.0, 1.0, 1.0, 1.0]
    loss = 0.0
    for i in 1:6
        loss += weights[i] * sum(abs2, train_data[i,:] .- pred[i,:])
    end
    return loss / length(train_data)
end

# Train Neural ODE
p_initial_node = ComponentArray(p_nn)
node_losses = Float64[]

function callback_node(state, loss_val)
    push!(node_losses, loss_val)
    if length(node_losses) % 100 == 0
        println("  NODE Iter $(length(node_losses)): Loss = $(round(loss_val, digits=5))")
    end
    return false
end

# Time Neural ODE training
node_train_time = @elapsed begin
    optf_node = Optimization.OptimizationFunction((x,p) -> loss_node(x), 
                                                   Optimization.AutoZygote())
    optprob_node = Optimization.OptimizationProblem(optf_node, p_initial_node)

    res_node_1 = Optimization.solve(optprob_node, ADAM(0.01), 
                                     callback=callback_node, maxiters=500)

    optprob_node_2 = remake(optprob_node, u0=res_node_1.u)
    res_node_2 = Optimization.solve(optprob_node_2, 
                                     Optim.BFGS(initial_stepnorm=0.001),
                                     callback=callback_node, maxiters=300)

    trained_params_node = res_node_2.u
end

println("  Training time: $(round(node_train_time, digits=2)) seconds")

# Generate NODE predictions
node_train_pred = predict_node_train(trained_params_node)
u_last = train_data[:, end]
prob_node_test = NeuralODE(dudt_nn, (t_test[1], t_test[end]), Tsit5(), saveat=t_test)
node_test_pred = Array(prob_node_test(u_last, trained_params_node, st_nn)[1])

println("✓ Neural ODE training complete")

# ============================================
# EVALUATION METRICS
# ============================================
println("\n" * "="^60)
println("COMPREHENSIVE PERFORMANCE EVALUATION")
println("="^60)

function calc_r2(true_data, pred_data)
    ss_res = sum(abs2, true_data .- pred_data)
    ss_tot = sum(abs2, true_data .- mean(true_data))
    return 1 - ss_res / (ss_tot + eps(FloatType))
end

function calc_all_metrics(true_data, pred_data)
    # R² Score
    r2 = calc_r2(true_data, pred_data)
    
    # Mean Squared Error
    mse = mean(abs2, true_data .- pred_data)
    
    # Root Mean Squared Error
    rmse = sqrt(mse)
    
    # Mean Absolute Error
    mae = mean(abs, true_data .- pred_data)
    
    # Mean Absolute Percentage Error (avoid division by zero)
    mape = mean(abs.((true_data .- pred_data) ./ (true_data .+ eps(FloatType)))) * 100
    
    # Maximum Error
    max_error = maximum(abs.(true_data .- pred_data))
    
    # Normalized RMSE (percentage of data range)
    nrmse = rmse / (maximum(true_data) - minimum(true_data) + eps(FloatType)) * 100
    
    return r2, mse, rmse, mae, mape, max_error, nrmse
end

# Calculate all metrics for both models
ude_train_metrics = calc_all_metrics(train_data, ude_train_pred)
ude_test_metrics = calc_all_metrics(test_data, ude_test_pred)
node_train_metrics = calc_all_metrics(train_data, node_train_pred)
node_test_metrics = calc_all_metrics(test_data, node_test_pred)

# Extract metrics for display
ude_train_r2, ude_train_mse, ude_train_rmse, ude_train_mae, ude_train_mape, ude_train_maxe, ude_train_nrmse = ude_train_metrics
ude_test_r2, ude_test_mse, ude_test_rmse, ude_test_mae, ude_test_mape, ude_test_maxe, ude_test_nrmse = ude_test_metrics
node_train_r2, node_train_mse, node_train_rmse, node_train_mae, node_train_mape, node_train_maxe, node_train_nrmse = node_train_metrics
node_test_r2, node_test_mse, node_test_rmse, node_test_mae, node_test_mape, node_test_maxe, node_test_nrmse = node_test_metrics

# Display comprehensive metrics
println("\nMODEL PERFORMANCE METRICS:")
println("-"^80)
println("                    |         UDE          |      Neural ODE      |")
println("Metric              |   Train   |   Test   |   Train   |   Test   |")
println("-"^80)
println("R² Score            | $(lpad(round(ude_train_r2, digits=3), 9)) | $(lpad(round(ude_test_r2, digits=3), 8)) | $(lpad(round(node_train_r2, digits=3), 9)) | $(lpad(round(node_test_r2, digits=3), 8)) |")
println("MSE                 | $(lpad(round(ude_train_mse, digits=1), 9)) | $(lpad(round(ude_test_mse, digits=1), 8)) | $(lpad(round(node_train_mse, digits=1), 9)) | $(lpad(round(node_test_mse, digits=1), 8)) |")
println("RMSE                | $(lpad(round(ude_train_rmse, digits=2), 9)) | $(lpad(round(ude_test_rmse, digits=2), 8)) | $(lpad(round(node_train_rmse, digits=2), 9)) | $(lpad(round(node_test_rmse, digits=2), 8)) |")
println("MAE                 | $(lpad(round(ude_train_mae, digits=2), 9)) | $(lpad(round(ude_test_mae, digits=2), 8)) | $(lpad(round(node_train_mae, digits=2), 9)) | $(lpad(round(node_test_mae, digits=2), 8)) |")
println("MAPE (%)            | $(lpad(round(ude_train_mape, digits=1), 9)) | $(lpad(round(ude_test_mape, digits=1), 8)) | $(lpad(round(node_train_mape, digits=1), 9)) | $(lpad(round(node_test_mape, digits=1), 8)) |")
println("Max Error           | $(lpad(round(ude_train_maxe, digits=1), 9)) | $(lpad(round(ude_test_maxe, digits=1), 8)) | $(lpad(round(node_train_maxe, digits=1), 9)) | $(lpad(round(node_test_maxe, digits=1), 8)) |")
println("Normalized RMSE (%) | $(lpad(round(ude_train_nrmse, digits=1), 9)) | $(lpad(round(ude_test_nrmse, digits=1), 8)) | $(lpad(round(node_train_nrmse, digits=1), 9)) | $(lpad(round(node_test_nrmse, digits=1), 8)) |")
println("-"^80)

# Training efficiency comparison
println("\nTRAINING EFFICIENCY:")
println("-"^50)
println("UDE Training Time:        $(round(ude_train_time, digits=2)) seconds")
println("Neural ODE Training Time: $(round(node_train_time, digits=2)) seconds")
println("Speed Ratio (UDE/NODE):   $(round(ude_train_time/node_train_time, digits=2))x")
println("Total Iterations:")
println("  UDE:  $(length(ude_losses)) iterations")
println("  NODE: $(length(node_losses)) iterations")

# Per-compartment metrics
println("\nPER-COMPARTMENT TEST PERFORMANCE (R² Score):")
println("-"^50)

compartment_names = ["Susceptible (S)", "Infected (I)", "Quarantined (Q)", 
                     "Recovered (R)", "Deaths (D)", "Vaccinated (V)"]
compartments_short = ["S", "I", "Q", "R", "D", "V"]

# ============================================
# VISUALIZATIONS
# ============================================
println("\nGENERATING VISUALIZATIONS")
println("-"^50)

# 1. Combined Truth vs Both Models
plt_combined = plot(size=(2400, 1600), layout=(2,3),
                   left_margin=15Plots.mm, right_margin=15Plots.mm,
                   bottom_margin=15Plots.mm, top_margin=20Plots.mm,
                   titlefontsize=14, legendfontsize=8)

for i in 1:6
    # Truth data - subsample for visualization to avoid overcrowding
    train_step = max(1, div(length(t_train), 100))  # Show ~100 points for training
    test_step = max(1, div(length(t_test), 50))     # Show ~50 points for testing
    
    scatter!(plt_combined[i], t_train[1:train_step:end], train_data[i,1:train_step:end], 
            label="Truth (Train)", color=:black, markersize=2,
            markershape=:circle, markeralpha=0.6)
    
    scatter!(plt_combined[i], t_test[1:test_step:end], test_data[i,1:test_step:end], 
            label="Truth (Test)", color=:gray40, markersize=2,
            markershape=:square, markeralpha=0.6)
    
    # UDE predictions
    plot!(plt_combined[i], t_train, ude_train_pred[i,:], 
          label="UDE Train", lw=2.5, color=:blue, alpha=1.0)
    plot!(plt_combined[i], t_test, ude_test_pred[i,:], 
          label="UDE Test", lw=2.5, color=:blue, alpha=0.7, ls=:dash)
    
    # Neural ODE predictions
    plot!(plt_combined[i], t_train, node_train_pred[i,:], 
          label="NODE Train", lw=2.5, color=:red, alpha=1.0)
    plot!(plt_combined[i], t_test, node_test_pred[i,:], 
          label="NODE Test", lw=2.5, color=:red, alpha=0.7, ls=:dash)
    
    # Strong separation line
    vline!(plt_combined[i], [t_train[end]], 
           color=:green, linestyle=:solid, alpha=1.0, lw=5, label="Split")
    
    # Shaded regions
    vspan!(plt_combined[i], [0, t_train[end]], color=:blue, alpha=0.03, label="")
    vspan!(plt_combined[i], [t_train[end], t_test[end]], color=:orange, alpha=0.03, label="")
    
    title!(plt_combined[i], compartment_names[i], titlefont=font(12, "bold"))
    xlabel!(plt_combined[i], "Time (days)", guidefont=font(9, "bold"))
    ylabel!(plt_combined[i], "Population", guidefont=font(9, "bold"))
    
    # Metrics
    train_r2_ude = calc_r2(train_data[i,:], ude_train_pred[i,:])
    test_r2_ude = calc_r2(test_data[i,:], ude_test_pred[i,:])
    train_r2_node = calc_r2(train_data[i,:], node_train_pred[i,:])
    test_r2_node = calc_r2(test_data[i,:], node_test_pred[i,:])
    
    metrics_text = "UDE:  Test R²=$(round(test_r2_ude, digits=3))\n" *
                  "NODE: Test R²=$(round(test_r2_node, digits=3))"
    
    annotate!(plt_combined[i], 0.98, 0.98,
             text(metrics_text, :right, :top, 7, :black, font(7, "bold")))
    
    plot!(plt_combined[i], legend=:best, legendfontsize=6,
          grid=true, gridalpha=0.2, framestyle=:box)
end

plot!(plt_combined, 
      plot_title="Complete Comparison: Truth vs UDE vs Neural ODE (1000 points)\n" *
                "UDE: Train R²=$(round(ude_train_r2, digits=3)), Test R²=$(round(ude_test_r2, digits=3)) | " *
                "NODE: Train R²=$(round(node_train_r2, digits=3)), Test R²=$(round(node_test_r2, digits=3))",
      plot_titlefontsize=15, plot_titlefont=font(15, "bold"))

display(plt_combined)
savefig(plt_combined, "truth_vs_both_models_comparison_1000pts.png")

# 2. Training Loss Comparison
plt_loss = plot(size=(1200, 800), layout=(1,2),
               left_margin=10Plots.mm, right_margin=10Plots.mm,
               bottom_margin=10Plots.mm, top_margin=15Plots.mm)

# UDE Loss
plot!(plt_loss[1], 1:length(ude_losses), ude_losses,
      label="UDE Loss", lw=2, color=:blue, alpha=0.8)
xlabel!(plt_loss[1], "Iteration", guidefont=font(10, "bold"))
ylabel!(plt_loss[1], "Loss", guidefont=font(10, "bold"))
title!(plt_loss[1], "UDE Training Loss", titlefont=font(12, "bold"))
plot!(plt_loss[1], yscale=:log10, grid=true, gridalpha=0.3)

# Neural ODE Loss
plot!(plt_loss[2], 1:length(node_losses), node_losses,
      label="NODE Loss", lw=2, color=:red, alpha=0.8)
xlabel!(plt_loss[2], "Iteration", guidefont=font(10, "bold"))
ylabel!(plt_loss[2], "Loss", guidefont=font(10, "bold"))
title!(plt_loss[2], "Neural ODE Training Loss", titlefont=font(12, "bold"))
plot!(plt_loss[2], yscale=:log10, grid=true, gridalpha=0.3)

plot!(plt_loss, plot_title="Training Loss Comparison (1000 points)",
      plot_titlefontsize=14, plot_titlefont=font(14, "bold"))

display(plt_loss)
savefig(plt_loss, "training_loss_comparison_1000pts.png")

# 3. Prediction Error Analysis
plt_error = plot(size=(2400, 1200), layout=(2,3),
                left_margin=15Plots.mm, right_margin=15Plots.mm,
                bottom_margin=15Plots.mm, top_margin=20Plots.mm)

for i in 1:6
    # Calculate errors
    ude_train_error = ude_train_pred[i,:] .- train_data[i,:]
    ude_test_error = ude_test_pred[i,:] .- test_data[i,:]
    node_train_error = node_train_pred[i,:] .- train_data[i,:]
    node_test_error = node_test_pred[i,:] .- test_data[i,:]
    
    # Plot errors
    plot!(plt_error[i], t_train, ude_train_error, 
          label="UDE Train Error", lw=2, color=:blue, alpha=0.8)
    plot!(plt_error[i], t_test, ude_test_error, 
          label="UDE Test Error", lw=2, color=:blue, alpha=0.6, ls=:dash)
    plot!(plt_error[i], t_train, node_train_error, 
          label="NODE Train Error", lw=2, color=:red, alpha=0.8)
    plot!(plt_error[i], t_test, node_test_error, 
          label="NODE Test Error", lw=2, color=:red, alpha=0.6, ls=:dash)
    
    # Zero line
    hline!(plt_error[i], [0], color=:black, alpha=0.5, lw=1, label="")
    
    # Separation line
    vline!(plt_error[i], [t_train[end]], 
           color=:green, linestyle=:solid, alpha=0.8, lw=3, label="Split")
    
    title!(plt_error[i], compartment_names[i], titlefont=font(12, "bold"))
    xlabel!(plt_error[i], "Time (days)", guidefont=font(9, "bold"))
    ylabel!(plt_error[i], "Prediction Error", guidefont=font(9, "bold"))
    
    plot!(plt_error[i], legend=:best, legendfontsize=6,
          grid=true, gridalpha=0.2, framestyle=:box)
end

plot!(plt_error, plot_title="Prediction Error Analysis (1000 points)",
      plot_titlefontsize=15, plot_titlefont=font(15, "bold"))

display(plt_error)
savefig(plt_error, "prediction_error_analysis_1000pts.png")

# 4. R² Score Comparison
compartment_r2_ude_train = [calc_r2(train_data[i,:], ude_train_pred[i,:]) for i in 1:6]
compartment_r2_ude_test = [calc_r2(test_data[i,:], ude_test_pred[i,:]) for i in 1:6]
compartment_r2_node_train = [calc_r2(train_data[i,:], node_train_pred[i,:]) for i in 1:6]
compartment_r2_node_test = [calc_r2(test_data[i,:], node_test_pred[i,:]) for i in 1:6]

plt_r2 = plot(size=(1400, 900), layout=(2,1),
              left_margin=12Plots.mm, right_margin=12Plots.mm,
              bottom_margin=12Plots.mm, top_margin=15Plots.mm)

# Training R²
bar!(plt_r2[1], compartments_short, compartment_r2_ude_train,
     label="UDE Train", color=:blue, alpha=0.7, width=0.35)
bar!(plt_r2[1], compartments_short, compartment_r2_node_train,
     label="NODE Train", color=:red, alpha=0.7, width=0.35)
title!(plt_r2[1], "Training R² Score by Compartment", titlefont=font(12, "bold"))
xlabel!(plt_r2[1], "Compartment", guidefont=font(10, "bold"))
ylabel!(plt_r2[1], "R² Score", guidefont=font(10, "bold"))
plot!(plt_r2[1], legend=:best, grid=true, gridalpha=0.3)

# Testing R²
bar!(plt_r2[2], compartments_short, compartment_r2_ude_test,
     label="UDE Test", color=:blue, alpha=0.7, width=0.35)
bar!(plt_r2[2], compartments_short, compartment_r2_node_test,
     label="NODE Test", color=:red, alpha=0.7, width=0.35)
title!(plt_r2[2], "Testing R² Score by Compartment", titlefont=font(12, "bold"))
xlabel!(plt_r2[2], "Compartment", guidefont=font(10, "bold"))
ylabel!(plt_r2[2], "R² Score", guidefont=font(10, "bold"))
plot!(plt_r2[2], legend=:best, grid=true, gridalpha=0.3)

plot!(plt_r2, plot_title="R² Score Comparison (1000 points)",
      plot_titlefontsize=14, plot_titlefont=font(14, "bold"))

display(plt_r2)
savefig(plt_r2, "r2_score_comparison_1000pts.png")

# Print per-compartment performance
for i in 1:6
    println("$(compartments_short[i]): UDE Test R²=$(round(compartment_r2_ude_test[i], digits=3)), NODE Test R²=$(round(compartment_r2_node_test[i], digits=3))")
end

# ============================================
# SAVE RESULTS
# ============================================
# Create comprehensive results DataFrame
results_df = DataFrame(
    Model = ["UDE", "UDE", "Neural ODE", "Neural ODE"],
    Dataset = ["Train", "Test", "Train", "Test"],
    R2 = [ude_train_r2, ude_test_r2, node_train_r2, node_test_r2],
    MSE = [ude_train_mse, ude_test_mse, node_train_mse, node_test_mse],
    RMSE = [ude_train_rmse, ude_test_rmse, node_train_rmse, node_test_rmse],
    MAE = [ude_train_mae, ude_test_mae, node_train_mae, node_test_mae],
    MAPE = [ude_train_mape, ude_test_mape, node_train_mape, node_test_mape],
    MaxError = [ude_train_maxe, ude_test_maxe, node_train_maxe, node_test_maxe],
    NRMSE = [ude_train_nrmse, ude_test_nrmse, node_train_nrmse, node_test_nrmse]
)

CSV.write("model_results_1000pts.csv", results_df)

# ============================================
# FINAL SUMMARY
# ============================================
println("\n" * "="^60)
println("ANALYSIS COMPLETE - 1000 POINTS")
println("="^60)

# Determine best model based on multiple criteria
ude_score = 0
node_score = 0

# Test R² comparison
if ude_test_r2 > node_test_r2
    ude_score += 1
else
    node_score += 1
end

# RMSE comparison
if ude_test_rmse < node_test_rmse
    ude_score += 1
else
    node_score += 1
end

# Training time comparison
if ude_train_time < node_train_time
    ude_score += 1
else
    node_score += 1
end

# Generalization gap comparison
ude_gen_gap = abs(ude_train_r2 - ude_test_r2)
node_gen_gap = abs(node_train_r2 - node_test_r2)
if ude_gen_gap < node_gen_gap
    ude_score += 1
else
    node_score += 1
end

println("\nMODEL COMPARISON SUMMARY:")
println("-"^50)
println("Criteria Winner:")
println("  Test R²:            $(ude_test_r2 > node_test_r2 ? "UDE" : "Neural ODE")")
println("  Test RMSE:          $(ude_test_rmse < node_test_rmse ? "UDE" : "Neural ODE")")
println("  Training Speed:     $(ude_train_time < node_train_time ? "UDE" : "Neural ODE")")
println("  Generalization:     $(ude_gen_gap < node_gen_gap ? "UDE" : "Neural ODE")")

println("\nOverall Winner: $(ude_score > node_score ? "UDE" : "Neural ODE") ($(max(ude_score, node_score))/4 criteria)")

if ude_score > node_score
    println("  ✓ UDE demonstrates superior overall performance")
    println("    - Test R² = $(round(ude_test_r2, digits=3)), RMSE = $(round(ude_test_rmse, digits=2))")
else
    println("  ✓ Neural ODE demonstrates superior overall performance")
    println("    - Test R² = $(round(node_test_r2, digits=3)), RMSE = $(round(node_test_rmse, digits=2))")
end

println("\nGenerated PNG files:")
println("  - truth_vs_both_models_comparison_1000pts.png")
println("  - training_loss_comparison_1000pts.png") 
println("  - prediction_error_analysis_1000pts.png")
println("  - r2_score_comparison_1000pts.png")
println("  - model_results_1000pts.csv")