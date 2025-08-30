using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, Plots, ComponentArrays
using CSV, DataFrames, Statistics
using Dates

Random.seed!(1234)
rng = Random.default_rng()

println("NEURAL ODE FOR SIQRDV MODEL")
println("="^60)
println("Started at: $(Dates.now())")

# ============================================
# 1. Generate Synthetic Data
# ============================================
println("\n1. Generating synthetic SIQRDV data...")

# Initial conditions: S, I, Q, R, D, V
u0 = [990.0, 10.0, 0.0, 0.0, 0.0, 0.0]
tspan = (0.0, 160.0)
t = range(0, 160, length=161)

# True parameters: β, κ, γ, γq, δ, δq, ν
p_true = [0.3, 0.05, 0.1, 0.08, 0.01, 0.005, 0.02]

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

# Generate ground truth data
prob = ODEProblem(siqrdv!, u0, tspan, p_true)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

# Split data: 75% train, 25% test
n_train = 121  # 75% of 161 points
t_train = t[1:n_train]
t_test = t[n_train+1:end]
train_data = ode_data[:, 1:n_train]
test_data = ode_data[:, n_train+1:end]

println("  Generated $(size(ode_data, 2)) time points")
println("  Training: $(n_train) points (t=0 to t=$(t_train[end]))")
println("  Testing: $(length(t_test)) points (t=$(t_test[1]) to t=$(t_test[end]))")

# ============================================
# 2. Neural ODE Architecture Setup
# ============================================
println("\n2. Setting up Neural ODE architecture...")

# Define neural network architecture
dudt_nn = Lux.Chain(
    Lux.Dense(6, 32, tanh),     # Input layer: 6 compartments → 32 hidden
    Lux.Dense(32, 32, tanh),    # Hidden layer: 32 → 32
    Lux.Dense(32, 6)            # Output layer: 32 → 6 compartments
)

# Initialize parameters and state
p_nn, st_nn = Lux.setup(rng, dudt_nn)
println("  Network architecture: 6 → 32 → 32 → 6")
println("  Activation: tanh")
println("  Total parameters: $(sum(length, Lux.parameterlength(dudt_nn)))")

# Create Neural ODE problem for training
prob_train = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5(), saveat=t_train)

# Prediction function for training data
function predict_train(p)
    Array(prob_train(u0, p, st_nn)[1])
end

# Weighted loss function
function loss_train(p)
    pred = predict_train(p)
    
    # Compartment-specific weights (emphasize infected and quarantined)
    weights = [1.0, 5.0, 2.0, 1.0, 1.0, 1.0]  # S, I, Q, R, D, V
    loss = 0.0
    
    for i in 1:6
        compartment_loss = sum(abs2, train_data[i,:] .- pred[i,:])
        loss += weights[i] * compartment_loss
    end
    
    # Normalize by total number of data points
    return loss / length(train_data)
end

# ============================================
# 3. Setup Checkpoint System
# ============================================
println("\n3. Setting up checkpoint system...")

# Create DataFrame to store all checkpoints
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

# Function to calculate metrics
function calc_metrics(true_data, pred_data)
    mse = sum(abs2, true_data .- pred_data) / length(true_data)
    mae = sum(abs, true_data .- pred_data) / length(true_data)
    
    # R² score
    ss_tot = sum(abs2, true_data .- mean(true_data))
    ss_res = sum(abs2, true_data .- pred_data)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, mae, r2
end

# Function to save checkpoint
function save_checkpoint(epoch, iteration, loss, params, optimizer_name, lr, elapsed_time)
    # Calculate current predictions and metrics
    try
        train_pred = predict_train(params)
        train_mse, train_mae, train_r2 = calc_metrics(train_data, train_pred)
        
        # Add to DataFrame
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
        
        # Save to CSV (overwrite with updated data)
        CSV.write("training_checkpoints.csv", checkpoints_df)
        
        # Also save best model checkpoint if this is the best loss so far
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

# ============================================
# 4. Training Process with Checkpointing
# ============================================
println("\n4. Training Neural ODE with checkpointing...")
println("  Optimization strategy: ADAM → BFGS")
println("  Checkpoints will be saved every 50 iterations")

# Track training progress
losses = Float64[]
train_times = Float64[]
start_time = time()
current_epoch = 1
checkpoint_interval = 50

# Enhanced callback function with checkpointing
training_callback = function(state, l)
    push!(losses, l)
    push!(train_times, time() - start_time)
    iteration = length(losses)
    
    # Save checkpoint at intervals
    if iteration % checkpoint_interval == 0 || iteration == 1
        current_lr = iteration <= 500 ? 0.01 : 0.001  # Approximate LR
        optimizer_name = iteration <= 500 ? "ADAM" : "BFGS"
        
        if save_checkpoint(current_epoch, iteration, l, state.u, 
                          optimizer_name, current_lr, train_times[end])
            println("Checkpoint saved at iteration $iteration | Loss = $(round(l, digits=6)) | Time = $(round(train_times[end], digits=2))s")
        end
    end
    
    # Regular progress update
    if iteration % 100 == 0
        println("Iteration $iteration: Loss = $(round(l, digits=6)) | Time = $(round(train_times[end], digits=2))s")
    end
    
    return false
end

# Initialize parameters
pinit = ComponentArray(p_nn)

# Setup optimization problem
optf = Optimization.OptimizationFunction((x, p) -> loss_train(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, pinit)

# Phase 1: ADAM optimizer (global search)
println("\n  Phase 1: ADAM Optimization (Epochs 1-5)")
println("  " * "─"^40)

# Run ADAM in epochs
adam_iters_per_epoch = 100
for epoch in 1:5
    current_epoch = epoch
    println("  Epoch $epoch:")
    
    # Run optimization for this epoch
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
    
    # End of epoch summary
    epoch_loss = losses[end]
    println("Epoch $epoch complete: Loss = $(round(epoch_loss, digits=6))")
end

# Phase 2: BFGS optimizer (local refinement)
println("\n Phase 2: BFGS Refinement (Epochs 6-7)")
println("  " * "─"^40)

optprob2 = remake(optprob, u0=result1.u)

# Run BFGS in epochs
bfgs_iters_per_epoch = 100
for epoch in 6:7
    current_epoch = epoch
    println("Epoch $epoch:")
    
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
    
    # End of epoch summary
    epoch_loss = losses[end]
    println("Epoch $epoch complete: Loss = $(round(epoch_loss, digits=6))")
end

final_params = result2.u
total_time = time() - start_time

println("\n  Training completed!")
println("  Total epochs: $current_epoch")
println("  Total iterations: $(length(losses))")
println("  Final loss: $(round(losses[end], digits=8))")
println("  Total time: $(round(total_time, digits=2)) seconds")

# ============================================
# 5. Model Evaluation & Forecasting
# ============================================
println("\n5.Evaluating model performance...")

# Training set predictions
train_pred = predict_train(final_params)

# Multi-step ahead forecast (use last training point as initial condition)
u_last = train_data[:, end]
prob_test = NeuralODE(dudt_nn, (t_test[1], t_test[end]), Tsit5(), saveat=t_test)
test_forecast = Array(prob_test(u_last, final_params, st_nn)[1])

# One-step ahead forecast (more conservative approach)
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

# ============================================
# 6. Calculate Final Performance Metrics
# ============================================
println("\n6. Computing final performance metrics...")

# Training metrics
train_mse, train_mae, train_r2 = calc_metrics(train_data, train_pred)

# Test metrics (multi-step)
test_mse_multi, test_mae_multi, test_r2_multi = calc_metrics(test_data, test_forecast)

# Test metrics (1-step)
test_mse_1step, test_mae_1step, test_r2_1step = calc_metrics(test_data, test_1step)

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

# ============================================
# 7. Save Final Checkpoint Summary
# ============================================
println("\n7. Saving final checkpoint summary...")

# Create final summary checkpoint
final_checkpoint_df = DataFrame(
    metric = String[],
    value = Float64[]
)

# Add final metrics
push!(final_checkpoint_df, ("final_epoch", Float64(current_epoch)))
push!(final_checkpoint_df, ("total_iterations", Float64(length(losses))))
push!(final_checkpoint_df, ("final_loss", losses[end]))
push!(final_checkpoint_df, ("min_loss", minimum(losses)))
push!(final_checkpoint_df, ("min_loss_iteration", Float64(argmin(losses))))

# Add training metrics
push!(final_checkpoint_df, ("train_mse", train_mse))
push!(final_checkpoint_df, ("train_mae", train_mae))
push!(final_checkpoint_df, ("train_r2", train_r2))

# Add test metrics (multi-step)
push!(final_checkpoint_df, ("test_mse_multi", test_mse_multi))
push!(final_checkpoint_df, ("test_mae_multi", test_mae_multi))
push!(final_checkpoint_df, ("test_r2_multi", test_r2_multi))

# Add test metrics (1-step)
push!(final_checkpoint_df, ("test_mse_1step", test_mse_1step))
push!(final_checkpoint_df, ("test_mae_1step", test_mae_1step))
push!(final_checkpoint_df, ("test_r2_1step", test_r2_1step))

# Add training info
push!(final_checkpoint_df, ("training_time_seconds", total_time))
push!(final_checkpoint_df, ("training_time_minutes", total_time/60))
push!(final_checkpoint_df, ("n_train_points", Float64(n_train)))
push!(final_checkpoint_df, ("n_test_points", Float64(length(t_test))))

# Save final checkpoint
CSV.write("final_checkpoint.csv", final_checkpoint_df)
println("  Saved: final_checkpoint.csv")

# Also save predictions for potential reuse
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

# ============================================
# 8. Visualization with Checkpoint Info
# ============================================
println("\n8. Creating visualizations...")

# Plot 1: Compartment dynamics with predictions
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
    
    # Mark train/test split
    vline!(plt_results[i], [t_train[end]], color=:gray, ls=:dash, 
           alpha=0.5, label="Train/Test Split", lw=1)
    
    # Formatting
    title!(plt_results[i], compartment_names[i], titlefontsize=10)
    xlabel!(plt_results[i], "Time (days)", guidefontsize=8)
    ylabel!(plt_results[i], "Population", guidefontsize=8)
    
    # Legend only on first subplot
    if i == 1
        plot!(plt_results[i], legend=:topright, legendfontsize=6)
    else
        plot!(plt_results[i], legend=false)
    end
    
    # Grid
    plot!(plt_results[i], grid=true, gridalpha=0.3, minorgrid=true, minorgridalpha=0.1)
end

# Add overall title with checkpoint info
best_loss_iter = argmin(losses)
plot!(plt_results, plot_title="Neural ODE: SIQRDV Model | Final Loss: $(round(losses[end], digits=6)) | Best Loss: $(round(minimum(losses), digits=6)) @ Iter $best_loss_iter", 
      plot_titlefontsize=12)

savefig(plt_results, "results.png")
println("  Saved: results.png")

# Plot 2: Enhanced training loss curve with checkpoints
plt_loss = plot(size=(1000, 600), dpi=100)

# Main loss curve
plot!(plt_loss, 1:length(losses), losses, 
      label="Training Loss", lw=2, color=:blue, yscale=:log10)

# Mark checkpoint saves
checkpoint_iters = checkpoint_interval:checkpoint_interval:length(losses)
checkpoint_losses = [losses[min(i, length(losses))] for i in checkpoint_iters]
scatter!(plt_loss, checkpoint_iters, checkpoint_losses, 
         label="Checkpoints", color=:red, markersize=4, alpha=0.7)

# Mark epoch boundaries
epoch_boundaries = [100, 200, 300, 400, 500, 600]  # Based on our training schedule
for (i, boundary) in enumerate(epoch_boundaries)
    if boundary <= length(losses)
        vline!(plt_loss, [boundary], color=:gray, ls=:dash, alpha=0.3, 
               label= i == 1 ? "Epoch Boundaries" : false, lw=1)
        annotate!(plt_loss, boundary, minimum(losses) * 1.5, 
                 text("Epoch $(i+1)", 8, :gray, rotation=90))
    end
end

# Mark optimizer transition
adam_end = 500
vline!(plt_loss, [adam_end], color=:red, ls=:dash, alpha=0.5, 
       label="ADAM → BFGS", lw=2)

# Add smoothed trend
if length(losses) > 20
    window = 20
    smoothed = [mean(losses[max(1,i-window):min(length(losses),i+window)]) 
                for i in 1:length(losses)]
    plot!(plt_loss, 1:length(losses), smoothed, 
          label="Smoothed (window=$window)", lw=2, color=:orange, alpha=0.7)
end

# Mark best loss
best_iter = argmin(losses)
best_loss = minimum(losses)
scatter!(plt_loss, [best_iter], [best_loss], 
         label="Best Loss: $(round(best_loss, digits=6)) @ Iter $best_iter", 
         color=:green, markersize=8, markershape=:star)

# Formatting
xlabel!(plt_loss, "Iteration")
ylabel!(plt_loss, "Loss (log scale)")
title!(plt_loss, "Neural ODE Training Convergence with Checkpoints\n$(length(checkpoints_df.epoch)) Checkpoints Saved | $(current_epoch) Epochs Total")
plot!(plt_loss, legend=:topright, grid=true, gridalpha=0.3, 
      minorgrid=true, minorgridalpha=0.1)

savefig(plt_loss, "loss.png")
println("  Saved: loss.png")

# ============================================
# 9. Training Summary Report
# ============================================
println("\n9. Training Summary Report:")
println("─"^40)

# Checkpoint analysis
println("\nCheckpoint Statistics:")
println("  Total checkpoints saved: $(nrow(checkpoints_df))")
println("  Best checkpoint:")
best_idx = argmin(checkpoints_df.loss)
best_cp = checkpoints_df[best_idx, :]
println("    - Epoch: $(best_cp.epoch)")
println("    - Iteration: $(best_cp.iteration)")
println("    - Loss: $(round(best_cp.loss, digits=8))")
println("    - Train R²: $(round(best_cp.train_r2, digits=4))")
println("    - Time: $(round(best_cp.time_elapsed, digits=2))s")

# Loss improvement over epochs
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

# Check convergence
recent_losses = losses[max(1, end-19):end]
loss_std = std(recent_losses)
loss_trend = (recent_losses[end] - recent_losses[1]) / length(recent_losses)

println("\nConvergence Analysis (last 20 iterations):")
println("  Mean loss: $(round(mean(recent_losses), digits=8))")
println("  Std dev:   $(round(loss_std, digits=8))")
println("  Trend:     $(round(loss_trend, digits=10))")
println("  Stability: $(loss_std < 1e-4 ? "Excellent ✓" : loss_std < 1e-3 ? "Good ✓" : "Moderate")")

# ============================================
# 10. Final Summary
# ============================================
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
println("  • Best test R² (1-step): $(round(test_r2_1step, digits=4))")
println("  • Training time: $(round(total_time/60, digits=2)) minutes")
println("  • Checkpoints saved: $(nrow(checkpoints_df))")
println("\nCompleted at: $(Dates.now())")
println("="^60)