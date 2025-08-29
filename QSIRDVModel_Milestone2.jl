using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, Plots, ComponentArrays
using JLD2, Statistics

Random.seed!(1234)
rng = Random.default_rng()

println("NEURAL ODE FOR SIQRDV MODEL")
println("="^60)

# ============================================
# 1. Generate Data
# ============================================
u0 = [990.0, 10.0, 0.0, 0.0, 0.0, 0.0]  # S, I, Q, R, D, V
tspan = (0.0, 160.0)
t = range(0, 160, length=161)
p_true = [0.3, 0.05, 0.1, 0.08, 0.01, 0.005, 0.02]

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

# Split data: 75% train, 25% test
n_train = 121  # 75% of 161
t_train = t[1:n_train]
t_test = t[n_train+1:end]
train_data = ode_data[:, 1:n_train]
test_data = ode_data[:, n_train+1:end]

println("Data: $(n_train) train, $(length(t_test)) test points")

# ============================================
# 2. Neural ODE Setup
# ============================================
dudt_nn = Lux.Chain(
    Lux.Dense(6, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 6)
)

p_nn, st_nn = Lux.setup(rng, dudt_nn)
println("Network: 6 → 32 → 32 → 6")

# Training prediction
prob_train = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5(), saveat=t_train)

function predict_train(p)
    Array(prob_train(u0, p, st_nn)[1])
end

# Simple weighted loss
function loss_train(p)
    pred = predict_train(p)
    
    # Higher weight on Infected compartment
    weights = [1.0, 5.0, 2.0, 1.0, 1.0, 1.0]
    loss = 0.0
    
    for i in 1:6
        loss += weights[i] * sum(abs2, train_data[i,:] .- pred[i,:])
    end
    
    return loss / length(train_data)
end

# ============================================
# 3. Training
# ============================================
println("\nTraining...")

losses = Float64[]
training_callback = function(state, l)
    push!(losses, l)
    if length(losses) % 100 == 0
        println("  Iter $(length(losses)): Loss = $(round(l, digits=4))")
    end
    return false
end

# Initialize and train
pinit = ComponentArray(p_nn)
optf = Optimization.OptimizationFunction((x, p) -> loss_train(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, pinit)

# ADAM
result1 = Optimization.solve(optprob, ADAM(0.01), callback=training_callback, maxiters=500)

# BFGS refinement
optprob2 = remake(optprob, u0=result1.u)
result2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.001),
                             callback=training_callback, maxiters=200)

final_params = result2.u

# ============================================
# 4. Forecasting
# ============================================
println("\nForecasting...")

# Training fit
train_pred = predict_train(final_params)

# Multi-step forecast
u_last = train_data[:, end]
prob_test = NeuralODE(dudt_nn, (t_test[1], t_test[end]), Tsit5(), saveat=t_test)
test_forecast = Array(prob_test(u_last, final_params, st_nn)[1])

# 1-step forecast
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
# 5. Calculate Errors
# ============================================
function calc_mse(true_data, pred_data)
    sum(abs2, true_data .- pred_data) / length(true_data)
end

train_mse = calc_mse(train_data, train_pred)
test_mse_multi = calc_mse(test_data, test_forecast)
test_mse_1step = calc_mse(test_data, test_1step)

println("\nResults:")
println("  Train MSE: $(round(train_mse, digits=6))")
println("  Test MSE (multi-step): $(round(test_mse_multi, digits=6))")
println("  Test MSE (1-step): $(round(test_mse_1step, digits=6))")

# ============================================
# 6. Save Checkpoint
# ============================================
checkpoint = Dict(
    "params" => final_params,
    "train_mse" => train_mse,
    "test_mse_multi" => test_mse_multi,
    "test_mse_1step" => test_mse_1step
)
@save "checkpoint.jld2" checkpoint
println("\nSaved: checkpoint.jld2")

# ============================================
# 7. Plotting
# ============================================
plt = plot(layout=(3,2), size=(1200, 800))
names = ["S", "I", "Q", "R", "D", "V"]
colors = [:blue, :red, :orange, :green, :purple, :brown]

for i in 1:6
    # Truth
    plot!(plt[i], t, ode_data[i,:], 
          label="Truth", lw=3, color=:black, alpha=0.7)
    
    # Training fit
    plot!(plt[i], t_train, train_pred[i,:], 
          label="Train", lw=2, color=colors[i])
    
    # Multi-step forecast
    plot!(plt[i], t_test, test_forecast[i,:], 
          label="Forecast", lw=2, color=colors[i], ls=:dash)
    
    # 1-step forecast
    plot!(plt[i], t_test, test_1step[i,:], 
          label="1-step", lw=2, color=colors[i], ls=:dot, alpha=0.6)
    
    # Train/test split
    vline!(plt[i], [t_train[end]], color=:gray, ls=:dash, alpha=0.5, label=false)
    
    title!(plt[i], names[i])
    xlabel!(plt[i], "Time")
    ylabel!(plt[i], "Population")
    
    plot!(plt[i], legend=(i==1 ? :best : false), legendfontsize=6)
end

savefig(plt, "results.png")

# Loss curve
plt_loss = plot(losses, xlabel="Iteration", ylabel="Loss", 
                title="Training Loss", lw=2, yscale=:log10)
savefig(plt_loss, "loss.png")

# ============================================
# 8. Training Stability Note
# ============================================
println("\nTraining Stability:")
println("  - Solver: Tsit5() (5th order Runge-Kutta)")
println("  - ADAM: 500 iters, lr=0.01")
println("  - BFGS: 200 iters, stepnorm=0.001")
println("  - Final loss: $(round(losses[end], digits=6))")
println("  - Convergence: $(std(losses[end-10:end]) < 1e-3 ? "Good" : "Moderate")")

println("\nDone! Files: results.png, loss.png, checkpoint.jld2")
println("="^60)