# Example

The example in the article is restated below, with a self-contained code block to
reproduce the results.
```math
\begin{aligned}
x_{t+1} & = \pm x_t + w_t \\
y_t     & = x_t + v_t \\
x_0     & = 0,
\end{aligned}
```

where $i\in\{-1,1\},\ u_t = sin(t/5)$, $v_t$ and $w_t$ are unit intensity, uncorrelated, Gaussian white noise. Further, $Q = I,\ R = I,\ P_0 = I$. 

```@example Gammas
using Hypatia
using LinearAlgebra
using MinimaxEstimation
using Plots
using Random
using LaTeXStrings

Random.seed!(10)
n_steps = 20
F1 = 1
F2 = -1
F = F1  # The "true" generating system
H = 1
x0 = 0
(Q, R, P0) = (1, 1, 1) # Norm
kf1 = KalmanFilter(x0, F1, H, P0, Q, R)
kf2 = KalmanFilter(x0, F2, H, P0, Q, R)
#mini = MinimaxMMAE([kf1, kf2],Mosek.Optimizer)
mini = MinimaxMMAE([kf1, kf2],() -> Hypatia.Optimizer())
bayes = BayesianMMAE([kf1, kf2])
v = randn((1, n_steps)) # Measurement noise
w = randn((1, n_steps)) # Process disturbance
y = zeros(n_steps+1)    # Noisy output
z = zeros(n_steps+1)    # Hx -> Noise-free output
yhat1 = zeros(n_steps+1)   # 
yhat2 = zeros(n_steps+1)
yhatmini = zeros(n_steps+1,1)
yhatbayes = zeros(n_steps+1,1)
γmins = zeros(n_steps+1)
γminimaxs = zeros(n_steps+1)
γbayes = zeros(n_steps+1)
vals = zeros(n_steps+1)

function bisectgamma(filter::BayesianMMAE; γmax::Real = 100)
    yhat = predict(filter)
    γmin = sqrt(maximum([maximum(eigvals(kf.H*kf.P*kf.H')) for kf in filter.filterbank]))
    γmax < γmin ? error(DomainError((γmax, γmin), "γmax smaller than γmin")) : nothing
    γ = (γmin + γmax)/2 
    for k = 1:20
        J(filter, yhat, γ) < 0 ? γmax = γ : γmin = γ
        γ = (γmin + γmax)/2
    end
    return γ
end

function J(filter::BayesianMMAE, yhat, γ)
    cost = -Inf
    for f in filter.filterbank
        W = (I - γ^(-2)*f.H*f.P*f.H')
        cost = max((yhat - f.H*f.x)'*pinv(W)*(yhat - f.H*f.x) - γ^2*f.c[], cost)
    end
    return cost
end

x = x0
for k=1:n_steps
        # Updates all Kalman filters and calculates the conditional probabilities
    update!(bayes, y[k])   
    yhat1[k+1] = predict(kf1)[1]
    yhat2[k+1] = predict(kf2)[1]
    γmins[k+1] = sqrt(maximum([maximum(eigvals(kf1.P)) maximum(eigvals(kf2.P))]))
    yhatmini[k+1,:], vals[k+1], γminimaxs[k+1] = predict(mini, γmax = 3)
    yhatbayes[k+1,:] = predict(bayes)
    γbayes[k+1] = bisectgamma(bayes, γmax = 3)
    global x = F*x + w[1,k]
    y[k+1] = (H*x)[1] + v[k]
    z[k+1] = (H*x)[1]
end

plot(
    ([γmins γminimaxs γbayes])[2:end,:],
    labels = [L"\max\sqrt{P_{N,i}}" "Minimax" "Bayes"],
    linewidth = 3,
    xlabel = "N",
    ylabel = L"\gamma_N^\star",
    markershape = :o)
savefig("experiment.svg"); nothing # hide
```

![](experiment.svg)

```@example OptimizationProblem
using Hypatia
#using Mosek, MosekTools
using LinearAlgebra
using MinimaxEstimation
using Plots
using Random
using LaTeXStrings

Random.seed!(10)
n_steps = 20
F1 = 1
F2 = -1
F = F1  # The "true" generating system
H = 1
x0 = 0
(Q, R, P0) = (1, 1, 1) # Norm
kf1 = KalmanFilter(x0, F1, H, P0, Q, R)
kf2 = KalmanFilter(x0, F2, H, P0, Q, R)
#mini = MinimaxMMAE([kf1, kf2],Mosek.Optimizer)
mini = MinimaxMMAE([kf1, kf2],() -> Hypatia.Optimizer())
bayes = BayesianMMAE([kf1, kf2])
v = randn((1, n_steps)) # Measurement noise
w = randn((1, n_steps)) # Process disturbance
y = zeros(n_steps+1)    # Noisy output
z = zeros(n_steps+1)    # Hx -> Noise-free output
yhat1 = zeros(n_steps+1)   # 
yhat2 = zeros(n_steps+1)
yhatmini = zeros(n_steps+1,1)
yhatbayes = zeros(n_steps+1,1)
γmins = zeros(n_steps+1)
γminimaxs = zeros(n_steps+1)
γbayes = zeros(n_steps+1)
vals = zeros(n_steps+1)

function bisectgamma(filter::BayesianMMAE; γmax::Real = 100)
    yhat = predict(filter)
    γmin = sqrt(maximum([maximum(eigvals(kf.H*kf.P*kf.H')) for kf in filter.filterbank]))
    γmax < γmin ? error(DomainError((γmax, γmin), "γmax smaller than γmin")) : nothing
    γ = (γmin + γmax)/2 
    for k = 1:20
        J(filter, yhat, γ) < 0 ? γmax = γ : γmin = γ
        γ = (γmin + γmax)/2
    end
    return γ
end

function J(filter::BayesianMMAE, yhat, γ)
    cost = -Inf
    for f in filter.filterbank
        W = (I - γ^(-2)*f.H*f.P*f.H')
        cost = max((yhat - f.H*f.x)'*pinv(W)*(yhat - f.H*f.x) - γ^2*f.c[], cost)
    end
    return cost
end

x = x0
for k=1:5
    # Updates all Kalman filters and calculates the conditional probabilities
    update!(bayes, y[k])   
    yhat1[k+1] = predict(kf1)[1]
    yhat2[k+1] = predict(kf2)[1]
    global x = F*x + w[1,k]
    y[k+1] = (H*x)[1] + v[k]
    z[k+1] = (H*x)[1]
end

yhat1 = H*predict(kf1)
yhat2 = H*predict(kf2)
f1 = x-> (yhat1[1] - x)^2/(1-γ^(-2)*kf1.P[1,1]) - γ^2*kf1.c[]
f2 = x-> (yhat2[1] - x)^2/(1-γ^(-2)*kf2.P[1,1]) - γ^2*kf2.c[]

yhatmini, Jmini, γ = predict(mini, γmax = 3)
yhatbayes = predict(bayes)
Jbayes = J(bayes, yhatbayes, γ)
lower = -2
upper = 0
plt = plot(lower:0.01:upper, f1, linestyle = :dash, linewidth = 2, labels = L"J^+_5")
plot!(plt, lower:0.01:upper, f2, linestyle = :dash, linewidth = 2, labels = L"J^-_5")
plot!(plt, lower:0.01:upper, x -> max(f1(x), f2(x)), linecolor = :black, linewidth = 2, labels = L"\max\{J^+_5, J^-_5\}")
plot!(plt,yhatmini, [Jmini], marker = :o, markersize = 8, markercolor = :blue, labels = "Minimax")
plot!(plt,yhatbayes, [Jbayes], marker = :x, markersize = 8, markercolor = :green, labels = "Bayes")
xlabel!(L"\hat y_5")
ylabel!(L"J_5")
savefig("experiment_2.svg"); nothing # hide
```

![](experiment_2.svg)