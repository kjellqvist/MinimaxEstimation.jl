@doc raw"""
    update!(filter, y, u)

Update the internal states of the filter in accordance with the output y
and input u.

# Methods
    update!(filter::KalmanFilter, y, u)

Update the internal states of the KalmanFilter in according to
measured output $y$ and controlled input $u$.

The internal states are updated as follows

```math
\begin{aligned}
P_{t+1} & = Q + FP_{t}F^\top \quad - FP_{t}H^\top(R + HP_{t}H^\top)^{-1}H P_{t}F^\top \\
\breve{x}_{t+1}   & = F \breve{x}_{t} + K_{t}(y_t - H\breve{x}_{t})\\
K_{t}     & = FP_{t}H^\top(R + H P_{t} H^\top)^{-1} \\
c_{t+1} & = |H_i\breve x_{t} -y_t|^2_{(R + HP_{t}H^\top)^{-1}} + c_{t} \\
p(i|y_t) & = \frac{1}{(2\pi)^{n/2}\sqrt{|R + HP_tH^\top|}}e^{-1/2 (H\hat x_t - y_t)^\top (R + HP_tH^\top)^{-1}(Hx_t-y_t)}p(i|y_{t-1}).
\end{aligned}
```

The cummulative costs $c_t$ are states used by Minimax Adaptive Estimators, and $p(i|y_t)$ are states used by the standard adaptive estimator.


    update!(filter::MinimaxFilter, y, u)

Update each filter in the internal filterbank of the minimaxfilter object.

    update!(filter::BayesianFilter, y, u)

Update each filter in the internal filterbank of the standard multiple model adaptive filter
and normalize the probability of a given state being active conditioned on past measurements,
$p(i, y_t)$.
"""
function update!(filter::AbstractFilter, y, u)
end

function update!(
    filter::KalmanFilter{T},
    y::AbstractVector{T},
    u::AbstractVector{T},
    ) where {T<:Number}

    x = filter.x
    F = filter.F
    H = filter.H
    B = filter.B
    P = filter.P
    n = length(x)
    offset = filter.offset
    Re = (filter.R + H * P * H')
    K = F * P * H' / Re

    filter.c[] += ((H * x - y)' / Re) * (H * x - y)
    filter.pdy[] *= 1 / (2*π)^(n/2) / sqrt(det(Re)) * exp(-1/2 * ((H*x -y)' / Re)*(H*x - y))
    filter.x[:] = F*x + B*u + K*(y-H*x) + offset
    filter.P[:,:] = filter.Q + F*P*F' - F*P*(H' / Re) * H *P*F'
    return nothing
end

function update!(
    filter::MinimaxMMAE{T},
    y::AbstractVector{T},
    u::AbstractVector{T},
) where {T<:Number}
    for f in filter.filterbank
        update!(f, y, u)
    end

    return nothing
end

function update!(
    filter::BayesianMMAE{T},
    y::AbstractVector{T},
    u::AbstractVector{T},
) where {T<:Number}
    sum_pdys = 0
    for f in filter.filterbank
        update!(f, y, u)
        sum_pdys += f.pdy[]
    end

    for f in filter.filterbank
        f.pdy[] = f.pdy[]/sum_pdys
    end

    return nothing
end

@doc raw"""
    predict(filter)

Predict the states at the next time instance.

# Methods

    xhat = predict(filter::KalmanFilter)

Get the kalman filter prediction of the state at the next time-step, $\hat x_{t+1}$.

    yhat, val = predict(minimaxfilter)

Predict the output of the next time-step as the minimizing argument of
 the quadratically constrainde convex program
```math
J_N^\star(y)= \min_{\hat y} \max_i |\hat y - H_i\breve x_{N,i}|^2_{(I-\gamma^{-2}H_iP_{N,i}H_i^\top)^{-1}} - \gamma^2 c_{N,i}.
```

    yhat = predict(bayesianfilter)

Predict the output at the next time-step as the expected value
`` E[\hat y_{t+1}] = \sum_{i=1}^K \breve y_i p(i|y_t) ``, where $\breve y_i$ are the
corresponding KalmanFilter estimates.
"""
function predict(filter::AbstractFilter)
end

function predict(filter::KalmanFilter)
    return copy(filter.x)
end

function predict(filter::MinimaxMMAE, γ::Real; silent = false)
    γ <= 0 ? DomainError("γ must be positive") : nothing
    m = size(filter.filterbank[1].H)[1]
    K = length(filter.filterbank)

    # Setting up optimization model
    model = Model(filter.optimizer)
    silent ? set_silent(model) : unset_silent(model)
    set_silent(model)
    @variable(model, yhat[1:m])
    @variable(model, t)
    for f in filter.filterbank
        W = (I - γ^(-2)*f.H*f.P*f.H')
        @constraint(
            model,
            (yhat - f.H*f.x)'*pinv(W)*(yhat - f.H*f.x) - γ^2*f.c[] <= t
        )
    end

    @objective(model, Min, t)
    optimize!(model)
    yhat = value.(yhat)
    val = value(t)
    return yhat, val
end

function predict(filter::MinimaxMMAE; γmax = 100, silent = true)
    γmin = sqrt(maximum([maximum(eigvals(kf.H*kf.P*kf.H')) for kf in filter.filterbank]))
    γ = (γmin + γmax)/2
    yhat = []
    val = 0
    for k = 1:20
        yhat, val = predict(filter, γ, silent = silent)
        val < 0 ? γmax = γ : γmin = γ
        γ = (γmin + γmax)/2
    end
    return yhat, val, γ
end

function predict(filter::BayesianMMAE)
    yhat = [f.pdy[]*f.H*f.x for f in filter.filterbank] |> sum
    return yhat
end

# Lazy wrappers

function update!(filter::AbstractFilter{T}, y::Number) where {T<:Number}
    return update!(filter, [T(y)], 0)
end

function update!(filter::AbstractFilter{T}, y::AbstractVector) where {T<:Number}
    return update!(filter, T.(y), 0)
end

function update!(filter::AbstractFilter{T}, y::Number, u::Number) where {T<:Number}
    return update!(filter, [T(y)], [T(u)])
end

function update!(filter::AbstractFilter{T}, y::Number, u::AbstractVector) where {T<:Number}
    return update!(filter, [T(y)], T.(u))
end

function update!(filter::AbstractFilter{T}, y::AbstractVector, u::Number) where {T<:Number}
    return update!(filter, T.(y), [T.(u)])
end

function update!(
    filter::AbstractFilter{T},
    y::AbstractVector,
    u::AbstractVector,
) where {T<:Number}

    return update!(filter, T.(y), T.(u))
end