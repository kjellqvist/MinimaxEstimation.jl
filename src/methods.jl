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

function predict(filter::KalmanFilter)
    return copy(filter.x)
end

function predict(filter::MinimaxMMAE, silent = false)
    m = size(filter.filterbank[1].H)[1]
    K = length(filter.filterbank)
    γ = filter.γ

    # Setting up optimization model
    model = Model(filter.optimizer)
    silent ? set_silent(model) : unset_silent(model)
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