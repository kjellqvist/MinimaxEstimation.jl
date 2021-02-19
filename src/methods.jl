
# TODO: #5 Implement methods
function update!(filter::KalmanFilter{T}, y::AbstractVector{T}, u::AbstractVector{T}) where T<: Number
    return nothing
end

function update!(filter::MinimaxMMAE{T}, y::AbstractVector{T}, u::AbstractVector{T}) where T <: Number
    return nothing
end

function update!(filter::BayesianMMAE{T}, y::AbstractVector{T}, u::AbstractVector{T}) where T <: Number
    return nothing
end

function predict!(filter::KalmanFilter)
    return nothing
end

function predict!(filter::MinimaxMMAE)
    return nothing
end

function predict!(filter::BayesianMMAE)
    return nothing
end

function Tsyn(filterbank::AbstractVector{KalmanFilter}, γ::Real)
    return nothing
end


###############################################################
######################## Lazy Wrappers ########################
###############################################################

function update!(filter::AbstractFilter{T}, y::Number)  where T<: Number
    p = size(filter.filterbank[1].B)[2]
    return update!(filter, [T(y)], zeros(T,p))
end

function update!(filter::AbstractFilter{T}, y::AbstractVector)  where T<: Number
    p = size(filter.filterbank[1].B)[2]
    return update!(filter, T.(y), zeros(T,p))
end

function update!(filter::AbstractFilter{T}, y::Number, u::Number)  where T<: Number
    return update!(filter, [T(y)], [T(u)])
end

function update!(filter::AbstractFilter{T}, y::Number, u::AbstractVector)  where T<: Number
    return update!(filter, [T(y)], T.(u))
end

function update!(filter::AbstractFilter{T}, y::AbstractVector, u::Number)  where T<: Number
    return update!(filter, T.(y), [T.(u)])
end

function update!(filter::AbstractFilter{T}, y::AbstractVector, u::AbstractVector)  where T<: Number
    return update!(filter, T.(y), T.(u))
end