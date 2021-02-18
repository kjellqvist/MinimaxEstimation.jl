abstract type AbstractFilter{T} end


struct KalmanFilter{T<:Number} <: AbstractFilter{T}
    x::AbstractVector{T}
    F::AbstractMatrix{T}
    H::AbstractMatrix{T}
    B::AbstractMatrix{T}
    P0::AbstractMatrix{T}
    Q::AbstractMatrix{T}
    R::AbstractMatrix{T}
    offset::AbstractVector{T}
    c::Base.RefValue{Real}
    pdy::Base.RefValue{Real}
end

struct MinimaxMMAE{T<:Number} <: AbstractFilter{T}
    filterbank::AbstractVector{KalmanFilter{T}}
    gamma::Real
    optimizer::Function
end

struct BayesianMMAE{T<:Number} <: AbstractFilter{T}
    filterbank::AbstractVector{KalmanFilter{T}}
end