abstract type AbstractFilter{T} end


struct KalmanFilter{T<:AbstractFloat} <: AbstractFilter{T}
    x::AbstractVector{T}
    F::AbstractMatrix{T}
    H::AbstractMatrix{T}
    B::AbstractMatrix{T}
    P::AbstractMatrix{T}
    Q::AbstractMatrix{T}
    R::AbstractMatrix{T}
    offset::AbstractVector{T}
    c::Base.RefValue{T}
    pdy::Base.RefValue{T}
end

struct MinimaxMMAE{T<:AbstractFloat} <: AbstractFilter{T}
    filterbank::AbstractVector{KalmanFilter{T}}
    gamma::Real
    optimizer::Function

    function MinimaxMMAE(
        filterbank::AbstractVector{KalmanFilter{T}}, 
        γ::Real, 
        optimizer::Function) where T<: AbstractFloat
        γ <= 0 ? DomainError("γ must be positive") : nothing
        new{T}(filterbank, γ, optimizer)
    end
end

struct BayesianMMAE{T<:AbstractFloat} <: AbstractFilter{T}
    filterbank::AbstractVector{KalmanFilter{T}}
    function BayesianMMAE(filterbank::AbstractVector{KalmanFilter{T}}) where T<: AbstractFloat
        K = length(filterbank)
        for filter in filterbank
            filter.pdy[] = 1/K
        end
        new{T}(filterbank)
    end
    
end