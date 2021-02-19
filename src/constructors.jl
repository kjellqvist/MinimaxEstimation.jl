# TODO: #4 Implement constructors
function KalmanFilter(
    x::AbstractVector{T},
    F::AbstractMatrix{T},
    H::AbstractMatrix{T},
    B::AbstractMatrix{T},
    P::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    R::AbstractMatrix{T},
    offset::AbstractVector{T}
    ) where T<: AbstractFloat

    n = size(x)[1]
    m = size(H)[1]
    p = size(B)[2]
    # Validate Args
    size(F)[1] != n ? DimensionMismatch("Dimension mismatch of F") : nothing
    size(H)[2] != n ? DimensionMismatch("Dimension mismatch of H") : nothing
    size(B)[1] != n ? DimensionMismatch("Dimension mismatch of B") : nothing
    size(P)[1] != n ? DimensionMismatch("P and F incompatible") : nothing
    size(Q)[1] != n ? DimensionMismatch("Q and F incompatible") : nothing
    size(R)[1] != n ? DimensionMismatch("R and F incompatible") : nothing
    isposdef(P) ? DomainError("P must be positive-definite") : nothing
    isposdef(Q) ? DomainError("Q must be positive-definite") : nothing
    isposdef(R) ? DomainError("R must be positive-definite") : nothing

    c = Base.RefValue{Float64}(0)
    pdy = Base.RefValue{Float64}(1)
    x0 = copy(x)
    F = copy(F)
    H = copy(H)
    B = copy(B)
    P = copy(P)
    Q = copy(Q)
    R = copy(R)
    offset = copy(offset)
    KalmanFilter(x0, F, H, B, P, Q, R, offset, c, pdy)
end

function MinimaxMMAE(filterbank::AbstractVector{KalmanFilter}, γ)
end

function BayesianMMAE(filterbank::AbstractVector{KalmanFilter})
end


# lazy wrappers
function KalmanFilter(x0, F, H, P0, Q, R)
end

function KalmanFilter(x0, F, H, B, P0, Q, R)
end

function KalmanFilter(x0, F, H, B, P0, Q, R, offset)
end