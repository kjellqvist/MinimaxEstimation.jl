# TODO: #4 Implement constructors
function KalmanFilter(
    x::AbstractVector{T},
    F::AbstractMatrix{T},
    H::AbstractMatrix{T},
    B::AbstractMatrix{T},
    P::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    R::AbstractMatrix{T},
    offset::AbstractVector{T},
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

    c = Base.RefValue{T}(0)
    pdy = Base.RefValue{T}(1)
    x0 = copy(x)
    F = copy(F)
    H = copy(H)
    B = copy(B)
    P = copy(P)
    Q = copy(Q)
    R = copy(R)
    offset = copy(offset)
    KalmanFilter{T}(x0, F, H, B, P, Q, R, offset, c, pdy)
end





# lazy wrappers
function KalmanFilter(x0, F, H, P0, Q, R)
    size(x0) == () ? x0 = [x0] : nothing
    B = zeros(eltype(x0),size(x0,1))
    KalmanFilter(x0, F, H, B, P0, Q, R)
end

function KalmanFilter(x0, F, H, B, P0, Q, R)
    size(x0) == () ? x0 = [x0] : nothing
    offset = zeros(eltype(x0),size(x0))
    KalmanFilter(x0, F, H, B, P0, Q, R, offset)
end

function KalmanFilter(x0, F, H, B, P0, Q, R, offset)
    size(x0) == () ? x0 = [x0] : nothing
    size(F) == () ? F = fill(F,1,1) : nothing
    size(H) == () ? H = fill(H,1,1) : nothing
    size(B) == () ? B = fill(B,1,1) :
        typeof(B) <: AbstractVector ? B = reshape(B, (:,1)) : nothing
    size(P0) == () ? P0 = fill(P0, 1,1) : nothing
    size(Q) == () ? Q = fill(Q, 1,1) : nothing
    size(R) == () ? R = fill(R, 1,1) : nothing
    size(offset) == () ? offset = [offset] : nothing

    x0, F, H, B, P0, Q, R, offset = promote_arrays(x0, F, H, B, P0, Q, R, offset)
    
    return KalmanFilter(x0, F, H, B, P0, Q, R, offset)

end

# Utility functions
function promote_arrays(arrays...)
    eltype = Base.promote_eltype(arrays...)
    eltype <: AbstractFloat ? nothing : eltype = Float64
    return tuple([convert(Array{eltype}, array) for array in arrays]...)
end