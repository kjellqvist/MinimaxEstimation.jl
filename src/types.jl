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
    γ::Real
    optimizer::Function
    
    @doc raw"""
        mini = MinimaxMMAE(filterbank, γ, optimizer)
    Construct a Minimax Multiple Model Adaptive Estimator object.

    Create an estimator for the case where the dynamics belong to 
    the finite set of linear systems
    equations
    ```math
    \begin{aligned}
    x_{t+1} & = F_ix_t + B_iu_t + w_t + \text{offset}_i\\
    y_t     & = H_ix_t + v_t,
    \end{aligned}
    ```
    and the associated objective
    ```math
    \min_{\hat y_N}\max_{x_0, \mathbf{w}^N, \mathbf{v}^N, i}\Bigg\{ |\hat y_N - H_ix_N|^2 - \gamma^2\left(|x_0 - \hat x_0|^2_{P_0^{-1}} + \sum_{t=0}^{N-1} |w_t|^2_{Q^{-1}} + |v_t|^2_{R^{-1}}\right)\Bigg\}.
    ```

    Filterbank is an array of KalmanFilter object associated with each $i$.
    """
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

    @doc raw"""
        mmae = BayesianMMAE(filterbank)
    Construct a standard Multiple Model Adaptive Estimator object.

    Create an estimator for the case where the dynamics belong to 
    the finite set of linear systems
    equations
    ```math
    \begin{aligned}
    x_{t+1} & = F_ix_t + B_iu_t + w_t + \text{offset}_i\\
    y_t     & = H_ix_t + v_t,
    \end{aligned}
    ```
    filterbank is an array of KalmanFilter objects associated with the
    dynamics and disturbance characteristics of each $i$.
    """
    function BayesianMMAE(filterbank::AbstractVector{KalmanFilter{T}}) where T<: AbstractFloat
        K = length(filterbank)
        for filter in filterbank
            filter.pdy[] = 1/K
        end
        new{T}(filterbank)
    end
    
end