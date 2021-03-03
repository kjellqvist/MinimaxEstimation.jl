"""
Main module for `MinimaxEstimation.jl`

"""
module MinimaxEstimation
    using LinearAlgebra
    using JuMP

    export KalmanFilter, MinimaxMMAE, BayesianMMAE
    export update!, predict
    export Tsyn

    include("types.jl")
    include("constructors.jl")
    include("methods.jl")
end