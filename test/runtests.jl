using MinimaxEstimation
using Ipopt
using Test
using LinearAlgebra

@testset "MinimaxEstimation.jl" begin
    include("testconstructors.jl")
    include("testmethods.jl")
end
