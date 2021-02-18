@testset "Test Constructors" begin
    @testset "Kalman Filter Constructors" begin
        @test KalmanFilter(x0, F, H, P0, Q, R)
        @test KalmanFilter(x0, F, H, B, P0, Q, R)
        @test KalmanFilter(x0, F, H, B, P0, Q, R, offset)
    end

    @testset "Minimax MMAE Constructors" begin
        @test MinimaxMMAE(filterbank::AbstractVector{KalmanFilter}, γ)
    end

    @testset "Bayesian MMAE Constructors" begin
        @test BayesianMMAE(filterbank::AbstractVector{KalmanFilter}, γ)
    end
end