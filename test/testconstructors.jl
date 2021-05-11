@testset "Test Constructors" begin
    @testset "Kalman Filter Constructors" begin
        x₀ = 1
        F = 2
        H = 3
        P₀ = 5
        Q = 7
        R = 11
        kf = KalmanFilter(x₀, F, H, P₀, Q, R)
        @test typeof(kf) == KalmanFilter{Float64}
        # Promotion
        x₀ = [2;0]
        F = Float32.([0.5 1; 0 1])
        B = Int32.([1;0])
        H = [1 0]
        P₀ = I(2)
        Q = 10*I(2)
        R = 0.7*I(1)
        kf = KalmanFilter(x₀, F, H, B, P₀, Q, R)
        @test typeof(kf.B) == Matrix{Float64}
        @test kf.pdy[] == 1

        x₀ = [0.;2.]
        F = [1. 1;0 1]
        H = [1. 0]
        B = Matrix{Float64}(undef, 2,1)
        B[:,:] = [0; 1.]
        P₀ = [10. 0; 0 10.]
        Q = [1. 0; 0 1.]
        R = fill(1.,1,1)
        offset = [1.;1.]
        kf = KalmanFilter(x₀, F, H, B, P₀, Q, R, offset)
        expected = (x₀, F, H, B, P₀, Q, R, offset)
        reality = (kf.x, kf.F, kf.H, kf.B, kf.P, kf.Q, kf.R, kf.offset)

        @test expected==reality
    end

    @testset "Minimax & Bayesian MMAE Constructors" begin
        x₀ = [0.;2.]
        F = [1. 1;0 1]
        H = [1. 0]
        B = Matrix{Float64}(undef, 2,1)
        B[:,:] = [0; 1.]
        P₀ = [10. 0; 0 10.]
        Q = [1. 0; 0 1.]
        R = fill(1.,1,1)
        offset = [1.;1.]
        kf1 = KalmanFilter(x₀, F, H, B, P₀, Q, R, offset)   
        kf2 = KalmanFilter(x₀, -F, H, B, P₀, Q, R, offset)
        mini = MinimaxMMAE([kf1, kf2], () ->  Hypatia.Optimizer())
        @test mini.filterbank[1].F == -mini.filterbank[2].F
        bayesian = BayesianMMAE([kf1, kf2])
        @test bayesian.filterbank[1].pdy[] == 1/2
        @test bayesian.filterbank[1].F == -bayesian.filterbank[2].F
    end
end