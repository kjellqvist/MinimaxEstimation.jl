@testset "Test Constructors" begin
    @testset "Kalman Filter Constructors" begin
        x₀ = 1
        F = 2
        H = 3
        P₀ = 5
        Q = 7
        R = 11
        kf = KalmanFilter(x₀, F, H, P₀, Q, R)
        trues = Tuple([fill(el,1,1) for el in (x₀, F, H, P₀, Q, R)])
        @test (x₀, F, H, P₀, Q, R) == (kf.x₀, kf.F, kf.H, kf.P₀, kf.Q, kf.R)

        # Promotion
        F = Float16([0.5 1; 0 1])
        B = Int32.([1;0])
        H = [1 0]
        P₀ = I(2)
        Q = 10*I(2)
        R = 0.7*I(2)
        kf = KalmanFilter(x₀, F, H, B, P₀, Q, R)
        @test typeof(kf.B == Matrix{Float64})
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
        reality = (kf.x₀, kf.F, kf.H, kf.B, kf.P₀, kf.Q, kf.R, kf.offset)

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
        γ=10
        mini = MinimaxMMAE([kf1, kf2], γ, () ->  Hypatia.Optimizer())
        @test mini.filterbank[1].F == -mini.filterbank[2].F
        @test predict(mini) == [0;0]
        bayesian = BayesianMMAE([kf1, kf2])
        @test bayesian.filterbank[1].pdy[] = 1/2
        @test bayesian.filterbank[1].F == -bayesisan.filterbank[2].F
        @test predict(bayesian) == [0;0]
        # Filters have same covariance and estimation error -> equal probability.
        @test bayesian.filterbank[1].pdy[] = 1/2
    end
end