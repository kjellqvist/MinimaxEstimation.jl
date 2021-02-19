@testset "test_methods" begin
    @testset "Kalman Filter methods" begin
            x₀ = [1;0]
            F = [1. 1; 0 -1]
            H = [1 0]
            P₀ = 1. *I(2)
            Q = 1. *I(2)
            R = 2. *I(2)
            y = 2
            kf = KalmanFilter(x₀, F, H, P₀, Q, R)
            update!(kf, y)
            P1exp = [11/5 -3; -3 3]
            @test kf.P == P1exp
            K₀ = [1/5;0]
            xexpected = F*x₀ + K₀*(y-x₀)
            xtrue = predict(kf)
            @test xtrue == xexpected
            pdy = 1 / 2 / (2*π) / sqrt(5) * exp(-1 / 2 * 1 / 5)
            @test kf.pdy[] == pdy
    end

    @testset "Minimax MMAE methods" begin
        @testset "Minimax methods" begin
            x₀ = 1
            F = [1 1; 0 -1]
            H = [1 0]
            P₀ = I(2)
            Q = I(2)
            R = 2*I(2)
            y = 2
            kf1 = KalmanFilter(x₀, F, H, P₀, Q, R)
            kf2 = KalmanFilter(x₀, -F, H, P₀, Q, R)
            γ=10
            x1 = F*x₀ + K₀*(y-x₀)
            x2 = -F*x₀ + -K₀*(y-x₀)
            mini = MinimaxMMAE([kf1, kf2], γ)
            update!(mini, y)
            @test predict(mini.filterbank[1]) == x1
            @test predict(mini.filterbank[2]) == x2
            @test mini.filterbank[1].c[] == 1/5
            @test predict(mini) == 0
        end
    end

    @testset "Bayesian MMAE methods" begin
        @testset "Bayesian methods" begin
            x₀ = [1;0]
            F = [1 1; 0 -1]
            H = [1 0]
            P₀ = I(2)
            Q = I(2)
            R = 2*I(2)
            y = [2]
            kf1 = KalmanFilter(x₀, F, H, P₀, Q, R)
            kf2 = KalmanFilter(x₀, -F, H, P₀, Q, R)
            K = [1/5;0]
            x1 = F*x₀ + K * (y-H*x₀)[1]
            x2 = -F*x₀ + -K*(y-H*x₀)[1]
            bayesian = BayesianMMAE([kf1, kf2])
            update!(mini, y)
            @test predict(bayesian.filterbank[1]) == x1
            @test predict(bayesian.filterbank[2]) == x2
            @test predict(filter::KalmanFilter)
             # Filters have same covariance and estimation error -> equal probability.
            @test bayesian.filterbank[1].pdy[] = 1/2
    end
end