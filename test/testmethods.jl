@testset "test_methods" begin
    @testset "Kalman Filter methods" begin
            x₀ = [1;0]
            F = [1. 1; 0 -1]
            H = [1 0]
            P₀ = 3. *I(2)
            Q = 1. *I(2)
            R = 2. *I(1)
            y = 2
            kf = KalmanFilter(x₀, F, H, P₀, Q, R)
            update!(kf, y)
            P1exp = [26/5 -3; -3 4]
            @test kf.P ≈ P1exp
            K₀ = Matrix{Float64}(undef,2,1)
            K₀[:,:] = [3/5;0]
            xexpected = F*x₀ + K₀*([y]-H*x₀)
            xtrue = predict(kf)
            @test xtrue == xexpected
            pdy = 1 / (2*π) / sqrt(5) * exp(-1 / 2 * 1 / 5)
            @test kf.pdy[] == pdy
    end

    @testset "Minimax MMAE methods" begin
            x₀ = [1;0]
            F = [1. 1; 0 -1]
            H = [1 0]
            P₀ = 3. *I(2)
            Q = 1. *I(2)
            R = 2. *I(1)
            y = 2
            kf1 = KalmanFilter(x₀, F, H, P₀, Q, R)
            kf2 = KalmanFilter(x₀, -F, H, P₀, Q, R)
            γ=10
            K₀ = Matrix{Float64}(undef,2,1)
            K₀[:,:] = [3/5;0]
            x1 = F*x₀ + K₀*([y]-H*x₀)
            x2 = -F*x₀ + -K₀*([y]-H*x₀)
            mini = MinimaxMMAE([kf1, kf2], () ->  Hypatia.Optimizer())
            update!(mini, y)
            @test predict(mini.filterbank[1]) == x1
            @test predict(mini.filterbank[2]) == x2
            @test mini.filterbank[1].c[] == 1/5
           
            yhatreal, valreal = predict(mini, γ)
            P1exp = [26/5 -3; -3 4]
            valexp = x1'*H'*pinv(I - γ^(-2)*H*P1exp*H')*H*x1 - γ^2*kf1.c[]
            @test yhatreal[1] ≈ 0 atol=1e-12
            @test valexp ≈ valreal
    end

    @testset "Bayesian MMAE methods" begin
        x₀ = [1;0]
        F = [1. 1; 0 -1]
        H = [1 0]
        P₀ = 3. *I(2)
        Q = 1. *I(2)
        R = 2. *I(1)
        y = 2
        kf1 = KalmanFilter(x₀, F, H, P₀, Q, R)
        kf2 = KalmanFilter(x₀, -F, H, P₀, Q, R)
        γ=10
        K₀ = Matrix{Float64}(undef,2,1)
        K₀[:,:] = [3/5;0]
        x1 = F*x₀ + K₀*([y]-H*x₀)
        x2 = -F*x₀ + -K₀*([y]-H*x₀)
        bayesian = BayesianMMAE([kf1, kf2])
        update!(bayesian, y)
        @test predict(bayesian.filterbank[1]) == x1
        @test predict(bayesian.filterbank[2]) == x2
        @test predict(bayesian) == [0]
         # Filters have same covariance and estimation error -> equal probability.
        @test bayesian.filterbank[1].pdy[] == 1/2
    end
end