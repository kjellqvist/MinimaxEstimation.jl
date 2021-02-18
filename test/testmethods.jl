@testset "test_methods" begin
    @testset "Kalman Filter methods" begin
        @testset "Kalman Update" begin
            @test update!(filter::KalmanFilter{T}, y::AbstractVector{T}, u::AbstractVector{T})
        end

        @testset "Kalman Predict" begin
            @test predict!(filter::KalmanFilter)
        end
    end

    @testset "Minimax MMAE methods" begin
        @testset "Minimax Update" begin
            @test update!(filter::KalmanFilter{T}, y::AbstractVector{T}, u::AbstractVector{T})
        end

        @testset "Minimax Predict" begin
            @test predict!(filter::KalmanFilter)
        end
    end

    @testset "Bayesian MMAE methods" begin
        @testset "Bayesian Update" begin
            @test update!(filter::KalmanFilter{T}, y::AbstractVector{T}, u::AbstractVector{T})
        end

        @testset "Bayesian Predict" begin
            @test predict!(filter::KalmanFilter)
        end
    end
end