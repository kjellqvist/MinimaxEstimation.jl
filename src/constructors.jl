function KalmanFilter(x0, F, H, P0, Q, R)
end

function KalmanFilter(x0, F, H, B, P0, Q, R)
end

function KalmanFilter(x0, F, H, B, P0, Q, R, offset)
end

function MinimaxMMAE(filterbank::AbstractVector{KalmanFilter}, γ)
end

function BayesianMMAE(filterbank::AbstractVector{KalmanFilter})
end