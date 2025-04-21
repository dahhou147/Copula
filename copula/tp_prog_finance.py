import numpy as np
from pricing_model import geometric_brownian_motion


def portefeuille(S0, mu, sigma, N, T, M, rf):
    t, S = geometric_brownian_motion(S0, mu, sigma, N, T, M)
    portfolio_value = np.zeros_like(S)
    x = np.zeros_like(S)

    for i in range(N):
        risk_free_asset = np.exp(rf * t[i])
        risky_asset = S[i] / S0
        x[i] = (risky_asset < risk_free_asset).astype(float)
        portfolio_value[i] = x[i] * risk_free_asset + (1 - x[i]) * risky_asset

    return t, portfolio_value
