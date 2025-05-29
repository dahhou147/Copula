# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.stats as ss


class GeometricBrownianMotion:
    """Class for generating geometric Brownian motion paths."""

    def __init__(self, S0: float, mu: float, sigma: float, N: int, T: float, M: int):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.N = N
        self.T = T
        self.M = M

    def generate_paths(self):
        """Generate geometric Brownian motion paths."""
        dt = self.T / self.N
        t = np.linspace(0, self.T, self.N)
        dW = ss.norm.rvs(scale=np.sqrt(dt), size=(self.N - 1, self.M))
        W = np.cumsum(dW, axis=0)
        W = np.vstack([np.zeros(self.M), W])
        S = self.S0 * np.exp(
            (self.mu - 0.5 * self.sigma**2) * t[:, None] + self.sigma * W
        )
        return t, S


class BlackScholesPricer:
    """Class for Black-Scholes option pricing."""

    def __init__(
        self, S0: float, K: float, T: float, sigma: float, r: float, q: float = 0.0
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q

    def _d1_d2(self, S, tau):
        """Compute d1 and d2 for Black-Scholes formula."""
        d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )
        d2 = d1 - self.sigma * np.sqrt(tau)
        return d1, d2

    def price_call(self):
        """Black-Scholes call price."""
        d1, d2 = self._d1_d2(self.S0, self.T)
        return self.S0 * np.exp(-self.q * self.T) * ss.norm.cdf(d1) - self.K * np.exp(
            -self.r * self.T
        ) * ss.norm.cdf(d2)

    def price_put(self):
        """Black-Scholes put price."""
        d1, d2 = self._d1_d2(self.S0, self.T)
        return self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.S0 * np.exp(
            -self.q * self.T
        ) * ss.norm.cdf(-d1)

    def delta(self, S, tau, option_type="call"):
        """Calculate delta of the option."""
        d1, _ = self._d1_d2(S, tau)
        if option_type == "call":
            return np.exp(-self.q * self.T) * ss.norm.cdf(d1)
        if option_type == "put":
            return -np.exp(-self.q * self.T) * ss.norm.cdf(-d1)
        raise ValueError("option_type must be 'call' or 'put'")


class VolatilitySmile:
    """Class for volatility smile calculations."""

    def __init__(self, pricer: BlackScholesPricer):
        self.pricer = pricer

    def implied_volatility(self, strike: float, market_price: float):
        """Calculate implied volatility using Newton-Raphson method."""

        def f(sigma):
            self.pricer.sigma = sigma
            self.pricer.K = strike
            return self.pricer.price_call() - market_price

        try:
            return so.newton(f, 0.2, maxiter=50)
        except RuntimeError:
            return np.nan

    def volatility_smile(self, strikes: np.ndarray, market_prices: np.ndarray):
        """Calculate the implied volatility curve."""
        return np.array(
            [
                self.implied_volatility(strike, price)
                for strike, price in zip(strikes, market_prices)
            ]
        )

    def plot_smile(self, strikes, market_prices):
        """Plot the implied volatility curve."""
        smile = self.volatility_smile(strikes, market_prices)
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, smile * 100, "o-")
        plt.xlabel("strike price")
        plt.ylabel("implied volatility (%)")
        plt.title("Volatility Smile")
        plt.grid(True)
        return plt.gcf()


class DeltaHedger:
    """Class for delta hedging simulation."""

    def __init__(self, pricer: BlackScholesPricer, paths: np.ndarray, N: int, T: float):
        self.pricer = pricer
        self.paths = paths
        self.N = N
        self.T = T

    def hedge(self, option_type="call"):
        """Perform delta hedging simulation."""
        pnl = np.zeros((self.N, self.paths.shape[1]))
        dt = self.T / self.N

        for i in range(self.paths.shape[1]):
            S0 = self.paths[0, i]
            tau = self.T
            delta_old = self.pricer.delta(S0, tau, option_type)
            option_value_old = (
                self.pricer.price_call()
                if option_type == "call"
                else self.pricer.price_put()
            )
            cash_position = option_value_old - delta_old * S0
            portfolio_value = delta_old * S0 + cash_position

            for j in range(1, self.N):
                St = self.paths[j, i]
                tau = self.T - j * dt
                delta_new = self.pricer.delta(St, tau, option_type)
                delta_diff = delta_new - delta_old
                cash_position *= np.exp(self.pricer.r * dt)
                cash_position -= delta_diff * St
                portfolio_value = delta_new * St + cash_position
                option_value = (
                    self.pricer.price_call()
                    if option_type == "call"
                    else self.pricer.price_put()
                )
                pnl[j, i] = portfolio_value - option_value
                delta_old = delta_new

        return pnl


class GirsanovSimulator:
    """Class for simulating paths under risk-neutral measure using Girsanov theorem."""

    def __init__(
        self, S0: float, mu: float, r: float, sigma: float, N: int, T: float, M: int
    ):
        self.S0 = S0
        self.mu = mu
        self.r = r
        self.sigma = sigma
        self.N = N
        self.T = T
        self.M = M

    def generate_paths(self):
        """Generate paths under risk-neutral measure."""
        dt = self.T / self.N
        t = np.linspace(0, self.T, self.N)

        theta = (self.mu - self.r) / self.sigma
        dW = ss.norm.rvs(scale=np.sqrt(dt), size=(self.N - 1, self.M))
        dW_tilde = dW - theta * np.sqrt(dt)

        W_tilde = np.cumsum(dW_tilde, axis=0)
        W_tilde = np.vstack([np.zeros(self.M), W_tilde])

        return self.S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * t[:, None] + self.sigma * W_tilde
        )


# %%
if __name__ == "__main__":
    pass
