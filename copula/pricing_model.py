# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.stats as ss


def geometric_brownian_motion(S0, mu, sigma, N, T, M):
    """Generate geometric Brownian motion paths.

    Args:
        S0 (float): Initial stock price
        mu (float): Drift
        sigma (float): Volatility
        N (int): Number of time steps
        T (float): Time horizon
        M (int): Number of paths
    """
    dt = T / N
    t = np.linspace(0, T, N)
    dW = ss.norm.rvs(scale=np.sqrt(dt), size=(N - 1, M))
    W = np.cumsum(dW, axis=0)
    W = np.vstack([np.zeros(M), W])
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t[:, None] + sigma * W)
    return t, S


class EuropeanOptions:
    def __init__(
        self,
        S0: float,
        strike_price: float,
        maturity: float,
        sigma: float,
        mu: float,
        r: float,
        dividend: float = 0.0,
        N: int = 252,
        M: int = 1000,
    ):
        """Initialize the European Option with Smile Volatility.

        Args:
            S0: Initial stock price
            strike_price: Strike price
            maturity: Time to maturity
            sigma: Volatility
            r: Risk-free rate
            dividend: Dividend yield
            N: Number of time steps
            M: Number of Monte Carlo paths
        """
        self.S0 = S0
        self.K = strike_price
        self.T = maturity
        self.sigma = sigma
        self.r = r
        self.q = dividend
        self.N = N
        self.mu = mu
        self.M = M
        self.S = None

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

    def implied_volatility(
        self, strike, market_price: float, option_type: str = "call"
    ):
        """Calculate implied volatility using Newton-Raphson method.

        Args:
            strike: Strike price
            market_price: Observed market price
            option_type: Type of option ('call' or 'put')
        """

        def f(sigma):
            self.sigma = sigma
            self.K = strike
            return (
                self.price_call() if option_type == "call" else self.price_put()
            ) - market_price

        try:
            return so.newton(f, 0.2, maxiter=50)
        except RuntimeError:
            return np.nan

    def volatility_smile(
        self, strikes: np.ndarray, market_prices: np.ndarray, option_type: str = "call"
    ) -> np.ndarray:
        """Calculate the implied volatility curve."""
        return np.array(
            [
                self.implied_volatility(strike, price, option_type)
                for strike, price in zip(strikes, market_prices)
            ]
        )

    def plot_volatility_smile(self, strikes, market_prices, option_type="call"):
        """Plot the implied volatility curve."""
        smile = self.volatility_smile(strikes, market_prices, option_type)
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, smile * 100, "o-")
        plt.xlabel("Prix d'exercice")
        plt.ylabel("Volatilité implicite (%)")
        plt.title("Smile de volatilité")
        plt.grid(True)
        return plt.gcf()

    def generate_paths(self):
        """Generate paths for the underlying asset."""
        self.S = self.generate_path_girsanov()
        return self.S

    def delta(self, S, tau, option_type="call"):
        """Calculate delta of the option."""
        d1, _ = self._d1_d2(S, tau)
        if option_type == "call":
            return np.exp(-self.q * self.T) * ss.norm.cdf(d1)
        elif option_type == "put":
            return -np.exp(-self.q * self.T) * ss.norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def call_in_time(self, S, tau):
        """Calculate the call option price in time."""
        d1, d2 = self._d1_d2(S, tau)
        return np.exp(-self.q * tau) * (
            S * ss.norm.cdf(d1) - self.K * np.exp(-self.r * tau) * ss.norm.cdf(d2)
        )

    def put_in_time(self, S, tau):
        """Calculate the put option price in time."""
        d1, d2 = self._d1_d2(S, tau)
        return np.exp(-self.q * tau) * (
            self.K * np.exp(-self.r * tau) * ss.norm.cdf(-d2) - S * ss.norm.cdf(-d1)
        )

    def delta_hedging(self, option_type="call"):
        """Perform delta hedging simulation.

        Args:
            option_type: Type of option ('call' or 'put')

        Returns:
            np.ndarray: Array of PnL values
        """
        if self.S is None:
            self.generate_paths()

        pnl = np.zeros((self.N, self.M))
        dt = self.T / self.N

        for i in range(self.M):
            S0 = self.S[0, i]
            tau = self.T
            delta_old = self.delta(S0, tau, option_type)
            option_value_old = (
                self.call_in_time(S0, tau)
                if option_type == "call"
                else self.put_in_time(S0, tau)
            )
            cash_position = option_value_old - delta_old * S0
            portfolio_value = delta_old * S0 + cash_position
            for j in range(1, self.N):
                St = self.S[j, i]
                tau = self.T - j * dt
                delta_new = self.delta(St, tau, option_type)
                delta_diff = delta_new - delta_old
                cash_position *= np.exp(self.r * dt)
                cash_position -= delta_diff * St
                portfolio_value = delta_new * St + cash_position
                option_value = (
                    self.call_in_time(St, tau)
                    if option_type == "call"
                    else self.put_in_time(St, tau)
                )
                pnl[j, i] = portfolio_value - option_value
                delta_old = delta_new

        return pnl

    def generate_path_girsanov(self):
        """Generate paths for the underlying asset under the risk-neutral measure using Girsanov theorem.

        This method implements the change of measure from the real-world measure (with drift mu)
        to the risk-neutral measure (with drift r) using Girsanov's theorem.

        Returns:
            np.ndarray: Matrix of simulated paths under the risk-neutral measure
        """
        dt = self.T / self.N
        t = np.linspace(0, self.T, self.N)

        theta = (self.mu - self.r) / self.sigma

        dW = ss.norm.rvs(scale=np.sqrt(dt), size=(self.N - 1, self.M))

        dW_tilde = dW - theta * np.sqrt(dt)

        W_tilde = np.cumsum(dW_tilde, axis=0)
        W_tilde = np.vstack([np.zeros(self.M), W_tilde])

        self.S = self.S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * t[:, None] + self.sigma * W_tilde
        )

        return self.S


# %%
# %%
if __name__ == "__main__":
    params = {
        "S0": 100,
        "strike_price": 120,
        "maturity": 1,
        "sigma": 0.2,
        "r": 0.05,
        "mu": 0.1,
        "dividend": 0.0,
        "N": 252,
        "M": 100,
    }
    option = EuropeanOptions(**params)

    # Plot delta hedging PnL
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    pnl = option.delta_hedging("call")
    plt.plot(np.mean(pnl, axis=1))
    plt.title("Delta Hedging PnL")
    plt.xlabel("Time Steps")
    plt.ylabel("PnL")
    plt.grid(True)

    # Plot simulated paths
    plt.subplot(1, 2, 2)
    simules = option.generate_path_girsanov()
    plt.plot(simules)
    plt.title("Simulated Paths under Risk-Neutral Measure")
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price")
    plt.grid(True)

    # Print option prices
    print(f"Call price (Black-Scholes): {option.price_call():.4f}")
    print(f"Put price (Black-Scholes): {option.price_put():.4f}")

    plt.tight_layout()
    plt.show()

# %%
