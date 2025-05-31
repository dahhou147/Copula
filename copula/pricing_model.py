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
        S = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t[:, None] + self.sigma * W)
        return t, S


class BlackScholesPricer:
    """Class for Black-Scholes option pricing."""

    def __init__(self, S0: float, K: float, T: float, sigma: float, r: float, q: float = 0.0):
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
            return np.exp(-self.q * tau) * ss.norm.cdf(d1)
        if option_type == "put":
            return -np.exp(-self.q * tau) * ss.norm.cdf(-d1)
        raise ValueError("option_type must be 'call' or 'put'")

    def copy(self):
        return BlackScholesPricer(self.S0, self.K, self.T, self.sigma, self.r, self.q)


class Greeks:
    """Class for Greeks calculations."""

    def __init__(self, pricer: BlackScholesPricer):
        self.pricer = pricer.copy()

    def delta(self, S, tau, option_type="call"):
        """Calculate delta of the option."""
        d1, _ = self.pricer._d1_d2(S, tau)
        if option_type == "call":
            return np.exp(-self.pricer.q * tau) * ss.norm.cdf(d1)
        if option_type == "put":
            return -np.exp(-self.pricer.q * tau) * ss.norm.cdf(-d1)
        raise ValueError("option_type must be 'call' or 'put'")

    def gamma(self, S, tau):
        """Calculate gamma of the option."""
        d1, _ = self.pricer._d1_d2(S, tau)
        return (
            np.exp(-self.pricer.q * tau) * ss.norm.pdf(d1) / (S * self.pricer.sigma * np.sqrt(tau))
        )

    def vega(self, S, tau):
        """Calculate vega of the option."""
        d1, _ = self.pricer._d1_d2(S, tau)
        return S * np.exp(-self.pricer.q * tau) * ss.norm.pdf(d1) * np.sqrt(tau)

    def theta(self, S, tau, option_type="call"):
        """Calculate theta of the option."""
        d1, d2 = self.pricer._d1_d2(S, tau)
        if option_type == "call":
            return -S * self.pricer.sigma * np.exp(-self.pricer.q * tau) * ss.norm.pdf(d1) / (
                2 * np.sqrt(tau)
            ) - self.pricer.r * self.pricer.K * np.exp(-self.pricer.r * tau) * ss.norm.cdf(d2)
        if option_type == "put":
            return -S * self.pricer.sigma * np.exp(-self.pricer.q * tau) * ss.norm.pdf(d1) / (
                2 * np.sqrt(tau)
            ) + self.pricer.r * self.pricer.K * np.exp(-self.pricer.r * tau) * ss.norm.cdf(-d2)
        raise ValueError("option_type must be 'call' or 'put'")


class VolatilitySmile:
    """Class for volatility smile calculations."""

    def __init__(self, pricer: BlackScholesPricer):
        self.pricer = pricer

    def implied_volatility(self, strike: float, market_price: float):
        """Calculate implied volatility using Newton-Raphson method."""

        def f(sigma):
            temp_pricer = self.pricer.copy()  # éviter les effets de bord
            temp_pricer.sigma = sigma
            temp_pricer.K = strike
            return temp_pricer.price_call() - market_price

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



class ConstructPortfolio:
    """Class for constructing a self-financing hedging portfolio."""

    def __init__(self, pricer: BlackScholesPricer, paths: np.ndarray, N: int, T: float, K1: float, K2: float):
        self.pricer = pricer
        self.paths = paths
        self.N = N
        self.T = T
        self.K1 = K1
        self.K2 = K2
        self.dt = T / N
        self.M = paths.shape[1]

        self.greeks = Greeks(pricer)
        self.pnl = np.zeros((N, self.M))
        self.portfolio_values = np.zeros((N, self.M))
        self.cash_positions = np.zeros((N, self.M))
        self.old_coeffs = np.zeros((N, self.M,3))

    def get_coefficients(self, S, tau, option_type="call"):
        """Solve for portfolio weights: alpha1 * option1 + alpha2 * option2 + alpha3 * stock = target greeks"""
        delta0 = self.greeks.delta(S, tau, option_type)
        gamma0 = self.greeks.gamma(S, tau)
        vega0 = self.greeks.vega(S, tau)

        # Setup hedge option pricers
        pricer1 = self.pricer.copy()
        pricer1.K = self.K1
        greeks1 = Greeks(pricer1)
        delta1 = greeks1.delta(S, tau, option_type)
        gamma1 = greeks1.gamma(S, tau)
        vega1 = greeks1.vega(S, tau)

        pricer2 = self.pricer.copy()
        pricer2.K = self.K2
        greeks2 = Greeks(pricer2)
        delta2 = greeks2.delta(S, tau, option_type)
        gamma2 = greeks2.gamma(S, tau)
        vega2 = greeks2.vega(S, tau)

        # Check if the matrix is well-conditioned
        matrix = np.array([
            [delta1, delta2, 1],
            [gamma1, gamma2, 0],
            [vega1, vega2, 0],
        ])
        b = np.array([delta0, gamma0, vega0])

        cond = np.linalg.cond(matrix)
        
        if cond > 1e10:
            lambda_reg = 1e-6
            matrix_reg = matrix.T @ matrix + lambda_reg * np.eye(3)
            coeffs = np.linalg.solve(matrix_reg, matrix.T @ b)
        else:
            try:
                coeffs = np.linalg.solve(matrix, b)
            except np.linalg.LinAlgError:
                return None, None, None

        return coeffs, pricer1, pricer2

    def hedge_portfolio(self, option_type="call"):
        """Perform dynamic hedging of the portfolio"""
        for i in range(self.M):
            S = self.paths[0, i]
            tau = self.T
            self.pricer.S0 = S

            coeffs, pricer1, pricer2 = self.get_coefficients(S, tau, option_type)
            if coeffs is None:
                continue

            val0 = self.pricer.price_call() if option_type == "call" else self.pricer.price_put()
            val1 = pricer1.price_call() if option_type == "call" else pricer1.price_put()
            val2 = pricer2.price_call() if option_type == "call" else pricer2.price_put()

            # Calculate initial cash position
            cash = val0 - (coeffs[0] * val1 + coeffs[1] * val2 + coeffs[2] * S)
            
            # Store initial positions
            self.cash_positions[0, i] = cash
            self.portfolio_values[0, i] = coeffs[0] * val1 + coeffs[1] * val2 + coeffs[2] * S + cash
            self.pnl[0, i] = self.portfolio_values[0, i] - val0
            self.old_coeffs[0, i, :] = coeffs

            old_coeffs = coeffs

            for j in range(1, self.N):
                S = self.paths[j, i]
                tau = self.T - j * self.dt
                self.pricer.S0 = S

                coeffs, pricer1, pricer2 = self.get_coefficients(S, tau, option_type)
                if coeffs is None:
                    # If we can't get new coefficients, keep the old ones
                    coeffs = old_coeffs

                val0 = self.pricer.price_call() if option_type == "call" else self.pricer.price_put()
                val1 = pricer1.price_call() if option_type == "call" else pricer1.price_put()
                val2 = pricer2.price_call() if option_type == "call" else pricer2.price_put()

                # Update cash position
                cash = self.cash_positions[j - 1, i] * np.exp(self.pricer.r * self.dt)

                # Adjust portfolio
                d_alpha = coeffs - old_coeffs
                cash -= d_alpha[0] * val1 + d_alpha[1] * val2 + d_alpha[2] * S

                self.cash_positions[j, i] = cash
                self.portfolio_values[j, i] = coeffs[0] * val1 + coeffs[1] * val2 + coeffs[2] * S + cash
                self.pnl[j, i] = self.portfolio_values[j, i] - val0
                self.old_coeffs[j, i, :] = coeffs

                old_coeffs = coeffs


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

def example_usage():

    
    # # Plot results
    # plt.figure(figsize=(15, 10))
    
    # # Plot 1: PnL distribution
    # plt.subplot(2, 2, 1)
    # plt.hist(final_pnl, bins=50, density=True)
    # plt.title('Final PnL Distribution')
    # plt.xlabel('PnL')
    # plt.ylabel('Density')
    
    # # Plot 2: Portfolio value evolution
    # plt.subplot(2, 2, 2)
    # mean_portfolio = np.mean(portfolio.portfolio_values, axis=1)
    # std_portfolio = np.std(portfolio.portfolio_values, axis=1)
    # plt.plot(mean_portfolio, label='Mean Portfolio Value')
    # plt.fill_between(mean_portfolio - std_portfolio,
    #                 mean_portfolio + std_portfolio,
    #                 alpha=0.2)
    # plt.title('Portfolio Value Evolution')
    # plt.xlabel('Time')
    # plt.ylabel('Portfolio Value')
    
    # # Plot 3: Cash position evolution
    # plt.subplot(2, 2, 3)
    # mean_cash = np.mean(portfolio.cash_positions, axis=1)
    # plt.plot(mean_cash)
    # plt.title('Cash Position Evolution')
    # plt.xlabel('Time')
    # plt.ylabel('Cash Position')
    
    # # Plot 4: PnL evolution
    # plt.subplot(2, 2, 4)
    # mean_pnl = np.mean(portfolio.pnl, axis=1)
    # plt.plot(mean_pnl)
    # plt.title('PnL Evolution')
    # plt.xlabel('Time')
    # plt.ylabel('PnL')
    
    # plt.tight_layout()
    
    # # Print performance metrics
    # print("\nPortfolio Performance Metrics:")
    # print(f"Mean Final PnL: {mean_pnl}")
    # print(f"Std Final PnL: {std_pnl}")
    # print(f"Sharpe Ratio: {sharpe_ratio}")
    # print(f"Min PnL: {np.min(final_pnl)}")
    # print(f"Max PnL: {np.max(final_pnl)}")
    
    
    return plt.gcf()
#%%
if __name__ == "__main__":
    """Example of portfolio construction and hedging with performance analysis."""
    # Market parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price of the option to hedge
    T = 1.0     # Time to maturity (1 year)
    r = 0.05    # Risk-free rate
    q = 0.0     # Dividend yield
    sigma = 0.2 # Volatility
    N = 252     # Number of time steps (daily rebalancing)
    M = 100    # Number of simulations
    mu = 0.1    # Drift
    # Create price paths
    gbm = GirsanovSimulator(S0,mu, r, sigma, N, T, M)
    paths = gbm.generate_paths()
    
    # Create pricer for the option to hedge
    pricer = BlackScholesPricer(S0, K, T, sigma, r, q)
    
    # Create portfolio with two hedging options
    # K1 and K2 are the strikes of the hedging options
    K1 = K * 0.9  # 10% OTM put
    K2 = K * 1.1  # 10% OTM call
    
    portfolio = ConstructPortfolio(pricer, paths, N, T, K1, K2)
    
    # Hedge portfolio
    portfolio.hedge_portfolio(option_type="call")
    
    # Calculate performance metrics
    final_pnl = portfolio.pnl[-1, :]
    mean_pnl = np.mean(final_pnl)
    std_pnl = np.std(final_pnl)
    sharpe_ratio = np.sqrt(252) * mean_pnl / std_pnl if std_pnl != 0 else 0
# %%
