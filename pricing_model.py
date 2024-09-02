import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


def geometric_brownian_motion(S0, mu, sigma, N, T, M):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    dB = ss.norm.rvs(scale=np.sqrt(dt), size=(N, M))
    dB = np.vstack((np.zeros(M), dB))
    B_t = np.cumsum(dB, axis=0)
    S_t = S0 * np.exp((mu - 0.5 * sigma**2) * t[:, None] + sigma * B_t)
    return t, S_t


def option_bounds_check(call_price, strike_price, r, maturity, S0):
    """
    Check if the call option price falls within the no-arbitrage bounds.
    """
    lower_bound = strike_price * np.exp(-r * maturity)
    upper_bound = S0 - lower_bound
    if upper_bound <= call_price <= S0:
        return True
    else:
        return False


class EuropeanOptionPricing:
    def __init__(
        self,
        S0,
        strike_price,
        maturity,
        sigma: float,
        r: np.ndarray,
        dividend: bool,
        ticket: str,
        N,
    ):
        self.S0 = S0
        self.strike_price = strike_price
        self.maturity = maturity
        self.sigma = sigma
        self.r = r
        self.dividend = dividend
        self.ticket = ticket
        self.N = N

    def calibrate(self):
        """
        Calibrate the model to get the average volatility and the asset's growth rate.
        This method would typically be implemented with historical data.
        """
        pass

    def predict_volatility(self):
        """
        Predict future volatility using a model like GARCH.
        This method would typically involve fitting a GARCH model to historical data.
        """
        pass

    def hitting_time(self, scenarios):
        """
        Determine the first time the stock hits the strike price and
        the last scenarios that hit the strike price.
        """
        pass

    def price_option_call(self, S):
        """Calculate the price of a call option using the Black-Scholes model"""
        sigma = self.sigma
        r = self.r
        T = self.maturity
        K = self.strike_price
        t = np.linspace(0, T, self.N)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
        return S * ss.norm.cdf(d1) - K * np.exp(-r * (T - t)) * ss.norm.cdf(d2)

    def price_option_put(self, S):
        """Calculate the price of a put option using the Black-Scholes model"""
        sigma = self.sigma
        r = self.r
        T = self.maturity
        K = self.strike_price
        t = np.linspace(0, T, self.N)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
        return K * np.exp(-r * (T - t)) * ss.norm.cdf(-d2) - S * ss.norm.cdf(-d1)

    def grecque_indices(self):
        """etude de l'impact des parametres sur le prix de l'option"""
        pass

if __name__ == "__main__":
    S0 = 100
    mu = 0.05
    sigma = 0.2
    N = 100
    T = 1
    M = 1000

    t, S_t = geometric_brownian_motion(S0, mu, sigma, N, T, M)

    plt.plot(t, S_t[:, 0])
    plt.title("Geometric Brownian Motion")
    plt.xlabel("Time (t)")
    plt.ylabel("Stock Price (S_t)")
    plt.show()

