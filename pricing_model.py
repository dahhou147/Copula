#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

def geometric_brownian_motion(S0, mu, sigma, N, T, M):
    dt = T / N
    t = np.linspace(0, T, N)
    dB = ss.norm.rvs(scale=np.sqrt(dt), size=(N-1, M))
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
    if lower_bound <= call_price <= upper_bound:
        return True
    else:
        return False
    
def portefeuille(S0, mu, sigma, N, T, M, rf):
    """
    Calculate the value of a portfolio consisting of a risk-free asset and a risky asset,
    adjusting the proportion dynamically based on their values.
    
    Parameters:
    S0 (float): Initial price of the risky asset.
    mu (float): Expected return of the risky asset.
    sigma (float): Volatility of the risky asset.
    N (int): Number of time steps.
    T (float): Total time period.
    M (int): Number of simulation paths.
    rf (float): Risk-free interest rate.
    
    Returns:
    np.ndarray: Portfolio values over time.
    """
    t, S = geometric_brownian_motion(S0, mu, sigma, N, T, M)
    portfolio_value = np.zeros_like(S)
    x = np.zeros_like(S)
    
    for i in range(N):
        risk_free_asset = np.exp(rf * t[i])
        risky_asset = S[i] / S0
        x[i] = (risky_asset > risk_free_asset).astype(float)
        portfolio_value[i] = x[i] * risk_free_asset + (1 - x[i]) * risky_asset
    
    return t, portfolio_value



class EuropeanOptionPricing:
    def __init__(
        self,  
        S0: float,  # prix de l'actif sous-jacent
        strike_price: float,    # prix d'exercice
        maturity: float,    # maturity
        sigma: float, # volatility
        r: float,  # risk-free interest rate
        dividend: bool, # dividend yield
        ticket: str,   
        N: int, # number of time steps
    ):
        self.S0 = S0
        self.strike_price = strike_price
        self.maturity = maturity
        self.sigma = sigma
        self.r = r  # taux d'intérêt sans risque
        self.dividend = dividend
        self.ticket = ticket
        self.N = N
        self.M = 500
        self.S = None
        self.portefeuille = 0

    def calibrate(self, historical_prices):
        """
        Calibrate the model to get the average volatility and the asset's growth rate.
        This method uses historical price data.
        """
        log_returns = np.diff(np.log(historical_prices))
        self.sigma = np.std(log_returns) * np.sqrt(self.N)
        self.mu = np.mean(log_returns) * self.N

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

    def GBM(self):
        t, self.S = geometric_brownian_motion(self.S0, self.r, self.sigma, self.N, self.maturity, self.M)
        return self.S

    def price_option_call(self):
        """Calculate the price of a call option using the Black-Scholes model"""
        self.GBM()
        S = self.S
        sigma = self.sigma
        r = self.r
        T = self.maturity
        K = self.strike_price
        t = np.linspace(0, T, self.N)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t[:, None])) / (sigma * np.sqrt(T - t[:, None]))
        d2 = d1 - sigma * np.sqrt(T - t[:, None])
        call_prices = S * ss.norm.cdf(d1) - K * np.exp(-r * (T - t[:, None])) * ss.norm.cdf(d2)
        return call_prices

    def delta_hedging(self, option_type: str):
        self.GBM()
        S = self.S
        sigma = self.sigma
        r = self.r
        T = self.maturity
        K = self.strike_price
        t = np.linspace(0, T, self.N)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t[:, None])) / (sigma * np.sqrt(T - t[:, None]))

        if option_type == "call":
            return ss.norm.cdf(d1)
        elif option_type == "put":
            return ss.norm.cdf(d1) - 1
        else:
            print("Le delta est calculé pour juste les options call et put")
            return None

    def pnl_portefeuille(self, option_type):
        delta = self.delta_hedging(option_type)
        if option_type == "call":
            option = self.price_option_call()
        else:
            option = self.price_option_put()
        cash = self.portefeuille + option[0] - delta[0] * self.S[0]
        portefeuille_final = delta[-1] * self.S[-1] + cash * np.exp(self.r * self.maturity) - option[-1]
        return portefeuille_final - self.portefeuille

    def price_option_put(self):
        """Calculate the price of a put option using the Black-Scholes model"""
        self.GBM()
        S = self.S
        sigma = self.sigma
        r = self.r
        T = self.maturity
        K = self.strike_price
        t = np.linspace(0, T, self.N)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t[:, None])) / (sigma * np.sqrt(T - t[:, None]))
        d2 = d1 - sigma * np.sqrt(T - t[:, None])
        put_prices = K * np.exp(-r * (T - t[:, None])) * ss.norm.cdf(-d2) - S * ss.norm.cdf(-d1)
        return put_prices

    def grecque_indices(self):
        """Les indices grecs : vega, theta, lambda, ect."""
        pass
    def plot_simulation(self):
        """Plot the simulated paths of the geometric Brownian motion."""
        if self.S is None:
            self.GBM()
        plt.figure(figsize=(10, 6))
        plt.plot(self.S)
        plt.title('Simulated Geometric Brownian Motion Paths')
        plt.xlabel('Time Steps')
        plt.ylabel('Asset Price')
        plt.show()

    def plot_option_prices(self):
        """Plot the call and put option prices over time."""
        call_prices = self.price_option_call()
        put_prices = self.price_option_put()
        plt.figure(figsize=(10, 6))
        plt.plot(call_prices, label='Call Option Prices')
        plt.plot(put_prices, label='Put Option Prices')
        plt.title('Option Prices Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Option Price')
        plt.legend()
        plt.show()

# %%
