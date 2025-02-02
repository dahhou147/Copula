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

class EuropeanOptionPricing:
    def __init__(
        self,
        S0: float,
        strike_price: float,
        maturity: float,
        sigma: float,
        r: float,
        dividend: bool,
        ticket: str,
        N: int,
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

    def pnl_portefeuille(self,option_type):
        delta = self.delta_hedging(option_type)
        if option_type == "call":
          option = self.price_option_call()
        else :
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
