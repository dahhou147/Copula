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
        S0=100,
        strike_price =100,
        maturity=1,
        sigma = 0.2,
        r = 0.05,
        dividend = False,
        ticket = "",
        paths_nb =100,
    ):
        self.S0 = S0
        self.strike_price = strike_price
        self.maturity = maturity
        self.sigma = sigma
        self.r = r
        self.dividend = dividend
        self.ticket = ticket
        self.paths_nb =paths_nb
        
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

    def price_option_call(self,S0):
        """Calculate the price of a call option using the Black-Scholes model"""
        S =self.S0
        sigma = self.sigma
        r = self.r
        T = self.maturity
        K = self.strike_price
        N = self.paths_nb
        t = np.linspace(0,T,N)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
        d2 = d1 - sigma * np.sqrt(T-t)
        return S * ss.norm.cdf(d1) - K * np.exp(-r * (T-t)) * ss.norm.cdf(d2)
    

    def path_of_call(self):
        t,S_t  = self.geometric_brownian_motion()
        return [self.price_option_call(S) for S in S_t.T]
            

    def price_option_put(self,S0):
        """Calculate the price of a put option using the Black-Scholes model"""
        sigma = self.sigma
        r = self.r
        T = self.maturity
        K = self.strike_price
        t = np.linspace(0, T, self.N)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
        return K * np.exp(-r * (T - t)) * ss.norm.cdf(-d2) - S0 * ss.norm.cdf(-d1)

    def grecque_indices(self):
        """etude de l'impact des parametres sur le prix de l'option"""
        pass
if __name__ == "__main__":
    import yfinance as yf
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as ss
    from sklearn.model_selection import train_test_split
    ticker ="AAPL"
    df = yf.download(ticker,period="1mo")
    data = df["Close"].to_numpy()
    std = data.std(ddof=1)
    mu = np.log(data[-1]/data[0])/len(data)
    euro  = EuropeanOptionPricing(246.78,222.5,1,std,mu,False,"",100)
    call = euro.price_option_call(247.06)
    plt.plot(call)
    print(call[-1])
 