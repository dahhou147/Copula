import numpy as np
import scipy.stats as ss
import scipy.optimize as so

def geometric_brownian_motion(S0, mu, sigma, N, T, M):
    """Generate geometric Brownian motion paths"""
    dt = T / N
    t = np.linspace(0, T, N)
    dW = ss.norm.rvs(scale=np.sqrt(dt), size=(N-1, M))
    W = np.cumsum(dW, axis=0)
    W = np.vstack([np.zeros(M), W])
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t[:, None] + sigma * W)
    return t, S

class EuropeanOptionPricing:
    def __init__(
        self,  
        S0: float,
        strike_price: float,
        maturity: float,
        sigma: float,
        r: float,
        dividend: float = 0.0,  # Changed to continuous dividend yield
        N: int = 252,  # Default to 1 year with daily steps
        M: int = 1000  # Number of simulations
    ):
        self.S0 = S0
        self.K = strike_price
        self.T = maturity
        self.sigma = sigma
        self.r = r
        self.q = dividend  
        self.N = N
        self.M = M
        self.S = None  
        
    def _d1_d2(self, S, tau):
        """Compute d1 and d2 for Black-Scholes formula"""
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)
        return d1, d2

    def price_call(self):
        """Black-Scholes call price"""
        d1, d2 = self._d1_d2(self.S0, self.T)
        return self.S0 * np.exp(-self.q*self.T) * ss.norm.cdf(d1) - self.K * np.exp(-self.r*self.T) * ss.norm.cdf(d2)

    def price_put(self):
        """Black-Scholes put price"""
        d1, d2 = self._d1_d2(self.S0, self.T)
        return self.K * np.exp(-self.r*self.T) * ss.norm.cdf(-d2) - self.S0 * np.exp(-self.q*self.T) * ss.norm.cdf(-d1)

    def delta(self, option_type: str):
        """Option delta at time t=0"""
        d1, _ = self._d1_d2(self.S0, self.T)
        if option_type == "call":
            return np.exp(-self.q*self.T) * ss.norm.cdf(d1)
        elif option_type == "put":
            return np.exp(-self.q*self.T) * (ss.norm.cdf(d1) - 1)
        raise ValueError("Invalid option type")

    def implied_volatility(self, market_price: float, option_type: str = 'call'):
        """Calculate implied volatility using Newton-Raphson method"""
        def f(sigma):
            self.sigma = sigma
            return (self.price_call() if option_type == 'call' else self.price_put()) - market_price
        
        try:
            return so.newton(f, 0.2, maxiter=50)  # 0.2 as initial guess
        except RuntimeError:
            return np.nan

    def volatility_smile(self, strikes: np.ndarray, market_prices: np.ndarray, option_type: str = 'call'):
        """Calculate volatility smile for given strikes and market prices"""
        return [self.implied_volatility(price, option_type) for price in market_prices]

    def delta_hedge_pnl(self, option_type: str, n_simulations: int = 100):
        """Simulate delta hedging P&L for multiple paths"""
        if self.S is None:
            self._simulate_paths()
            
        dt = self.T / self.N
        pnls = []
        
        for path in range(n_simulations):
            S_path = self.S[:, path]
            cash = self.price_call() if option_type == 'call' else self.price_put()
            shares = 0.0
            
            for i in range(self.N-1):
                tau = self.T - i*dt
                S = S_path[i]
                
                d1, _ = self._d1_d2(S, tau)
                delta = ss.norm.cdf(d1) if option_type == 'call' else ss.norm.cdf(d1) - 1
                
                shares_new = delta
                cash -= (shares_new - shares) * S
                shares = shares_new
                
                cash *= np.exp(self.r * dt)
                
            final_payoff = max(S_path[-1] - self.K, 0) if option_type == 'call' else max(self.K - S_path[-1], 0)
            pnl = cash + shares * S_path[-1] - final_payoff
            pnls.append(pnl)
            
        return np.array(pnls)

    def _simulate_paths(self):
        """Generate GBM paths"""
        _, self.S = geometric_brownian_motion(
            S0=self.S0,
            mu=self.r - self.q,  
            sigma=self.sigma,
            N=self.N,
            T=self.T,
            M=self.M
        )

if __name__ == "__main__":
    params = {
        'S0': 100,
        'strike_price': 120,
        'maturity': 1,
        'sigma': 0.2,
        'r': 0.05,
        'dividend': 0.0,
        'N': 252,
        'M': 1000
    }
    
    option = EuropeanOptionPricing(**params)
    
    print(f"Call price: {option.price_call():.2f}")
    
    print(f"Call delta: {option.delta('call'):.2f}")
    market_price = 10.50
    iv = option.implied_volatility(market_price, 'call')
    print(f"Implied volatility: {iv:.2%}")
    strikes = np.array([90, 100, 110])
    market_prices = np.array([15.0, 10.5, 7.5])
    smile = option.volatility_smile(strikes, market_prices)
    print("Volatility smile:", np.round(smile, 4))
    pnls = option.delta_hedge_pnl('call')
    print(f"Delta hedging P&L stats: Mean={pnls.mean():.2f}, Std={pnls.std():.2f}")