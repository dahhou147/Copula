#%%
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from dataclasses import dataclass


def calibrate_heston(S, V, dt):
    returns = np.log(S[1:] / S[:-1])
    
    mean_return = np.mean(returns) / dt
    var_return = np.var(returns) / dt
    
    # Estimate kappa (mean reversion speed)
    kappa = -np.corrcoef(V[1:], V[:-1])[0,1] / dt
    
    # Estimate theta (long-run variance)
    theta = np.mean(V)
    
    # Estimate vol_vol (volatility of variance)
    vol_vol = np.std(np.diff(V)) / np.sqrt(dt)
    
    # Estimate correlation
    rho = np.corrcoef(returns, np.diff(V))[0,1]
    
    return {
        'kappa': kappa,
        'theta': theta, 
        'vol_vol': vol_vol,
        'rho': rho,
        'mu': mean_return
    }
@dataclass
class parametre:
    mu :float
    vol_vol : float
    theta : float
    theta :float
    rho : float
    k = float

class HestonModel:
    def __init__(self, mu=0.6, vol_vol=0.2, k=0.6, theta=0.3, N=100, M=100, V_0=0.2, S_0=100, rho=0.2, T=1):
        self.mu = mu
        self.vol_vol = vol_vol
        self.k = k
        self.theta = theta
        self.N = N
        self.M = M
        self.V_0 = V_0
        self.S_0 = S_0
        self.rho = rho
        self.T = T
        self.dt = T / N

    def heston_MC(self):
        dw = ss.multivariate_normal.rvs(mean=[0, 0], cov=[[self.dt, self.rho * self.dt], [self.rho * self.dt, self.dt]], size=(self.N-1, self.M))
        dw = np.vstack((np.zeros((1, self.M, 2)), dw))
        dw_0 = dw[:, :, 0]
        dw_1 = dw[:, :, 1]

        V_t = np.zeros((self.N, self.M))
        V_t[0] = self.V_0
        S_t = np.zeros((self.N, self.M))
        S_t[0] = self.S_0

        for i in range(1, self.N):
            V_t[i] = V_t[i-1] + self.k * (self.theta - V_t[i-1]) * self.dt + self.vol_vol * np.sqrt(V_t[i-1]) * dw_1[i]
            S_t[i] = S_t[i-1] * np.exp((self.mu - 0.5 * V_t[i-1]) * self.dt + np.sqrt(V_t[i-1]) * dw_0[i])

        V_t = np.maximum(V_t, 0)

        return V_t, S_t

    def calibration(self):
        pass


    def call_option(self):

        pass
#%%
if __name__=="__main__":
    heston_model = HestonModel()
    V_t, S_t = heston_model.generate_paths()
    plt.plot(S_t)
    plt.show()
