from scipy.optimize import minimize, Bounds
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


bound = Bounds(1e-10, np.inf)


class EvarAlpha:
    def __init__(self, X, alpha):
        self.X = X
        self.alpha = alpha

    def moment(self, t):
        exp = np.exp(t * self.X)
        return np.mean(exp)

    def evar_main_ft(self, t):
        return np.power(t, -1) * (np.log(self.moment(t) / self.alpha))

    def evar(self):
        result = minimize(
            self.evar_main_ft, x0=0.5, bounds=bound
        )  # la fonction depend juste de t et pas de alpha
        return result.fun

    def short_fall(self):
        # il fallait calculer les var pour différente quantiles et prendre la moyenne de ces var
        pass


def x(alpha):
    return np.sqrt(-2 * np.log(alpha))


if __name__ == "__main__":
    M = 1000
    X = ss.norm.rvs(size = M)
    alphas = np.linspace(0, 1, 100)
    L = [x(alpha) for alpha in alphas]
    L_empiric = [EvarAlpha(X, alpha).evar() for alpha in alphas]
    plt.plot(alphas, L)
    plt.plot(alphas, L_empiric)
    plt.show()