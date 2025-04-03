from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import kendalltau, multivariate_normal, multivariate_t, t


def array_to_symm_matrix(U: np.ndarray) -> np.ndarray:
    k = len(U)
    size = 0.5 * (1 + np.sqrt(1 + 8 * k))
    if not size.is_integer():
        raise ValueError("the size of the array is not of the form n(n-1)/2")
    V = np.identity(int(size))
    ui_1 = np.triu_indices(size, k=1)
    V[ui_1], V.T[ui_1] = U, U
    return V


def kendall_to_linear_correlation(x, y):
    "calculate the correlation from kendall's tau"
    tau, _ = kendalltau(x, y)
    return np.sin(tau * np.pi / 2)


def Normalizing(U: np.ndarray) -> np.ndarray:
    delta = np.diag(np.diag(U) ** 0.5)
    delta_inv = np.linalg.inv(delta)
    return delta_inv @ U @ delta_inv

def matrix_to_sdp(U: np.ndarray) -> np.ndarray:
    """Takes a matrix and returns its semi-definite positive (SDP) version."""
    eig_values, eig_vectors = np.linalg.eig(U)
    diag = np.where(eig_values < 0, 1e-8, eig_values)
    result = eig_vectors @ np.diag(diag) @ eig_vectors.T
    return result


def approxi_var_cov(U: np.ndarray) -> np.ndarray:
    samples = ndtr(U)
    return Normalizing(np.cov(samples.T))


def approxi_kendall_cov(U: np.ndarray) -> np.ndarray:
    _, n = U.shape
    ui = np.triu_indices(n, k=1)
    flat = [kendall_to_linear_correlation(U.T[i], U.T[j]) for i, j in zip(*ui)]
    sigma = matrix_to_sdp(array_to_symm_matrix(flat))
    return Normalizing(sigma)


def appro_mle_cov_df(df: float, sigma: np.ndarray, U: np.ndarray) -> float:
    quantiles = t.ppf(U, df=df)
    l_pdf = t.logpdf(quantiles, df=df).sum(axis=1)
    mvt_l_pdf = multivariate_t.logpdf(quantiles, shape=sigma, df=df)
    log_likelihood = mvt_l_pdf - l_pdf
    return -log_likelihood.sum()


@dataclass
class GaussianCopulaParametres:
    sigma: np.ndarray


@dataclass
class StudentCopulaParametres:
    sigma: np.ndarray
    df: float


class Copula:
    def __init__(self, parametres) -> None:
        """
        Initializes a Copula object with the specified parameters.
        Args:
            parametres: The parameters for the Copula.
        """
        self.parametres = parametres

    def fit(self):
        """
        Fits the Copula model.
        Returns:
            self: The fitted Copula object.
        """
        return self

    def rvs(self, shape):
        """
        Generates random variates from the Copula.
        Args:
            shape: The shape of the output random variates.
        Returns:
            self: The Copula object.
        """
        return self


class GaussianCopula(Copula):

    def __init__(self, parametres: GaussianCopulaParametres | None = None) -> None:
        super().__init__(parametres)

    def fit(self, samples: np.ndarray) -> Copula:
        sigma = approxi_var_cov(samples)
        self.parametres = GaussianCopulaParametres(sigma=sigma)
        return self

    def rvs(self, size: int | tuple[int]):
        sigma = self.parametres.sigma
        samples = multivariate_normal.rvs(cov=sigma, size=size)
        return ndtr(samples)


class StudentCopula(Copula):

    def __init__(self, parametres: StudentCopulaParametres | None = None) -> None:
        super().__init__(parametres)

    def fit(self, samples: np.ndarray) -> Copula:
        sigma = approxi_kendall_cov(samples)
        parameters = {
            "x0": 5,
            "args": (sigma, samples),
            "bounds": [(0.1, np.inf)],
            "method": "L-BFGS-B",
        }
        opt = minimize(appro_mle_cov_df, **parameters)
        self.parametres = StudentCopulaParametres(sigma=sigma, df=opt.x[0])
        return self

    def rvs(self, size: int | tuple[int]):
        sigma, df = self.parametres.sigma, self.parametres.df
        samples = multivariate_t.rvs(shape=sigma, df=df, size=size)
        return t.cdf(samples, df=df)


if __name__ == "__main__":
    cov = array_to_symm_matrix([0.1, 0.2, 0.13, 0.43, 0.9, 0.81])
    cov = matrix_to_sdp(cov)
    samples = multivariate_t.rvs(shape=cov, df=3, size=10000)
    X = t.cdf(samples, df=3)
    studentcopula = StudentCopula().fit(X)
    studentcopula.rvs(1000)
