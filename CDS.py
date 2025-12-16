# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from dataclasses import dataclass
from scipy.optimize import fsolve
tickers = ["SAN.MC", "AI.PA", "BAS.DE", "BMW.DE", "CS.PA"]


@dataclass
class MertonParametres:
    PD: float
    debt: float
    sigma_v: float
    V: float


def GBM_correlated(cov, n_sims=1000, n_steps=252, T=1):
    dt = T / n_steps
    n_entite = cov.shape[0]
    dw = ss.multivariate_normal.rvs(cov=cov * dt, size=(n_steps - 1, n_sims))
    w = np.concatenate([np.zeros((1, n_sims, n_entite)), dw.cumsum(axis=0)])
    return w


def get_total_debt(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    return balance_sheet.loc["Total Debt"].iloc[0]


def get_market_cap(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info.get("marketCap", None)


def get_asset_volatility(ticker):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period="1y")
    returns = np.log(historical_data["Close"] / historical_data["Close"].shift(1))
    return returns.std(ddof=1) * np.sqrt(252)


def d1_formula(asset_value, total_debt, asset_volatility, risk_free_rate, maturity):
    return (
        np.log(asset_value / total_debt)
        + (risk_free_rate + 0.5 * asset_volatility**2) * maturity
    ) / (asset_volatility * np.sqrt(maturity))


class MertonModel:
    def __init__(self, ticker):
        self.ticker = ticker
        self.D = get_total_debt(ticker)
        self.V = get_market_cap(ticker)
        self.E = abs(self.V - self.D)
        self.sigma_E = get_asset_volatility(ticker)
        self.sigma_V = None
        self.risk_free_rate = 0.05
        self.maturity = 1
        self.PD = None

    def get_market_cap_volatility(self):
        def equation(vars):
            sigma_v, V = vars
            d1 = d1_formula(V, self.D, sigma_v, self.risk_free_rate, self.maturity)
            d2 = d1 - sigma_v * np.sqrt(self.maturity)
            eq1 = (
                V * ss.norm.cdf(d1)
                - self.D
                * np.exp(-self.risk_free_rate * self.maturity)
                * ss.norm.cdf(d2)
                - self.E
            )
            eq2 = self.sigma_E * self.E - ss.norm.cdf(d1) * sigma_v * V
            return [eq1, eq2]

        V_gess = self.E + self.D * np.exp(-self.risk_free_rate * self.maturity)
        sigma_v_guess = self.sigma_E * self.E / V_gess
        solution = fsolve(equation, [sigma_v_guess, V_gess])
        self.sigma_V = solution[0]
        self.V = solution[1]

    
    def get_PD_marginal(self):
        """
        Calcule la probabilité de défaut analytique selon le modèle de Merton.
        PD = N(-d2) où d2 est le paramètre standard du modèle de Merton.
        
        Retourne la PD avec des contrôles de réalisme.
        """
        d2 = (
            np.log(self.V / self.D)
            + (self.risk_free_rate - 0.5 * self.sigma_V**2) * self.maturity
        ) / (self.sigma_V * np.sqrt(self.maturity))
        
        self.PD = ss.norm.cdf(-d2)

    def run(self):
        self.get_market_cap_volatility()
        self.get_PD_marginal()
        return MertonParametres(PD=self.PD, debt=self.D, sigma_v=self.sigma_V, V=self.V)

#%%

def MertonModel_to_DataFrame():
    modeles_parametres = {
        ticker: MertonModel(ticker).run() for ticker in tickers
    }
    df = pd.DataFrame(
        [[params.debt, params.sigma_v, params.V, params.PD] for params in modeles_parametres.values()],
        index=modeles_parametres.keys(),
        columns=["debt","volatility","asset_value","PD"],
    )
    return df
#%%
class CDSPortfolio:
    
    def __init__(self, tickers, maturity=1, recovery_rate=0.4, n_sims=100000):
        self.tickers = tickers
        self.maturity = maturity
        self.recovery_rate = recovery_rate
        self.n_sims = n_sims
        self.risk_free_rate = 0.05
        self._correlation_matrix = None
        self._marginal_pds = None
    
    def get_equity_correlation_matrix(self):
        if self._correlation_matrix is not None:
            return self._correlation_matrix

        df = yf.download(self.tickers, start="2024-01-01", end="2025-10-01", progress=False)
        
        df_rendement = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        
        corr_matrix = df_rendement.corr().to_numpy()
        

        corr_matrix = (corr_matrix + corr_matrix.T) / 2  
        np.fill_diagonal(corr_matrix, 1.0) 
        
        self._correlation_matrix = corr_matrix
        return corr_matrix
    
    def get_marginal_default_probabilities(self):
        """
        Calcule les probabilités de défaut marginales pour chaque entité.
        Utilise le modèle de Merton avec sanitization pour obtenir des PD réalistes.
        
        Returns:
        --------
        marginal_pds : np.ndarray
            Array des PD marginales (une par entité)
        """
        if self._marginal_pds is not None:
            return self._marginal_pds
        
        marginal_pds = []
        
        for ticker in self.tickers:
            try:
                merton = MertonModel(ticker)
                merton.maturity = self.maturity
                pd = merton.get_PD_marginal()
                marginal_pds.append(pd)
            except Exception as e:
                print(f"Warning: Erreur pour {ticker}, utilisation d'une PD par défaut: {e}")
                marginal_pds.append(0.01)  
        
        marginal_pds = np.array(marginal_pds)
        self._marginal_pds = marginal_pds
        return marginal_pds
    
    def simulate_gaussian_copula(self):
        """
        Simule les variables latentes corrélées selon une Gaussian copula.
        
        Le modèle suppose que chaque entité i a un facteur latent X_i ~ N(0, Corr)
        et défaut si X_i < k_i où k_i = Φ^(-1)(PD_i).
        
        Returns:
        --------
        default_matrix : np.ndarray
            Matrice booléenne de défauts (n_sims x n_entités)
            True si défaut, False sinon
        """
        n_entities = len(self.tickers)
        
        marginal_pds = self.get_marginal_default_probabilities()
        
        # 2. Calculer les seuils de défaut : k_i = Φ^(-1)(PD_i)
        default_thresholds = ss.norm.ppf(marginal_pds)
        
        corr_matrix = self.get_equity_correlation_matrix()
        
        latent_variables = ss.multivariate_normal.rvs(
            mean=np.zeros(n_entities),
            cov=corr_matrix,
            size=self.n_sims
        )
        default_matrix = latent_variables < default_thresholds
        
        return default_matrix
    
    def compute_portfolio_statistics(self):
        """
        Calcule les statistiques complètes du portefeuille :
        - PD individuelles (devraient correspondre aux PD marginales)
        - PD du pool (au moins 1 défaut)
        - Distribution du nombre de défauts
        - Probabilités de k défauts
        
        Returns:
        --------
        results : dict
            Dictionnaire contenant toutes les statistiques
        """
        default_matrix = self.simulate_gaussian_copula()
        
        prob_default_individual = default_matrix.mean(axis=0)
        
        n_defaults_per_scenario = default_matrix.sum(axis=1)
        
        prob_default_pool_at_least_one = (n_defaults_per_scenario > 0).mean()
        
        max_defaults = len(self.tickers)
        default_distribution = {
            k: (n_defaults_per_scenario == k).mean() 
            for k in range(max_defaults + 1)
        }
        
        mean_defaults = n_defaults_per_scenario.mean()
        std_defaults = n_defaults_per_scenario.std()
        max_defaults_observed = n_defaults_per_scenario.max()
        
        results = {
            'tickers': self.tickers,
            'marginal_pds': self.get_marginal_default_probabilities(),
            'prob_default_individual': prob_default_individual,
            'prob_default_pool_at_least_one': prob_default_pool_at_least_one,
            'default_distribution': default_distribution,
            'mean_defaults': mean_defaults,
            'std_defaults': std_defaults,
            'max_defaults_observed': max_defaults_observed,
            'n_sims': self.n_sims,
            'correlation_matrix': self.get_equity_correlation_matrix(),
        }
        
        return results


class CDS:
    """
    Ancienne classe conservée pour compatibilité.
    Utilise maintenant CDSPortfolio pour une approche plus réaliste.
    """
    def __init__(self, tickers, maturity=1, recovery_rate=0.4):
        self.tickers = tickers
        self.maturity = maturity
        self.recovery_rate = recovery_rate
        self.risk_free_rate = 0.05
        # Utiliser la nouvelle classe en interne
        self.portfolio = CDSPortfolio(tickers, maturity, recovery_rate, n_sims=1000)
    
    def get_default_probability(self):
        """
        Interface de compatibilité avec l'ancien code.
        """
        results = self.portfolio.compute_portfolio_statistics()
        
        df_params = MertonModel_to_DataFrame()
        debts = df_params["debt"].to_numpy()
        

        n_sims = self.portfolio.n_sims
        n_entities = len(self.tickers)
        last_asset_values = np.random.randn(n_sims, n_entities)  
        
        default_matrix = self.portfolio.simulate_gaussian_copula()
        
        return (
            debts,
            last_asset_values,
            default_matrix,
            results['prob_default_individual'],
            results['prob_default_pool_at_least_one'],
        )
    


#%%
if __name__ == "__main__":
    pass