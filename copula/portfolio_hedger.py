#%%
import numpy as np
from typing import List, Dict
from pricing_model import BlackScholesPricer, Greeks

class OptionPosition:
    """Classe pour représenter une position d'option."""
    
    def __init__(self, pricer: BlackScholesPricer, quantity: int, option_type: str):
        self.pricer = pricer
        self.quantity = quantity
        self.option_type = option_type
        self.greeks = Greeks(pricer)
    
    def get_delta(self, S: float, tau: float) -> float:
        """Retourne le delta de la position."""
        return self.quantity * self.greeks.delta(S, tau, self.option_type)
    
    def get_gamma(self, S: float, tau: float) -> float:
        """Retourne le gamma de la position."""
        return self.quantity * self.greeks.gamma(S, tau)
    
    def get_vega(self, S: float, tau: float) -> float:
        """Retourne le vega de la position."""
        return self.quantity * self.greeks.vega(S, tau)
    
    def get_price(self) -> float:
        """Retourne le prix de la position."""
        if self.option_type == "call":
            return self.quantity * self.pricer.price_call()
        return self.quantity * self.pricer.price_put()


class HedgedPortfolio:
    """Classe pour gérer un portefeuille d'options delta-gamma-vega neutre."""
    
    def __init__(self, S0: float, r: float, q: float = 0.0):
        self.S0 = S0
        self.r = r
        self.q = q
        self.positions: List[OptionPosition] = []
        self.underlying_position = 0.0  # Position sur l'actif sous-jacent
    
    def add_position(self, position: OptionPosition):
        """Ajoute une position d'option au portefeuille."""
        self.positions.append(position)
    
    def get_portfolio_greeks(self, S: float, tau: float) -> Dict[str, float]:
        """Calcule les grecques totales du portefeuille."""
        total_delta = self.underlying_position  # Delta de la position sous-jacente
        total_gamma = 0.0
        total_vega = 0.0
        
        for position in self.positions:
            total_delta += position.get_delta(S, tau)
            total_gamma += position.get_gamma(S, tau)
            total_vega += position.get_vega(S, tau)
        
        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "vega": total_vega
        }
    
    def hedge_portfolio(self, target_delta: float = 0.0, target_gamma: float = 0.0, target_vega: float = 0.0):
        """Hedge le portefeuille pour atteindre les objectifs delta, gamma et vega."""
        current_greeks = self.get_portfolio_greeks(self.S0, self.positions[0].pricer.T)
        
        # Ajuster la position sous-jacente pour le delta
        delta_adjustment = target_delta - current_greeks["delta"]
        self.underlying_position += delta_adjustment
        
        # Si on a besoin d'ajuster gamma et vega, on peut ajouter des positions d'options
        if abs(current_greeks["gamma"] - target_gamma) > 1e-6 or abs(current_greeks["vega"] - target_vega) > 1e-6:
            # Créer une position d'option pour ajuster gamma et vega
            # Note: Dans la pratique, il faudrait optimiser les paramètres de cette option
            hedge_option = BlackScholesPricer(
                self.S0,
                self.S0 * 1.1,  # Strike à 10% OTM
                self.positions[0].pricer.T,
                0.3,  # Volatilité
                self.r,
                self.q
            )
            
            # Calculer la quantité nécessaire pour atteindre les objectifs
            gamma_needed = target_gamma - current_greeks["gamma"]
            vega_needed = target_vega - current_greeks["vega"]
            
            # Créer une position d'option de couverture
            hedge_position = OptionPosition(hedge_option, 1, "call")
            hedge_gamma = hedge_position.get_gamma(self.S0, hedge_option.T)
            hedge_vega = hedge_position.get_vega(self.S0, hedge_option.T)
            
            # Calculer la quantité optimale
            quantity = max(
                abs(gamma_needed / hedge_gamma),
                abs(vega_needed / hedge_vega)
            )
            
            # Ajouter la position de couverture
            self.add_position(OptionPosition(hedge_option, int(quantity), "call"))
    
    def get_portfolio_value(self) -> float:
        """Calcule la valeur totale du portefeuille."""
        total_value = self.underlying_position * self.S0
        for position in self.positions:
            total_value += position.get_price()
        return total_value

#%%
def example_usage():
    """Exemple d'utilisation du portefeuille hedgé."""
    # Paramètres du marché
    S0 = 100.0
    r = 0.05
    q = 0.0
    
    # Créer le portefeuille
    portfolio = HedgedPortfolio(S0, r, q)
    
    # Créer une position d'option
    option1 = BlackScholesPricer(S0, 100, 1.0, 0.2, r, q)
    position1 = OptionPosition(option1, 1, "call")
    portfolio.add_position(position1)
    
    # Hedge le portefeuille
    portfolio.hedge_portfolio(target_delta=0.0, target_gamma=0.0, target_vega=0.0)
    
    # Afficher les grecques finales
    final_greeks = portfolio.get_portfolio_greeks(S0, 1.0)
    print("Grecques finales du portefeuille:")
    print(f"Delta: {final_greeks['delta']:.4f}")
    print(f"Gamma: {final_greeks['gamma']:.4f}")
    print(f"Vega: {final_greeks['vega']:.4f}")
    print(f"Valeur du portefeuille: {portfolio.get_portfolio_value():.2f}")


def calculate_hedge_coefficients(
    short_option: BlackScholesPricer,
    hedge_options: List[BlackScholesPricer],
    S: float,
    tau: float
) -> Dict[str, float]:
    """
    Calcule les coefficients de couverture pour une position short.
    
    Args:
        short_option: L'option short que l'on veut couvrir
        hedge_options: Liste des options disponibles pour la couverture
        S: Prix du sous-jacent
        tau: Temps jusqu'à l'expiration
    
    Returns:
        Dict contenant les coefficients de couverture pour chaque option
    """
    # Calculer les grecques de l'option short
    short_greeks = Greeks(short_option)
    short_delta = -short_greeks.delta(S, tau, "call")  # Négatif car position short
    short_gamma = -short_greeks.gamma(S, tau)
    short_vega = -short_greeks.vega(S, tau)
    
    # Créer la matrice des grecques pour les options de couverture
    n_hedge_options = len(hedge_options)
    A = np.zeros((3, n_hedge_options))
    
    for i, hedge_option in enumerate(hedge_options):
        hedge_greeks = Greeks(hedge_option)
        A[0, i] = hedge_greeks.delta(S, tau, "call")
        A[1, i] = hedge_greeks.gamma(S, tau)
        A[2, i] = hedge_greeks.vega(S, tau)
    
    # Vecteur des grecques à couvrir
    b = np.array([short_delta, short_gamma, short_vega])
    
    # Résoudre le système d'équations pour trouver les coefficients
    try:
        coefficients = np.linalg.solve(A.T @ A, A.T @ b)
        return {f"option_{i}": coeff for i, coeff in enumerate(coefficients)}
    except np.linalg.LinAlgError:
        print("Le système d'équations n'a pas de solution unique.")
        return {}


def example_hedge_calculation():
    """Exemple de calcul des coefficients de couverture."""
    # Paramètres du marché
    S0 = 100.0
    r = 0.05
    q = 0.0
    sigma = 0.2
    T = 1.0
    
    # Créer l'option short (ATM call)
    short_option = BlackScholesPricer(S0, S0, T, sigma, r, q)
    
    # Créer les options de couverture (différents strikes)
    hedge_options = [
        BlackScholesPricer(S0, S0 * 0.9, T, sigma, r, q),  # OTM call
        BlackScholesPricer(S0, S0 * 1.1, T, sigma, r, q),  # ITM call
        BlackScholesPricer(S0, S0, T, sigma, r, q)         # ATM call
    ]
    
    # Calculer les coefficients de couverture
    coefficients = calculate_hedge_coefficients(short_option, hedge_options, S0, T)
    
    # Afficher les résultats
    print("Coefficients de couverture:")
    for option, coeff in coefficients.items():
        print(f"{option}: {coeff:.4f}")
    
    # Vérifier la neutralisation
    total_delta = 0
    total_gamma = 0
    total_vega = 0
    
    # Ajouter la contribution de l'option short
    short_greeks = Greeks(short_option)
    total_delta -= short_greeks.delta(S0, T, "call")
    total_gamma -= short_greeks.gamma(S0, T)
    total_vega -= short_greeks.vega(S0, T)
    
    # Ajouter la contribution des options de couverture
    for i, hedge_option in enumerate(hedge_options):
        hedge_greeks = Greeks(hedge_option)
        coeff = coefficients[f"option_{i}"]
        total_delta += coeff * hedge_greeks.delta(S0, T, "call")
        total_gamma += coeff * hedge_greeks.gamma(S0, T)
        total_vega += coeff * hedge_greeks.vega(S0, T)
    
    print("\nVérification de la neutralisation:")
    print(f"Delta total: {total_delta:.6f}")
    print(f"Gamma total: {total_gamma:.6f}")
    print(f"Vega total: {total_vega:.6f}")


if __name__ == "__main__":
    example_hedge_calculation() 
# %%
