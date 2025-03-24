# Copula
Copula is a library designed to simulate and calibrate pricing models for various options (call, put, barrier, butterfly).
### cash flow portfolio

from pricing_model import EuropeanOptionPricing
import numpy as np

# Configuration des paramètres
params = {
    'S0': 100,            # Prix spot
    'strike_price': 100,  # Prix d'exercice
    'maturity': 1,        # Maturité en années
    'sigma': 0.2,         # Volatilité
    'r': 0.05,            # Taux sans risque
    'dividend': 0.0       # Rendement de dividende
}

### Création de l'objet option
option = EuropeanOptionPricing(**params)

### Calcul du prix de l'option call
prix_call = option.price_call()
print(f"Prix de l'option call: {prix_call:.2f}")

### Calculer la volatilité implicite à partir d'un prix de marché
prix_marche = 10.50
vol_implicite = option.implied_volatility(option.K, prix_marche, 'call')
print(f"Volatilité implicite: {vol_implicite:.2%}")

### Définir une gamme de prix d'exercice et de prix de marché correspondants
strikes = np.linspace(80, 120, 9)
market_prices = [/* prix de marché observés */]

### Visualiser le smile de volatilité
option.plot_volatility_smile(strikes, market_prices)
