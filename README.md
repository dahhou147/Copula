# Copula

Copula est une bibliothèque conçue pour simuler et calibrer des modèles de tarification pour diverses options (call, put, barrière, butterfly).

## Installation

```bash
pip install -r requirements.txt
```

## Fonctionnalités

- Tarification d'options européennes (call/put) avec le modèle Black-Scholes
- Calcul des grecques (delta)
- Calcul de volatilité implicite
- Construction de smiles de volatilité
- Simulation de mouvement brownien géométrique
- Simulation de stratégies de couverture delta
- Calibration automatique des paramètres du modèle à partir de données de marché réelles
- Implémentation du modèle de market making Avellaneda-Stoikov (2008)
- Visualisation des résultats

## Exemples d'utilisation

### Tarification d'une option européenne

```python
from pricing_model import EuropeanOptionPricing

# Configuration des paramètres
params = {
    'S0': 100,            # Prix spot
    'strike_price': 100,  # Prix d'exercice
    'maturity': 1,        # Maturité en années
    'sigma': 0.2,         # Volatilité
    'r': 0.05,            # Taux sans risque
    'dividend': 0.0       # Rendement de dividende
}

# Création de l'objet option
option = EuropeanOptionPricing(**params)

# Calcul du prix de l'option call
prix_call = option.price_call()
print(f"Prix de l'option call: {prix_call:.2f}")
```

### Calibration à partir de données de marché

```python
from calibration import ModelCalibrator

# Initialisation du calibrateur avec un ticker boursier
calibrator = ModelCalibrator("AAPL")

# Récupération des données de marché
calibrator.fetch_market_data()

# Si des dates d'expiration sont disponibles
if calibrator.expiry_dates and len(calibrator.expiry_dates) > 0:
    # Choisir une date d'expiration
    expiry = calibrator.expiry_dates[0]
    
    # Afficher le smile de volatilité
    calibrator.plot_volatility_smile(expiry, option_type='call')
    
    # Calibrer le modèle
    params = calibrator.calibrate_model(expiry, option_type='call')
```

### Utilisation de l'outil de calibration interactif

Pour calibrer facilement le modèle avec une interface interactive:

```bash
python example_calibration.py
```

Suivez les instructions à l'écran pour:
1. Entrer le symbole de l'action (ex: AAPL, MSFT, GOOGL)
2. Sélectionner une date d'expiration parmi celles disponibles
3. Choisir le type d'option (call ou put)

L'outil affichera ensuite le smile de volatilité et les paramètres calibrés.

### Market Making avec le modèle Avellaneda-Stoikov

Pour simuler une stratégie de market making basée sur le modèle Avellaneda-Stoikov:

```bash
python example_market_making.py
```

Paramètres configurables:
- `--initial_price` : Prix initial de l'actif
- `--volatility` : Volatilité du prix de l'actif
- `--risk_aversion` : Paramètre d'aversion au risque (gamma)
- `--order_book_liquidity` : Paramètre de liquidité du carnet d'ordres (kappa)
- `--n_simulations` : Nombre de simulations à exécuter

```python
from market_making import AvellanedaStoikovMarketMaker

# Initialiser le market maker
mm = AvellanedaStoikovMarketMaker(
    initial_price=100.0,
    volatility=0.2,
    risk_aversion=0.1,
    order_book_liquidity=1.5
)

# Simuler une session de trading
mm.simulate_trading_session(n_steps=252, seed=42)

# Afficher les résultats
mm.plot_results()
```

Le modèle Avellaneda-Stoikov résout deux problèmes principaux des market makers:
1. La gestion du risque d'inventaire
2. La détermination du spread optimal entre prix d'achat et de vente

## Licence

MIT
