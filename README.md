# Options Hedging Strategy

Ce projet implémente une stratégie de couverture d'options avancée utilisant le modèle Black-Scholes. La stratégie vise à neutraliser les risques delta, gamma et vega d'un portefeuille d'options.

## Fonctionnalités

- **Prix d'options** : Implémentation du modèle Black-Scholes pour le pricing d'options
- **Calcul des Grecques** : Delta, Gamma, Vega, Theta
- **Couverture multi-grecques** : Neutralisation simultanée de delta, gamma et vega
- **Simulation Monte Carlo** : Génération de chemins de prix sous la mesure risque-neutre
- **Analyse de performance** : Métriques de performance et visualisations

## Structure du Code

### Classes Principales

1. **BlackScholesPricer**
   - Pricing d'options européennes
   - Calcul des paramètres d1 et d2
   - Support des options call et put

2. **Greeks**
   - Calcul des grecques (delta, gamma, vega, theta)
   - Support des options call et put

3. **ConstructPortfolio**
   - Construction d'un portefeuille de couverture
   - Neutralisation des risques delta, gamma et vega
   - Gestion dynamique du portefeuille

4. **GirsanovSimulator**
   - Simulation de chemins de prix sous la mesure risque-neutre
   - Utilisation du théorème de Girsanov

### Exemple d'Utilisation

```python
# Paramètres de marché
S0 = 100.0  # Prix initial
K = 100.0   # Strike
T = 1.0     # Maturité
r = 0.05    # Taux sans risque
sigma = 0.2 # Volatilité
N = 252     # Nombre de pas de temps
M = 100     # Nombre de simulations

# Création des chemins de prix
simulator = GirsanovSimulator(S0, mu, r, sigma, N, T, M)
paths = simulator.generate_paths()

# Création du portefeuille de couverture
pricer = BlackScholesPricer(S0, K, T, sigma, r)
portfolio = ConstructPortfolio(pricer, paths, N, T, K*0.9, K*1.1)

# Exécution de la couverture
portfolio.hedge_portfolio(option_type="call")
```

## Installation

```bash
pip install numpy scipy matplotlib
```

## Dépendances

- NumPy
- SciPy
- Matplotlib

## Fonctionnalités Avancées

1. **Couverture Multi-grecques**
   - Utilisation de deux options de couverture avec des strikes différents
   - Position sur le sous-jacent pour la neutralisation complète

2. **Gestion du Risque**
   - Régularisation des coefficients pour éviter les positions extrêmes
   - Gestion des cas d'erreur numériques

3. **Analyse de Performance**
   - Distribution du PnL
   - Évolution de la valeur du portefeuille
   - Position en cash
   - Métriques de performance (Sharpe ratio, etc.)

## Améliorations Futures

- [ ] Ajout des coûts de transaction
- [ ] Support des options américaines
- [ ] Implémentation d'autres modèles de volatilité
- [ ] Optimisation des paramètres de couverture
- [ ] Backtesting sur données historiques

## Auteur

[Votre nom]

## Licence

Ce projet est sous licence MIT.
