import argparse
import numpy as np
import matplotlib.pyplot as plt
from market_making import AvellanedaStoikovMarketMaker

def main():
    """
    Programme principal démontrant l'utilisation du modèle de market making Avellaneda-Stoikov
    """
    # Configurer l'analyseur d'arguments
    parser = argparse.ArgumentParser(description="Simuler la stratégie de market making Avellaneda-Stoikov")
    
    # Paramètres du modèle
    parser.add_argument("--initial_price", type=float, default=100.0, help="Prix initial de l'actif")
    parser.add_argument("--initial_inventory", type=float, default=0.0, help="Inventaire initial du market maker")
    parser.add_argument("--target_inventory", type=float, default=0.0, help="Inventaire cible du market maker")
    parser.add_argument("--volatility", type=float, default=0.2, help="Volatilité du prix de l'actif")
    parser.add_argument("--risk_aversion", type=float, default=0.1, help="Paramètre d'aversion au risque (gamma)")
    parser.add_argument("--order_book_liquidity", type=float, default=1.5, help="Paramètre de liquidité du carnet d'ordres (kappa)")
    parser.add_argument("--trading_duration", type=float, default=1.0, help="Durée de la session de trading en années")
    parser.add_argument("--dt", type=float, default=1/252, help="Pas de temps pour la simulation")
    
    # Paramètres de simulation
    parser.add_argument("--n_steps", type=int, default=252, help="Nombre de pas de temps pour la simulation")
    parser.add_argument("--arrival_rate", type=float, default=0.5, help="Taux d'arrivée des ordres")
    parser.add_argument("--seed", type=int, default=42, help="Graine pour la génération de nombres aléatoires")
    parser.add_argument("--n_simulations", type=int, default=1, help="Nombre de simulations à exécuter")
    parser.add_argument("--output_prefix", type=str, default="mm_simulation", help="Préfixe pour les fichiers de sortie")
    
    # Analyser les arguments
    args = parser.parse_args()
    
    if args.n_simulations == 1:
        # Exécuter une seule simulation
        run_single_simulation(args)
    else:
        # Exécuter plusieurs simulations
        run_multiple_simulations(args)
    
def run_single_simulation(args):
    """
    Exécute une seule simulation et affiche les résultats
    """
    print(f"Exécution d'une simulation avec les paramètres suivants:")
    print(f"  Prix initial: {args.initial_price}")
    print(f"  Volatilité: {args.volatility}")
    print(f"  Aversion au risque (gamma): {args.risk_aversion}")
    print(f"  Liquidité du carnet d'ordres (kappa): {args.order_book_liquidity}")
    
    # Initialiser le market maker
    mm = AvellanedaStoikovMarketMaker(
        initial_price=args.initial_price,
        initial_inventory=args.initial_inventory,
        target_inventory=args.target_inventory,
        volatility=args.volatility,
        risk_aversion=args.risk_aversion,
        order_book_liquidity=args.order_book_liquidity,
        trading_duration=args.trading_duration,
        dt=args.dt
    )
    
    # Simuler une session de trading
    mm.simulate_trading_session(n_steps=args.n_steps, arrival_rate=args.arrival_rate, seed=args.seed)
    
    # Afficher les résultats
    fig = mm.plot_results()
    output_file = f"{args.output_prefix}_results.png"
    plt.savefig(output_file)
    print(f"Résultats de la simulation enregistrés dans {output_file}")
    
    # Analyser les performances
    perf = mm.analyze_performance()
    print("\nPerformances de la stratégie:")
    for key, value in perf.items():
        print(f"  {key}: {value:.2f}")
    
def run_multiple_simulations(args):
    """
    Exécute plusieurs simulations et analyse la distribution des résultats
    """
    print(f"Exécution de {args.n_simulations} simulations avec les paramètres suivants:")
    print(f"  Prix initial: {args.initial_price}")
    print(f"  Volatilité: {args.volatility}")
    print(f"  Aversion au risque (gamma): {args.risk_aversion}")
    print(f"  Liquidité du carnet d'ordres (kappa): {args.order_book_liquidity}")
    
    # Exécuter plusieurs simulations
    results = AvellanedaStoikovMarketMaker.run_multiple_simulations(
        n_simulations=args.n_simulations,
        initial_price=args.initial_price,
        initial_inventory=args.initial_inventory,
        target_inventory=args.target_inventory,
        volatility=args.volatility,
        risk_aversion=args.risk_aversion,
        order_book_liquidity=args.order_book_liquidity,
        trading_duration=args.trading_duration,
        dt=args.dt
    )
    
    # Enregistrer les résultats dans un fichier CSV
    output_csv = f"{args.output_prefix}_results.csv"
    results.to_csv(output_csv, index=False)
    print(f"Résultats des simulations enregistrés dans {output_csv}")
    
    # Afficher la distribution des résultats
    fig = AvellanedaStoikovMarketMaker.plot_simulation_distribution(results)
    output_file = f"{args.output_prefix}_distribution.png"
    plt.savefig(output_file)
    print(f"Distribution des résultats enregistrée dans {output_file}")
    
    # Afficher les statistiques descriptives
    print("\nStatistiques descriptives:")
    print(results.describe())

def experiments_with_different_parameters():
    """
    Exécute des expériences avec différentes valeurs de paramètres
    pour étudier leur impact sur les performances
    """
    # Paramètres de base
    base_params = {
        'initial_price': 100.0,
        'initial_inventory': 1000.0,
        'target_inventory': 0.0,
        'volatility': 0.2,
        'risk_aversion': 0.1,
        'order_book_liquidity': 1.5,
        'trading_duration': 1.0,
        'dt': 1/252
    }
    
    # Valeurs de gamma (aversion au risque) à tester
    gamma_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    # Exécuter des simulations pour chaque valeur de gamma
    gamma_results = []
    
    for gamma in gamma_values:
        params = base_params.copy()
        params['risk_aversion'] = gamma
        
        mm = AvellanedaStoikovMarketMaker(**params)
        mm.simulate_trading_session(n_steps=252, seed=42)
        
        perf = mm.analyze_performance()
        perf['gamma'] = gamma
        
        gamma_results.append(perf)
    
    # Créer un DataFrame avec les résultats
    import pandas as pd
    gamma_df = pd.DataFrame(gamma_results)
    
    # Tracer l'impact de gamma sur le PnL final et le spread moyen
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    axs[0].plot(gamma_df['gamma'], gamma_df['final_pnl'], 'o-')
    axs[0].set_xlabel('Aversion au risque (gamma)')
    axs[0].set_ylabel('PnL final')
    axs[0].grid(True)
    axs[0].set_title('Impact de gamma sur le PnL final')
    
    axs[1].plot(gamma_df['gamma'], gamma_df['avg_spread'], 'o-')
    axs[1].set_xlabel('Aversion au risque (gamma)')
    axs[1].set_ylabel('Spread moyen')
    axs[1].grid(True)
    axs[1].set_title('Impact de gamma sur le spread moyen')
    
    plt.tight_layout()
    plt.savefig('gamma_impact.png')
    print("Étude de l'impact de gamma enregistrée dans gamma_impact.png")

if __name__ == "__main__":
    main()
    
    # Décommenter pour exécuter des expériences avec différentes valeurs de paramètres
    # experiments_with_different_parameters() 