import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pricing_model import EuropeanOptionPricing, geometric_brownian_motion

class AvellanedaStoikovMarketMaker:
    """
    Implémentation du modèle de market making Avellaneda-Stoikov (2008)
    
    Ce modèle résout deux problèmes principaux des market makers:
    1. La gestion du risque d'inventaire
    2. La détermination du spread optimal entre prix d'achat et de vente
    """
    
    def __init__(
        self,
        initial_price: float,
        initial_inventory: float = 0.0,
        target_inventory: float = 0.0,
        volatility: float = 0.2,
        risk_aversion: float = 0.1,
        order_book_liquidity: float = 1.5,
        trading_duration: float = 1.0,
        dt: float = 1/252
    ):
        """
        Initialise le market maker avec les paramètres spécifiés.
        
        Args:
            initial_price: Prix initial de l'actif
            initial_inventory: Inventaire initial du market maker
            target_inventory: Inventaire cible du market maker
            volatility: Volatilité du prix de l'actif (sigma)
            risk_aversion: Paramètre d'aversion au risque d'inventaire (gamma)
            order_book_liquidity: Paramètre de liquidité du carnet d'ordres (kappa)
            trading_duration: Durée de la session de trading en unités de temps (T)
            dt: Pas de temps pour la simulation
        """
        self.price = initial_price
        self.inventory = initial_inventory
        self.target_inventory = target_inventory
        self.volatility = volatility
        self.risk_aversion = risk_aversion
        self.order_book_liquidity = order_book_liquidity
        self.trading_duration = trading_duration
        self.dt = dt
        
        # État de la simulation
        self.current_time = 0.0
        self.cash = 0.0
        self.pnl_history = []
        self.price_history = []
        self.inventory_history = []
        self.reservation_price_history = []
        self.bid_price_history = []
        self.ask_price_history = []
        
    def reservation_price(self) -> float:
        """
        Calcule le prix de réservation en fonction de l'inventaire actuel.
        
        Le prix de réservation est le prix de référence pour placer les ordres,
        qui tient compte du déséquilibre d'inventaire.
        
        Returns:
            Prix de réservation
        """
        # Déséquilibre d'inventaire par rapport à la cible
        q = self.inventory - self.target_inventory
        
        # Temps restant jusqu'à la fin de la session
        time_remaining = self.trading_duration - self.current_time
        
        # Prix de réservation selon Avellaneda-Stoikov
        r = self.price - q * self.risk_aversion * self.volatility**2 * time_remaining
        
        return r
    
    def optimal_spread(self) -> float:
        """
        Calcule le spread optimal entre les prix d'achat et de vente.
        
        Returns:
            Spread optimal
        """
        # Temps restant jusqu'à la fin de la session
        time_remaining = self.trading_duration - self.current_time
        
        # Spread optimal selon Avellaneda-Stoikov
        spread = (self.risk_aversion * self.volatility**2 * time_remaining + 
                  2/self.risk_aversion * np.log(1 + self.risk_aversion/self.order_book_liquidity))
        
        return spread
    
    def bid_ask_prices(self) -> tuple:
        """
        Calcule les prix d'achat (bid) et de vente (ask) optimaux.
        
        Returns:
            Tuple contenant (bid_price, ask_price)
        """
        r = self.reservation_price()
        s = self.optimal_spread()
        
        bid_price = r - s/2
        ask_price = r + s/2
        
        return bid_price, ask_price
    
    def update_state(self, new_price: float, trade_type: str = None, trade_volume: float = 0.0):
        """
        Met à jour l'état du market maker suite à un changement de prix et/ou à une transaction.
        
        Args:
            new_price: Nouveau prix du marché
            trade_type: Type de transaction ('buy', 'sell', ou None si pas de transaction)
            trade_volume: Volume de la transaction
        """
        old_price = self.price
        self.price = new_price
        
        # Mise à jour du temps
        self.current_time += self.dt
        
        # Mise à jour de l'inventaire et de la trésorerie en cas de transaction
        if trade_type == 'buy':
            self.inventory += trade_volume
            self.cash -= trade_volume * new_price
        elif trade_type == 'sell':
            self.inventory -= trade_volume
            self.cash += trade_volume * new_price
        
        # Calcul du PnL actuel
        pnl = self.cash + self.inventory * new_price
        
        # Mettre à jour les historiques
        self.price_history.append(new_price)
        self.inventory_history.append(self.inventory)
        self.pnl_history.append(pnl)
        
        r = self.reservation_price()
        bid, ask = self.bid_ask_prices()
        
        self.reservation_price_history.append(r)
        self.bid_price_history.append(bid)
        self.ask_price_history.append(ask)
    
    def simulate_trading_session(self, n_steps: int, arrival_rate: float = 0.5, seed: int = None):
        """
        Simule une session de trading complète avec des arrivées d'ordres aléatoires.
        
        Args:
            n_steps: Nombre de pas de temps pour la simulation
            arrival_rate: Taux d'arrivée des ordres (lambda)
            seed: Graine pour la génération de nombres aléatoires
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Générer un chemin de prix par mouvement brownien géométrique
        _, price_path = geometric_brownian_motion(
            S0=self.price,
            mu=0.0,
            sigma=self.volatility,
            N=n_steps,
            T=self.trading_duration,
            M=1
        )
        price_path = price_path.flatten()
        
        # Réinitialiser l'état
        self.current_time = 0.0
        self.cash = 0.0
        self.pnl_history = []
        self.price_history = []
        self.inventory_history = []
        self.reservation_price_history = []
        self.bid_price_history = []
        self.ask_price_history = []
        
        # Simuler la session de trading
        for step in range(n_steps):
            new_price = price_path[step]
            
            # Calculer les prix bid/ask
            bid_price, ask_price = self.bid_ask_prices()
            
            # Simuler les arrivées d'ordres aléatoires
            if np.random.random() < arrival_rate * self.dt:
                # Déterminer si l'ordre vient du côté acheteur ou vendeur
                if np.random.random() < 0.5:
                    # Ordre du côté acheteur (quelqu'un achète à notre ask)
                    if ask_price <= new_price * (1 + 0.01):  # Simplification: l'ordre est exécuté si notre ask est raisonnable
                        self.update_state(new_price, 'sell', 1.0)
                    else:
                        self.update_state(new_price)
                else:
                    # Ordre du côté vendeur (quelqu'un vend à notre bid)
                    if bid_price >= new_price * (1 - 0.01):  # Simplification: l'ordre est exécuté si notre bid est raisonnable
                        self.update_state(new_price, 'buy', 1.0)
                    else:
                        self.update_state(new_price)
            else:
                # Pas d'ordre, mettre à jour l'état sans transaction
                self.update_state(new_price)
    
    def plot_results(self):
        """
        Affiche les résultats de la simulation.
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Évolution des prix
        axs[0].plot(self.price_history, 'b-', label='Prix du marché')
        axs[0].plot(self.reservation_price_history, 'g-', label='Prix de réservation')
        axs[0].plot(self.bid_price_history, 'r--', label='Prix d\'achat (bid)')
        axs[0].plot(self.ask_price_history, 'r--', label='Prix de vente (ask)')
        axs[0].set_ylabel('Prix')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_title('Évolution des prix')
        
        # Évolution du PnL
        axs[1].plot(self.pnl_history, 'g-')
        axs[1].set_ylabel('PnL')
        axs[1].grid(True)
        axs[1].set_title('Évolution du Profit & Loss')
        
        # Évolution de l'inventaire
        axs[2].plot(self.inventory_history, 'b-')
        axs[2].axhline(y=self.target_inventory, color='r', linestyle='--', label='Inventaire cible')
        axs[2].set_ylabel('Inventaire')
        axs[2].set_xlabel('Pas de temps')
        axs[2].grid(True)
        axs[2].legend()
        axs[2].set_title('Évolution de l\'inventaire')
        
        plt.tight_layout()
        return fig
    
    def analyze_performance(self):
        """
        Analyse les performances de la stratégie de market making.
        
        Returns:
            Dict contenant les métriques de performance
        """
        final_pnl = self.pnl_history[-1] if self.pnl_history else 0
        max_pnl = max(self.pnl_history) if self.pnl_history else 0
        min_pnl = min(self.pnl_history) if self.pnl_history else 0
        
        max_inventory = max(self.inventory_history) if self.inventory_history else 0
        min_inventory = min(self.inventory_history) if self.inventory_history else 0
        final_inventory = self.inventory_history[-1] if self.inventory_history else 0
        
        avg_spread = np.mean([ask - bid for bid, ask in zip(self.bid_price_history, self.ask_price_history)])
        
        return {
            'final_pnl': final_pnl,
            'max_pnl': max_pnl,
            'min_pnl': min_pnl,
            'max_inventory': max_inventory,
            'min_inventory': min_inventory,
            'final_inventory': final_inventory,
            'avg_spread': avg_spread
        }
    
    @staticmethod
    def run_multiple_simulations(n_simulations: int, **kwargs):
        """
        Exécute plusieurs simulations et analyse la distribution des résultats.
        
        Args:
            n_simulations: Nombre de simulations à exécuter
            **kwargs: Arguments pour initialiser le market maker
        
        Returns:
            DataFrame contenant les résultats de toutes les simulations
        """
        results = []
        
        for i in range(n_simulations):
            mm = AvellanedaStoikovMarketMaker(**kwargs)
            mm.simulate_trading_session(n_steps=252, seed=i)
            
            perf = mm.analyze_performance()
            perf['simulation'] = i
            
            results.append(perf)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def plot_simulation_distribution(results_df):
        """
        Affiche la distribution des résultats de plusieurs simulations.
        
        Args:
            results_df: DataFrame contenant les résultats des simulations
        """
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distribution du PnL final
        axs[0].hist(results_df['final_pnl'], bins=30, alpha=0.7)
        axs[0].axvline(results_df['final_pnl'].mean(), color='r', linestyle='--', 
                        label=f'Moyenne: {results_df["final_pnl"].mean():.2f}')
        axs[0].set_xlabel('PnL final')
        axs[0].set_ylabel('Fréquence')
        axs[0].legend()
        axs[0].grid(True)
        
        # Distribution de l'inventaire final
        axs[1].hist(results_df['final_inventory'], bins=30, alpha=0.7)
        axs[1].axvline(results_df['final_inventory'].mean(), color='r', linestyle='--',
                        label=f'Moyenne: {results_df["final_inventory"].mean():.2f}')
        axs[1].set_xlabel('Inventaire final')
        axs[1].set_ylabel('Fréquence')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        return fig


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le market maker
    mm = AvellanedaStoikovMarketMaker(
        initial_price=100.0,
        initial_inventory=1000.0,
        target_inventory=0.0,
        volatility=0.2,
        risk_aversion=0.1,
        order_book_liquidity=1.5,
        trading_duration=1.0,
        dt=1/252
    )
    
    # Simuler une session de trading
    mm.simulate_trading_session(n_steps=252, arrival_rate=0.5, seed=42)
    
    # Afficher les résultats
    fig = mm.plot_results()
    plt.savefig('mm_simulation_results.png')
    
    # Analyser les performances
    perf = mm.analyze_performance()
    print("Performances de la stratégie:")
    for key, value in perf.items():
        print(f"  {key}: {value:.2f}")
    
    # Exécuter plusieurs simulations
    results = AvellanedaStoikovMarketMaker.run_multiple_simulations(
        n_simulations=100,
        initial_price=100.0,
        initial_inventory=0.0,
        target_inventory=0.0,
        volatility=0.2,
        risk_aversion=0.1,
        order_book_liquidity=1.5,
        trading_duration=1.0,
        dt=1/252
    )
    
    # Afficher la distribution des résultats
    fig2 = AvellanedaStoikovMarketMaker.plot_simulation_distribution(results)
    plt.savefig('mm_simulation_distribution.png') 