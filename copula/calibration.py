import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime
from pricing_model import EuropeanOptionPricing

class ModelCalibrator:
    """Classe permettant de calibrer les paramètres d'un modèle de pricing d'options
    à partir de données de marché pour un ticker spécifique"""
    
    def __init__(self, ticker): 
        """
        Initialise le calibrateur avec un ticker spécifique
        
        Args:
            ticker (str): Symbole du titre (ex: 'AAPL', 'GOOGL')
        """
        self.ticker = ticker
        self.stock_data = None
        self.option_chain = None
        self.spot_price = None
        self.risk_free_rate = None
        self.dividend_yield = None
        self.historical_volatility = None
        self.expiry_dates = None
        
    def fetch_market_data(self, period="1y"):
        """
        Récupère les données de marché pour le ticker spécifié
        
        Args:
            period (str): Période pour les données historiques (ex: '1y', '6mo')
        """
        stock = yf.Ticker(self.ticker)
        self.stock_data = stock.history(period=period)
        
        self.spot_price = self.stock_data['Close'].iloc[-1]
        
        log_returns = np.log(self.stock_data['Close'] / self.stock_data['Close'].shift(1))
        self.historical_volatility = log_returns.std() * np.sqrt(252)  # Annualisation
        
        self.dividend_yield = stock.info.get('dividendYield', 0.0)
        
        self.risk_free_rate = 0.03  
        
        self.option_chain = stock.option_chain
        self.expiry_dates = stock.options
        
        print(f"Données récupérées pour {self.ticker}")
        print(f"Prix spot: {self.spot_price:.2f}")
        print(f"Volatilité historique: {self.historical_volatility:.2%}")
        print(f"Taux de dividende: {self.dividend_yield:.2%}")
        print(f"Taux sans risque: {self.risk_free_rate:.2%}")
        print(f"Dates d'expiration disponibles: {self.expiry_dates}")
    
    def get_option_data(self, expiry_date):
        """
        Récupère les données d'options pour une date d'expiration spécifique
        
        Args:
            expiry_date (str): Date d'expiration au format 'YYYY-MM-DD'
        
        Returns:
            tuple: (calls_df, puts_df) DataFrames pour les calls et puts
        """
        if self.option_chain is None:
            raise ValueError("Données d'options non disponibles. Exécutez fetch_market_data() d'abord.")
        
        try:
            options = self.option_chain(expiry_date)
            return options.calls, options.puts
        except Exception as e:
            print(f"Aucune option disponible pour la date {expiry_date}: {e}")
            return None, None
    
    def calculate_time_to_maturity(self, expiry_date):
        """
        Calcule le temps jusqu'à l'échéance en années
        
        Args:
            expiry_date (str): Date d'expiration au format 'YYYY-MM-DD'
        
        Returns:
            float: Temps jusqu'à l'échéance en années
        """
        expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
        today = datetime.now()
        days = (expiry - today).days
        return max(days, 1) / 365.0
    
    def calibrate_implied_volatility(self, expiry_date, option_type='call'):
        """
        Calibre la volatilité implicite pour différents strikes
        
        Args:
            expiry_date (str): Date d'expiration au format 'YYYY-MM-DD'
            option_type (str): Type d'option ('call' ou 'put')
        
        Returns:
            tuple: (strikes, implied_vols) Prix d'exercice et volatilités implicites
        """
        calls, puts = self.get_option_data(expiry_date)
        
        if option_type == 'call':
            options_df = calls
        else:
            options_df = puts
        
        if options_df is None:
            return None, None
        
        options_df = options_df[options_df['volume'] > 10]
        
        maturity = self.calculate_time_to_maturity(expiry_date)
        
        model = EuropeanOptionPricing(
            S0=self.spot_price,
            strike_price=100,  # Sera mis à jour
            maturity=maturity,
            sigma=self.historical_volatility,
            r=self.risk_free_rate,
            dividend=self.dividend_yield
        )
        
        strikes = []
        implied_vols = []
        
        for _, row in options_df.iterrows():
            strike = row['strike']
            market_price = row['lastPrice']
            
            # Calcul de la volatilité implicite
            iv = model.implied_volatility(strike, market_price, option_type)
            
            if not np.isnan(iv) and 0.01 < iv < 2.0:  # Filtre des valeurs aberrantes
                strikes.append(strike)
                implied_vols.append(iv)
        
        return np.array(strikes), np.array(implied_vols)
    
    def plot_volatility_smile(self, expiry_date, option_type='call'):
        """
        Affiche le smile de volatilité pour une date d'expiration donnée
        
        Args:
            expiry_date (str): Date d'expiration au format 'YYYY-MM-DD'
            option_type (str): Type d'option ('call' ou 'put')
        """
        strikes, implied_vols = self.calibrate_implied_volatility(expiry_date, option_type)
        
        if strikes is None or len(strikes) == 0:
            print("Pas assez de données pour tracer le smile de volatilité")
            return
        
        # Calcul des moneyness (K/S)
        moneyness = strikes / self.spot_price
        
        plt.figure(figsize=(12, 6))
        plt.scatter(moneyness, implied_vols * 100, marker='o')
        plt.plot(moneyness, implied_vols * 100, 'r--')
        
        plt.axvline(x=1, color='gray', linestyle='--')
        plt.xlabel('Moneyness (K/S)')
        plt.ylabel('Volatilité implicite (%)')
        plt.title(f'Smile de volatilité pour {self.ticker} - {expiry_date} ({option_type})')
        plt.grid(True)
        plt.savefig(f'smile_volatility_{self.ticker}_{expiry_date}_{option_type}.png')
    
    def calibrate_model(self, expiry_date, option_type='call'):
        """
        Calibre les paramètres du modèle pour correspondre aux prix du marché
        
        Args:
            expiry_date (str): Date d'expiration au format 'YYYY-MM-DD'
            option_type (str): Type d'option ('call' ou 'put')
            
        Returns:
            dict: Paramètres calibrés
        """
        calls, puts = self.get_option_data(expiry_date)
        
        if option_type == 'call':
            options_df = calls
        else:
            options_df = puts
        
        if options_df is None or len(options_df) < 5:
            print("Pas assez de données pour calibrer le modèle")
            return None
        
        options_df = options_df[options_df['volume'] > 10]
        
        maturity = self.calculate_time_to_maturity(expiry_date)
        
        def objective_function(params):
            sigma = params[0]
            
            model = EuropeanOptionPricing(
                S0=self.spot_price,
                strike_price=100,  
                maturity=maturity,
                sigma=sigma,
                r=self.risk_free_rate,
                dividend=self.dividend_yield
            )
            
            total_error = 0
            for _, row in options_df.iterrows():
                strike = row['strike']
                market_price = row['lastPrice']
                
                model.K = strike
                
                if option_type == 'call':
                    model_price = model.price_call()
                else:
                    model_price = model.price_put()
                
                # Erreur quadratique
                error = (model_price - market_price) ** 2
                total_error += error
            
            return total_error
        
        initial_params = [self.historical_volatility]
        bounds = [(0.01, 2.0)]  # Contraintes sur la volatilité
        
        result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            calibrated_sigma = result.x[0]
            
            calibrated_params = {
                'S0': self.spot_price,
                'sigma': calibrated_sigma,
                'r': self.risk_free_rate,
                'q': self.dividend_yield,
                'T': maturity
            }
            
            print(f"Calibration réussie pour {self.ticker} - {expiry_date}")
            print(f"Volatilité calibrée: {calibrated_sigma:.2%}")
            print(f"Volatilité historique: {self.historical_volatility:.2%}")
            
            return calibrated_params
        else:
            print("Échec de la calibration")
            return None

# Exemple d'utilisation
if __name__ == "__main__":
    calibrator = ModelCalibrator("AAPL")
    
    calibrator.fetch_market_data()
    
    if calibrator.expiry_dates and len(calibrator.expiry_dates) > 0:
        expiry = calibrator.expiry_dates[0]
        
        calibrator.plot_volatility_smile(expiry, option_type='call')
        
        params = calibrator.calibrate_model(expiry, option_type='call')
        
        if params:
            model = EuropeanOptionPricing(
                S0=params['S0'],
                strike_price=params['S0'],  # ATM option
                maturity=params['T'],
                sigma=params['sigma'],
                r=params['r'],
                dividend=params['q']
            )
            
            print(f"Prix de l'option call ATM: {model.price_call():.2f}") 