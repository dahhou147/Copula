# %%
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd

from pricing_model import BlackScholesPricer, VolatilitySmile

ANNUALIZATION_FACTOR = 252 
DEFAULT_RISK_FREE_RATE = 0.03
MIN_TRADING_VOLUME = 10
MIN_IMPLIED_VOL = 0.01
MAX_IMPLIED_VOL = 2.0
PLOT_FIGSIZE = (12, 6)


class ModelCalibrator:
    """
    Class for calibrating option pricing model parameters from market data.
    
    This class handles the fetching of market data, calculation of implied volatilities,
    and calibration of model parameters for a specific stock ticker.
    """

    def __init__(self, ticker: str):
        """
        Initialize the calibrator with a specific ticker.

        Args:
            ticker (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
        """
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("Ticker must be a non-empty string")

        self.ticker = ticker
        self.stock_data: Optional[pd.DataFrame] = None
        self.option_chain = None
        self.spot_price: Optional[float] = None
        self.mu: Optional[float] = None
        self.risk_free_rate: Optional[float] = None
        self.dividend_yield: Optional[float] = None
        self.historical_volatility: Optional[float] = None
        self.expiry_dates = None

    def fetch_market_data(self, period: str = "1y") -> None:
        """
        Fetch market data for the specified ticker.

        Args:
            period (str): Period for historical data (e.g., '1y', '6mo')

        Raises:
            ValueError: If data fetching fails or required data is missing
        """
        try:
            stock = yf.Ticker(self.ticker)
            self.stock_data = stock.history(period=period)

            if self.stock_data.empty:
                raise ValueError(f"No historical data available for {self.ticker}")

            self.spot_price = float(self.stock_data["Close"].iloc[-1])
            
            # Calculate historical metrics
            log_returns = np.log(self.stock_data["Close"] / self.stock_data["Close"].shift(1))
            self.historical_volatility = float(log_returns.std() * np.sqrt(ANNUALIZATION_FACTOR))
            self.mu = float(log_returns.mean() * ANNUALIZATION_FACTOR)
            
            # Get dividend yield and options data
            self.dividend_yield = float(stock.info.get("dividendYield", 0.0))
            self.risk_free_rate = DEFAULT_RISK_FREE_RATE
            self.option_chain = stock.option_chain
            self.expiry_dates = stock.options

            self._print_market_data()

        except Exception as e:
            raise ValueError(f"Failed to fetch market data: {str(e)}")

    def _print_market_data(self) -> None:
        """Print fetched market data summary."""
        print(f"Data retrieved for {self.ticker}")
        print(f"Spot price: {self.spot_price:.2f}")
        print(f"Historical volatility: {self.historical_volatility:.2%}")
        print(f"Dividend yield: {self.dividend_yield:.2%}")
        print(f"Risk-free rate: {self.risk_free_rate:.2%}")
        print(f"Available expiry dates: {self.expiry_dates}")

    def get_option_data(self, expiry_date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Retrieve option data for a specific expiry date.

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            tuple: (calls_df, puts_df) DataFrames for calls and puts

        Raises:
            ValueError: If option data is not available
        """
        if self.option_chain is None:
            raise ValueError("Option data not available. Run fetch_market_data() first.")

        try:
            options = self.option_chain(expiry_date)
            return options.calls, options.puts
        except Exception as e:
            print(f"No options available for date {expiry_date}: {e}")
            return None, None

    def calculate_time_to_maturity(self, expiry_date: str) -> float:
        """
        Calculate time to maturity in years.

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            float: Time to maturity in years
        """
        try:
            expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
            today = datetime.now()
            days = (expiry - today).days
            return max(days, 1) / 365.0
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")

    def calibrate_implied_volatility(self, expiry_date: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate implied volatility for different strikes.

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            tuple: (strikes, implied_vols) Strike prices and implied volatilities

        Raises:
            ValueError: If calibration fails or not enough data
        """
        calls, _ = self.get_option_data(expiry_date)
        if calls is None:
            raise ValueError("No call options data available")

        calls = calls[calls["volume"] > MIN_TRADING_VOLUME]
        if calls.empty:
            raise ValueError("No options with sufficient trading volume")

        maturity = self.calculate_time_to_maturity(expiry_date)
        pricer = self._create_pricer(strike=100, maturity=maturity)  # Default strike will be updated
        smile_calculator = VolatilitySmile(pricer)

        strikes = []
        implied_vols = []

        for _, row in calls.iterrows():
            strike = row["strike"]
            market_price = row["lastPrice"]

            iv = smile_calculator.implied_volatility(strike, market_price)
            if not np.isnan(iv) and MIN_IMPLIED_VOL < iv < MAX_IMPLIED_VOL:
                strikes.append(strike)
                implied_vols.append(iv)

        if not strikes:
            raise ValueError("No valid implied volatilities found")

        return np.array(strikes), np.array(implied_vols)

    def _create_pricer(self, strike: float, maturity: float) -> BlackScholesPricer:
        """
        Create a Black-Scholes pricer with current market parameters.

        Args:
            strike (float): Option strike price
            maturity (float): Time to maturity in years

        Returns:
            BlackScholesPricer: Configured pricer instance
        """
        return BlackScholesPricer(
            S0=self.spot_price,
            K=strike,
            T=maturity,
            sigma=self.historical_volatility,
            r=self.risk_free_rate,
            q=self.dividend_yield,
        )

    def plot_volatility_smile(self, expiry_date: str) -> None:
        """
        Plot the volatility smile for a given expiry date.

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Raises:
            ValueError: If plotting fails
        """
        try:
            strikes, implied_vols = self.calibrate_implied_volatility(expiry_date)
            moneyness = strikes / self.spot_price

            plt.figure(figsize=PLOT_FIGSIZE)
            plt.scatter(moneyness, implied_vols * 100, marker="o")
            plt.plot(moneyness, implied_vols * 100, "r--")
            plt.axvline(x=1, color="gray", linestyle="--")
            plt.xlabel("Moneyness (K/S)")
            plt.ylabel("Implied Volatility (%)")
            plt.title(f"Volatility Smile for {self.ticker} - {expiry_date}")
            plt.grid(True)
            plt.savefig(f"smile_volatility_{self.ticker}_{expiry_date}.png")
            plt.close()

        except Exception as e:
            raise ValueError(f"Failed to plot volatility smile: {str(e)}")

    def calibrate_model(self, expiry_date: str) -> Optional[Dict[str, float]]:
        """
        Calibrate model parameters to match market prices.

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            Optional[Dict[str, float]]: Calibrated parameters or None if calibration fails

        Raises:
            ValueError: If calibration prerequisites are not met
        """
        calls, _ = self.get_option_data(expiry_date)
        if calls is None:
            raise ValueError("No call options data available")

        calls = calls[calls["volume"] > MIN_TRADING_VOLUME]
        if len(calls) < 5:
            raise ValueError("Not enough options data for calibration")

        maturity = self.calculate_time_to_maturity(expiry_date)

        def objective_function(params: np.ndarray) -> float:
            """Calculate total squared error between model and market prices."""
            sigma = params[0]
            pricer = self._create_pricer(strike=100, maturity=maturity)
            pricer.sigma = sigma

            total_error = 0
            for _, row in calls.iterrows():
                strike = row["strike"]
                market_price = row["lastPrice"]
                pricer.K = strike
                model_price = pricer.price_call()
                total_error += (model_price - market_price) ** 2

            return total_error

        try:
            result = minimize(
                objective_function,
                x0=[self.historical_volatility],
                bounds=[(MIN_IMPLIED_VOL, MAX_IMPLIED_VOL)],
                method="L-BFGS-B"
            )

            if not result.success:
                return None

            calibrated_sigma = result.x[0]
            self._print_calibration_results(calibrated_sigma, expiry_date)

            return {
                "S0": self.spot_price,
                "mu": self.mu,
                "sigma": calibrated_sigma,
                "r": self.risk_free_rate,
                "q": self.dividend_yield,
                "T": maturity,
            }

        except Exception as e:
            print(f"Calibration failed: {str(e)}")
            return None

    def _print_calibration_results(self, calibrated_sigma: float, expiry_date: str) -> None:
        """Print calibration results summary."""
        print(f"Successful calibration for {self.ticker} - {expiry_date}")
        print(f"Calibrated volatility: {calibrated_sigma:.2%}")
        print(f"Historical volatility: {self.historical_volatility:.2%}")


# %%
# Usage example
if __name__ == "__main__":
    calibrator = ModelCalibrator("AAPL")

    calibrator.fetch_market_data()

    if calibrator.expiry_dates and len(calibrator.expiry_dates) > 0:
        expiry = calibrator.expiry_dates[0]

        calibrator.plot_volatility_smile(expiry)

        params = calibrator.calibrate_model(expiry)

        if params:
            pricer = BlackScholesPricer(
                S0=params["S0"],
                K=params["S0"],
                T=params["T"],
                sigma=params["sigma"],
                r=params["r"],
                q=params["q"],
            )

            print(f"ATM call option price: {pricer.price_call():.2f}")

# %%
