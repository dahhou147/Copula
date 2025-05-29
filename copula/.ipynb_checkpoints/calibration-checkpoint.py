# %%
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd

from pricing_model import BlackScholesPricer, VolatilitySmile


class ModelCalibrator:
    """Class for calibrating option pricing model parameters
    from market data for a specific ticker"""

    def __init__(self, ticker):
        """
        Initialize the calibrator with a specific ticker

        Args:
            ticker (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
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
        Fetch market data for the specified ticker

        Args:
            period (str): Period for historical data (e.g., '1y', '6mo')
        """
        stock = yf.Ticker(self.ticker)
        self.stock_data = stock.history(period=period)

        self.spot_price = self.stock_data["Close"].iloc[-1]

        log_returns = np.log(self.stock_data["Close"] / self.stock_data["Close"].shift(1))
        self.historical_volatility = log_returns.std() * np.sqrt(252)  # Annualization

        self.dividend_yield = stock.info.get("dividendYield", 0.0)

        self.risk_free_rate = 0.03

        self.option_chain = stock.option_chain
        self.expiry_dates = stock.options

        print(f"Data retrieved for {self.ticker}")
        print(f"Spot price: {self.spot_price:.2f}")
        print(f"Historical volatility: {self.historical_volatility:.2%}")
        print(f"Dividend yield: {self.dividend_yield:.2%}")
        print(f"Risk-free rate: {self.risk_free_rate:.2%}")
        print(f"Available expiry dates: {self.expiry_dates}")

    def get_option_data(self, expiry_date):
        """
        Retrieve option data for a specific expiry date

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            tuple: (calls_df, puts_df) DataFrames for calls and puts
        """
        if self.option_chain is None:
            raise ValueError("Option data not available. Run fetch_market_data() first.")

        try:
            options = self.option_chain(expiry_date)
            return options.calls, options.puts
        except Exception as e:
            print(f"No options available for date {expiry_date}: {e}")
            return None, None

    def calculate_time_to_maturity(self, expiry_date):
        """
        Calculate time to maturity in years

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            float: Time to maturity in years
        """
        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
        today = datetime.now()
        days = (expiry - today).days
        return max(days, 1) / 365.0

    def calibrate_implied_volatility(self, expiry_date):
        """
        Calibrate implied volatility for different strikes

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            tuple: (strikes, implied_vols) Strike prices and implied volatilities
        """
        calls, puts = self.get_option_data(expiry_date)

        if calls is None or puts is None:
            return None, None

        # Use both calls and puts for better coverage of strikes
        options_df = pd.concat([calls, puts])
        options_df = options_df[options_df["volume"] > 10]

        maturity = self.calculate_time_to_maturity(expiry_date)

        pricer = BlackScholesPricer(
            S0=self.spot_price,
            K=100,  # Will be updated
            T=maturity,
            sigma=self.historical_volatility,
            r=self.risk_free_rate,
            q=self.dividend_yield,
        )
        smile_calculator = VolatilitySmile(pricer)

        strikes = []
        implied_vols = []

        for _, row in options_df.iterrows():
            strike = row["strike"]
            market_price = row["lastPrice"]

            # Calculate implied volatility
            iv = smile_calculator.implied_volatility(strike, market_price)
            print(f"Strike: {strike}, Implied Volatility: {iv}")
        return np.array(strikes), np.array(implied_vols)

    def plot_volatility_smile(self, expiry_date):
        """
        Plot the volatility smile for a given expiry date

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format
        """
        strikes, implied_vols = self.calibrate_implied_volatility(expiry_date)

        if strikes is None or len(strikes) == 0:
            raise ValueError("No data to plot volatility smile")

        # Calculate moneyness (K/S)
        moneyness = strikes / self.spot_price

        plt.figure(figsize=(12, 6))
        plt.scatter(moneyness, implied_vols * 100, marker="o")
        plt.plot(moneyness, implied_vols * 100, "r--")

        plt.axvline(x=1, color="gray", linestyle="--")
        plt.xlabel("Moneyness (K/S)")
        plt.ylabel("Implied Volatility (%)")
        plt.title(f"Volatility Smile for {self.ticker} - {expiry_date}")
        plt.grid(True)
        plt.savefig(f"smile_volatility_{self.ticker}_{expiry_date}.png")
        plt.show()

    def calibrate_model(self, expiry_date, option_type="call"):
        """
        Calibrate model parameters to match market prices

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format
            option_type (str): Option type ('call' or 'put')

        Returns:
            dict: Calibrated parameters
        """
        calls, puts = self.get_option_data(expiry_date)

        if option_type == "call":
            options_df = calls
        else:
            options_df = puts

        if options_df is None or len(options_df) < 5:
            print("Not enough data to calibrate the model")
            return None

        options_df = options_df[options_df["volume"] > 10]

        maturity = self.calculate_time_to_maturity(expiry_date)

        def objective_function(params):
            sigma = params[0]

            pricer = BlackScholesPricer(
                S0=self.spot_price,
                K=100,
                T=maturity,
                sigma=sigma,
                r=self.risk_free_rate,
                q=self.dividend_yield,
            )

            total_error = 0
            for _, row in options_df.iterrows():
                strike = row["strike"]
                market_price = row["lastPrice"]

                pricer.K = strike

                if option_type == "call":
                    model_price = pricer.price_call()
                else:
                    model_price = pricer.price_put()

                # Quadratic error
                error = (model_price - market_price) ** 2
                total_error += error

            return total_error

        initial_params = [self.historical_volatility]
        bounds = [(0.01, 2.0)]  # Volatility constraints

        result = minimize(objective_function, initial_params, bounds=bounds, method="L-BFGS-B")

        if result.success:
            calibrated_sigma = result.x[0]

            calibrated_params = {
                "S0": self.spot_price,
                "sigma": calibrated_sigma,
                "r": self.risk_free_rate,
                "q": self.dividend_yield,
                "T": maturity,
            }

            print(f"Successful calibration for {self.ticker} - {expiry_date}")
            print(f"Calibrated volatility: {calibrated_sigma:.2%}")
            print(f"Historical volatility: {self.historical_volatility:.2%}")

            return calibrated_params
        else:
            print("Calibration failed")
            return None


# %%
# Usage example
if __name__ == "__main__":
    calibrator = ModelCalibrator("AAPL")

    calibrator.fetch_market_data()

    # if calibrator.expiry_dates and len(calibrator.expiry_dates) > 0:
    #     expiry = calibrator.expiry_dates[0]

    #     calibrator.plot_volatility_smile(expiry)

    #     params = calibrator.calibrate_model(expiry, option_type="call")

    #     if params:
    #         pricer = BlackScholesPricer(
    #             S0=params["S0"],
    #             K=params["S0"],  # ATM option
    #             T=params["T"],
    #             sigma=params["sigma"],
    #             r=params["r"],
    #             q=params["q"],
    #         )

    #         print(f"ATM call option price: {pricer.price_call():.2f}")

# %%
