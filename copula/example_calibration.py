from typing import Optional, Dict, NoReturn
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from calibration import ModelCalibrator
from pricing_model import BlackScholesPricer, GirsanovSimulator, DeltaHedger


# TODO:
# - verifier que on prend bien les puts et calls
# - verifier que on prend bien les bonnes expirations
# - verifier que on prend bien les bonnes volatilites implicites
# - verifier que on prend bien les bonnes taux sans risque
# - verifier que on prend bien les bonnes dividendes
# - verifier que on prend bien les bonnes volatilites


# Constants
DAILY_STEPS = 252
NUM_PATHS = 1000
MIN_VOLUME = 10
PLOT_FIGSIZE = (10, 6)
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]


def setup_simulation(params: Dict[str, float], N: int = DAILY_STEPS, M: int = NUM_PATHS) -> tuple:
    """
    Setup simulation parameters and create necessary objects.

    Args:
        params (Dict[str, float]): Calibrated model parameters
        N (int): Number of time steps
        M (int): Number of paths

    Returns:
        tuple: (GirsanovSimulator, BlackScholesPricer) configured objects

    Raises:
        ValueError: If required parameters are missing
    """
    required_params = ["S0", "mu", "r", "sigma", "T", "q"]
    missing_params = [param for param in required_params if param not in params]
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

    simulator = GirsanovSimulator(
        S0=params["S0"],
        mu=params["mu"],
        r=params["r"],
        sigma=params["sigma"],
        N=N,
        T=params["T"],
        M=M,
    )

    # Create pricer
    pricer = BlackScholesPricer(
        S0=params["S0"],
        K=params["S0"],  # ATM option
        T=params["T"],
        sigma=params["sigma"],
        r=params["r"],
        q=params["q"],
    )

    return simulator, pricer


def plot_pnl_distribution(final_pnl: np.ndarray, ticker: str, expiry: str) -> None:
    """
    Plot the P&L distribution.

    Args:
        final_pnl (np.ndarray): Array of final P&L values
        ticker (str): Stock ticker
        expiry (str): Expiry date
    """
    try:
        plt.figure(figsize=PLOT_FIGSIZE)
        sns.histplot(final_pnl, bins=50, kde=True, stat="density")
        sns.kdeplot(final_pnl, color="red")
        plt.title(f"P&L Distribution for {ticker} - {expiry}")
        plt.xlabel("P&L")
        plt.ylabel("Density")
        plt.grid(True)

        filename = f"pnl_distribution_{ticker}_{expiry}.png"
        plt.savefig(filename)
        plt.close()
        print(f"P&L distribution plot saved as {filename}")

    except Exception as e:
        print(f"Error plotting P&L distribution: {str(e)}")


def print_hedging_results(final_pnl: np.ndarray) -> None:
    """
    Print hedging performance metrics.

    Args:
        final_pnl (np.ndarray): Array of final P&L values
    """
    try:
        mean_pnl = np.mean(final_pnl)
        std_pnl = np.std(final_pnl, ddof=1)
        var_95 = np.percentile(final_pnl, 5)

        print("\nDelta hedging results:")
        print(f"Mean P&L: {mean_pnl:.2f}")
        print(f"P&L Standard Deviation: {std_pnl:.2f}")
        print(f"95% VaR: {var_95:.2f}")
        print(f"Maximum loss: {np.min(final_pnl):.2f}")
        print(f"Maximum profit: {np.max(final_pnl):.2f}")

    except Exception as e:
        print(f"Error calculating hedging results: {str(e)}")


def get_user_input() -> tuple[str, str]:
    """
    Get and validate user input for ticker and expiry date.

    Returns:
        tuple: (ticker, selected_expiry)

    Raises:
        ValueError: If input validation fails
    """
    # Get ticker from user
    print("\nAvailable default tickers:", ", ".join(DEFAULT_TICKERS))
    ticker = input("Enter stock symbol (or press Enter for a list of defaults): ").strip().upper()

    if not ticker:
        print("\nSelect a ticker:")
        for i, default_ticker in enumerate(DEFAULT_TICKERS, 1):
            print(f"{i}. {default_ticker}")

        while True:
            try:
                choice = int(input("\nEnter number (1-3): "))
                if 1 <= choice <= len(DEFAULT_TICKERS):
                    ticker = DEFAULT_TICKERS[choice - 1]
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

    return ticker


def run_calibration(
    calibrator: ModelCalibrator,
) -> tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Run the calibration process.

    Args:
        calibrator (ModelCalibrator): Initialized calibrator object

    Returns:
        tuple: (calibrated_params, selected_expiry) or (None, None) if calibration fails
    """
    try:
        calibrator.fetch_market_data()

        if not calibrator.expiry_dates or len(calibrator.expiry_dates) == 0:
            raise ValueError("No expiry dates available for this ticker.")

        print("\nAvailable expiry dates:")
        for i, date in enumerate(calibrator.expiry_dates, 1):
            print(f"{i}. {date}")

        while True:
            try:
                choice = int(input("\nSelect an expiry date (number): "))
                if 1 <= choice <= len(calibrator.expiry_dates):
                    selected_expiry = calibrator.expiry_dates[choice - 1]
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

        print(f"\nPlotting volatility smile for {calibrator.ticker} - {selected_expiry}...")
        calibrator.plot_volatility_smile(selected_expiry)

        print(f"\nCalibrating model for {calibrator.ticker} - {selected_expiry}...")
        params = calibrator.calibrate_model(selected_expiry)

        if not params:
            raise ValueError("Calibration failed. Try another expiry date.")

        return params, selected_expiry

    except Exception as e:
        print(f"\nCalibration error: {str(e)}")
        return None, None


def run_simulation(params: Dict[str, float], ticker: str, expiry: str) -> NoReturn:
    """
    Run the hedging simulation with calibrated parameters.

    Args:
        params (Dict[str, float]): Calibrated model parameters
        ticker (str): Stock ticker
        expiry (str): Expiry date
    """
    try:
        print("\nSimulating price paths for delta hedging...")
        simulator, pricer = setup_simulation(params)
        paths = simulator.generate_paths()

        print("Performing delta hedging simulation...")
        hedger = DeltaHedger(pricer, paths, DAILY_STEPS, params["T"])
        pnl = hedger.hedge()

        final_pnl = pnl[-1, :]
        print_hedging_results(final_pnl)

        print("\nPlotting P&L distribution...")
        plot_pnl_distribution(final_pnl, ticker, expiry)

    except Exception as e:
        print(f"\nSimulation error: {str(e)}")


def main() -> int:
    """
    Main function for model calibration and hedging simulation.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Get user input and initialize calibrator
        ticker = get_user_input()
        calibrator = ModelCalibrator(ticker)

        # Run calibration
        params, selected_expiry = run_calibration(calibrator)
        if not params or not selected_expiry:
            return 1

        # Print calibrated parameters
        print("\nCalibrated parameters:")
        print(f"Spot price (S0): {params['S0']:.2f}")
        print(f"Volatility (sigma): {params['sigma']:.2%}")
        print(f"Risk-free rate (r): {params['r']:.2%}")
        print(f"Dividend yield (q): {params['q']:.2%}")
        print(f"Time to maturity (T): {params['T']:.4f} years")

        # Run simulation
        run_simulation(params, ticker, selected_expiry)
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
