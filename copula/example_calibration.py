from calibration import ModelCalibrator


def main():
    """
    Exemple d'utilisation du calibrateur de modèle pour un ticker donné
    """
    # Demande du ticker à l'utilisateur
    ticker = input("Entrez le symbole de l'action (ex: AAPL, MSFT, GOOGL): ").upper()

    # Initialisation du calibrateur
    print(f"Calibration du modèle pour {ticker}...")
    calibrator = ModelCalibrator(ticker)

    # Récupération des données de marché
    print("Récupération des données de marché...")
    calibrator.fetch_market_data()

    # Vérification des dates d'expiration disponibles
    if not calibrator.expiry_dates or len(calibrator.expiry_dates) == 0:
        print("Aucune date d'expiration disponible pour ce ticker.")
        return

    # Affichage des dates d'expiration disponibles
    print("\nDates d'expiration disponibles:")
    for i, date in enumerate(calibrator.expiry_dates):
        print(f"{i+1}. {date}")

    # Sélection de la date d'expiration
    while True:
        try:
            choice = int(input("\nSélectionnez une date d'expiration (numéro): "))
            if 1 <= choice <= len(calibrator.expiry_dates):
                selected_expiry = calibrator.expiry_dates[choice - 1]
                break
            else:
                print("Choix invalide. Veuillez réessayer.")
        except ValueError:
            print("Veuillez entrer un nombre.")

    # Sélection du type d'option
    while True:
        option_type = input("\nType d'option (call/put): ").lower()
        if option_type in ["call", "put"]:
            break
        else:
            print("Type d'option invalide. Veuillez entrer 'call' ou 'put'.")

    # Traçage du smile de volatilité
    print(f"\nTraçage du smile de volatilité pour {ticker} - {selected_expiry} ({option_type})...")
    calibrator.plot_volatility_smile(selected_expiry, option_type)

    # Calibration du modèle
    print(f"\nCalibration du modèle pour {ticker} - {selected_expiry} ({option_type})...")
    params = calibrator.calibrate_model(selected_expiry, option_type)

    if params:
        print("\nParamètres calibrés:")
        print(f"Prix spot (S0): {params['S0']:.2f}")
        print(f"Volatilité (sigma): {params['sigma']:.2%}")
        print(f"Taux sans risque (r): {params['r']:.2%}")
        print(f"Taux de dividende (q): {params['q']:.2%}")
        print(f"Temps jusqu'à l'échéance (T): {params['T']:.4f} années")
    else:
        print(
            "\nLa calibration a échoué. Essayez avec une autre date d'expiration ou un autre type d'option."
        )


if __name__ == "__main__":
    main()
