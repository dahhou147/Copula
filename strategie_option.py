import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as ss
import scipy.optimize as so


def capital_protected_strategie(portfolio, call_option, strike_price, rf, maturity):
    """ the portfolio is composed of indexs and we have a call option on the index and olbligation with strike price, rf is the risk free rate"""
    N=1000
    t= np.linspace(0,maturity,N)
    portfolio_value = strike_price*np.exp(-rf*(maturity-t)) + np.maximum(portfolio-strike_price,0)*np.exp(-rf*(maturity-t)) - call_option
    return t, portfolio_value


def bull_spread(St, K1, K2, rf, maturity):
    """ the portfolio is composed of indexs and we have a call option on the index and olbligation with strike price, rf is the risk free rate"""
    N=1000
    t= np.linspace(0,maturity,N)
    portfolio_value = np.maximum(St-K1,0)*np.exp(-rf*(maturity-t)) - np.maximum(St-K2,0)*np.exp(-rf*(maturity-t))
    return t, portfolio_value

def butterfly_spread(S0, K1, K2, K3, rf, maturity):
    """ the portfolio is composed of indexs and we have a call option on the index and olbligation with strike price, rf is the risk free rate"""
    N=1000
    t= np.linspace(0,maturity,N)
    portfolio_value = -np.maximum(S0-K1,0)*np.exp(-rf*(maturity-t)) + 2*np.maximum(S0-K2,0)*np.exp(-rf*(maturity-t)) - np.maximum(S0-K3,0)*np.exp(-rf*(maturity-t))
    return t, portfolio_value


def straddle(S0, K, call_option, put_option, rf, maturity):
    """ the portfolio is composed of indexs and we have a call option on the index and olbligation with strike price, rf is the risk free rate"""
    N=1000
    t= np.linspace(0,maturity,N)
    portfolio_value = np.maximum(S0-K,0)*np.exp(-rf*(maturity-t)) - call_option - put_option
    return t, portfolio_value

def box_spread(S0, K1, K2, call_option, put_option, rf, maturity):
    """ the portfolio is composed of indexs and we have a call option on the index and olbligation with strike price, rf is the risk free rate"""
    N=1000
    t= np.linspace(0,maturity,N)
    portfolio_value = np.maximum(S0-K1,0)*np.exp(-rf*(maturity-t)) - np.maximum(S0-K2,0)*np.exp(-rf*(maturity-t)) - call_option + put_option
    return t, portfolio_value


def another_strategy():
    pass
