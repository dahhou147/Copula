import aleatory as alea
import scipy.stats as ss
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
""" implimenter un model de pricing des options americaines et européene en utilsant cette et par compréhension de chaque fonction ce quelle fait """
""" dX ="""
 


def GBM(dt,r,sigma,x_0,size):
    L = [x_0]
    s=x_0
    k=0
    while k<size:
        dw = ss.norm.rvs(scale = np.sqrt(dt))
        s= s*(1+r*dt+sigma*dw)
        L.append(s)
        k+=1
    return L 

L= np.array([GBM(0.02,0.01,0.03,25,1000) for k in range(200)])
df= pd.DataFrame(L)
plt.plot(df.T)
plt.show()