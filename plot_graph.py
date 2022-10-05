import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def rollavg_cumsum_edges(a,n):
    'numpy.cumsum, edge handling'
    assert n%2==1
    N = len(a)
    cumsum_vec = np.cumsum(np.insert(np.pad(a,(n-1,n-1),'constant'), 0, 0)) 
    d = np.hstack((np.arange(n//2+1,n),np.ones(N-n)*n,np.arange(n,n//2,-1)))  
    return (cumsum_vec[n+n//2:-n//2+1] - cumsum_vec[n//2:-n-n//2]) / d

def rollavg_pandas(a,n):
    'Pandas rolling average'
    return pd.DataFrame(a).rolling(n, center=True, min_periods=1).mean().to_numpy()

def moving_mean_plot(mean_list:list,passos,avg_window=50,algo=0,figsize=(12,6)):
    if algo == 0:
        plot_list = rollavg_cumsum_edges(mean_list,avg_window)
    else:
        plot_list = rollavg_pandas(mean_list,avg_window)
    plt.figure(figsize=figsize)
    plt.plot(np.arange(passos),plot_list)
    plt.xlabel('Passos')
    plt.ylabel(f'last {avg_window} rewards')
    plt.show()