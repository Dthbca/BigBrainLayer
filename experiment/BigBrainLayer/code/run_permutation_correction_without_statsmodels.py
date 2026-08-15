"""Provide an equivalent BH implementation, then run the correction audit."""
import sys, types, runpy
import numpy as np

def multipletests(pvals, alpha=.05, method='fdr_bh'):
    if method != 'fdr_bh': raise ValueError('only fdr_bh is supported')
    p=np.asarray(pvals,float); n=len(p); order=np.argsort(p); ranked=p[order]
    adj=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1]
    adj=np.minimum(adj,1.0); q=np.empty(n); q[order]=adj
    return q<alpha,q,alpha/n,1-(1-alpha)**(1/n)

stats=types.ModuleType('statsmodels'); stats_stats=types.ModuleType('statsmodels.stats')
multi=types.ModuleType('statsmodels.stats.multitest'); multi.multipletests=multipletests
sys.modules['statsmodels']=stats; sys.modules['statsmodels.stats']=stats_stats
sys.modules['statsmodels.stats.multitest']=multi
runpy.run_path(r'D:\HomoloMap\correct_layer_permutation_results.py',run_name='__main__')
