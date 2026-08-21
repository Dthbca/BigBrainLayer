from pathlib import Path
import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT=Path(os.environ.get("NONLAMINAR_ROOT", r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging"))
OUT=ROOT/"results"/"hansen_style_meg_20260821"; OUT.mkdir(parents=True,exist_ok=True)
PATHS={"subclass":ROOT/"results"/"subclass_main_20260821"/"ratio_none_subclass",
       "cluster":ROOT/"results"/"cluster_secondary_20260821"/"ratio_none_cluster"}
N_SPINS=1000; SEED=42

def fit_r2_adj(X,y):
    good=np.isfinite(y); X=X[good]; y=y[good]
    X=StandardScaler().fit_transform(X); y=(y-y.mean())/y.std(ddof=0)
    A=np.column_stack([np.ones(len(X)),X]); coef=np.linalg.lstsq(A,y,rcond=None)[0]
    pred=A@coef; ss=np.sum((y-y.mean())**2); r2=1-np.sum((y-pred)**2)/ss
    n,p=X.shape; adj=1-(1-r2)*(n-1)/(n-p-1) if n>p+1 else np.nan
    return float(r2),float(adj),int(n),int(p)

def main():
    from HomoloMap.stats.nulls import SpinTest
    spinner=SpinTest(atlas="BN",n_spins=N_SPINS,method="Alexander-Bloch",seed=SEED)
    rows=[]
    for level,path in PATHS.items():
        Xdf=pd.read_csv(path/"X_bn.csv",index_col=0); Xdf.index=Xdf.index.astype(int)
        Y=pd.read_csv(path/"Y_meg_bn.csv",index_col=0); Y.index=Y.index.astype(int)
        common=Xdf.index.intersection(Y.index); X=Xdf.loc[common].to_numpy(float); Y=Y.loc[common]
        for outcome in Y.columns:
            y=Y[outcome].to_numpy(float); r2,adj,n,p=fit_r2_adj(X,y); null=[]
            for k in range(spinner.spins.shape[1]):
                yp=y[spinner.spins[:,k]]
                _,a,_,_=fit_r2_adj(X,yp); null.append(a)
            null=np.asarray(null,float)
            spin_p=(1+np.sum(null>=adj))/(len(null)+1)
            rows.append({"level":level,"outcome":outcome,"n":n,"n_features":p,
                         "r2":r2,"adjusted_r2":adj,"spin_p":spin_p,"n_spins":len(null),
                         "null_mean":float(np.nanmean(null)),"null_sd":float(np.nanstd(null))})
    res=pd.DataFrame(rows)
    for level in res.level.unique():
        m=res.level.eq(level); res.loc[m,"spin_q_bh_six_bands"]=multipletests(res.loc[m,"spin_p"],method="fdr_bh")[1]
    res.to_csv(OUT/"hansen_style_total_models.csv",index=False)
    audit={"model":"ordinary least squares with intercept","predictors":"z-scored mapped/reclosed ratio features",
           "outcome":"z-scored MEG band map","statistic":"adjusted R2","null":"Alexander-Bloch response-map rotation and complete model refit",
           "n_spins":N_SPINS,"seed":SEED,"multiple_testing":"Benjamini-Hochberg across six bands separately by resolution",
           "dominance_policy":"show dominance only where spin_q_bh_six_bands < 0.05",
           "generalization":"reported separately using existing lobe-wise OOF Ridge and held-out linear SHAP"}
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(res.to_string(index=False))

if __name__=="__main__": main()
