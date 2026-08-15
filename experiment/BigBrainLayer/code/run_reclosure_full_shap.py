"""Execute reclosure SHAP with closure over the full target composition."""
from pathlib import Path

runner=Path("/share/user_data/dthbca/public/experiment/BigBrainLayer/run_reclosure_shap_all_branches.py")
source=runner.read_text(encoding="utf-8")
start=source.index("def reclose("); end=source.index("def bname",start)
replacement='''def reclose(mapped,norm,present):
    out={k:v.copy().astype(float) for k,v in mapped.items()}
    if norm=="within_layer":
        for layer in LAYER_KEYS:
            den=out[layer].sum(axis=1); good=np.isfinite(den)&(den>0)
            out[layer].loc[good]=out[layer].loc[good].div(den[good],axis=0)
            out[layer].loc[~good]=0.0
    else:
        for ctype in out["l1"].columns:
            m=pd.concat({l:out[l][ctype] for l in LAYER_KEYS},axis=1); den=m.sum(axis=1)
            good=np.isfinite(den)&(den>0)
            for l in LAYER_KEYS:
                out[l].loc[good,ctype]=m.loc[good,l]/den[good]; out[l].loc[~good,ctype]=0.0
    return out

'''
exec(compile(source[:start]+replacement+source[end:],str(runner),"exec"),{"__name__":"__main__","__file__":str(runner)})
