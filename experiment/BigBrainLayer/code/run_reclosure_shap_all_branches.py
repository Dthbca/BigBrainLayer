"""Five-fold OOF Ridge-SHAP for the eight mask-aware reclosed pipelines."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from HomoloMap.datasets.layers import (LAYER_KEYS, LAYER_LABELS,
    fetch_bigbrain_layer_thickness, fetch_laminar_mask, load_layer_counts,
    normalize_layer_composition, relabel_layer_counts)
from HomoloMap.transforms.layers import make_layer_subcompositions

CLASS_DEF={"Excitatory":["L2/3 IT","L4 IT","L5 IT","L6 IT","L6 IT Car3","L5 ET","L5/6 NP","L6 CT","L6b"],
"Inhibitory":["Lamp5_Lhx6","Lamp5","Pax6","Sncg","Vip","Sst","Pvalb","Chandelier"],
"Non-neuron":["Astro","Oligo","OPC","Micro-PVM","Endo","VLMC"]}
ALIASES=dict(zip(LAYER_KEYS,LAYER_LABELS))

def reclose(mapped,norm,present):
    out={k:v.copy().astype(float) for k,v in mapped.items()}
    if norm=="within_layer":
        for layer in LAYER_KEYS:
            keep=present.loc[ALIASES[layer]].reindex(out[layer].columns).fillna(False).astype(bool)
            cols=list(keep[keep].index); out[layer].loc[:,~keep]=0.0
            den=out[layer][cols].sum(axis=1); good=np.isfinite(den)&(den>0)
            out[layer].loc[good,cols]=out[layer].loc[good,cols].div(den[good],axis=0)
            out[layer].loc[~good,cols]=0.0
    else:
        for ctype in out["l1"].columns:
            active=[l for l in LAYER_KEYS if bool(present.loc[ALIASES[l],ctype])]
            for l in set(LAYER_KEYS)-set(active): out[l].loc[:,ctype]=0.0
            m=pd.concat({l:out[l][ctype] for l in active},axis=1); den=m.sum(axis=1)
            good=np.isfinite(den)&(den>0)
            for l in active:
                out[l].loc[good,ctype]=m.loc[good,l]/den[good]; out[l].loc[~good,ctype]=0.0
    return out

def bname(norm,clr,relative):
    return f"{norm}__clr_{str(clr).lower()}__thickness_{'relative' if relative else 'absolute'}"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--package",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); sys.path.insert(0,str(a.package)); a.output.mkdir(parents=True,exist_ok=True)
    raw,mapping=load_layer_counts(a.data,source_atlas="D99",mapping_column="subclass",unmapped="drop",return_mapping=True)
    class_map={c:g for g,cs in CLASS_DEF.items() for c in cs}; alphas=np.logspace(-3,3,25)
    all_layer=[]; all_ctype=[]; all_class=[]; all_summary=[]; thickness={}
    for norm,mode in (("within_layer","within_layer"),("within_region_cross_layer","within_region")):
        normalized=normalize_layer_composition(raw,mode=mode,zero_policy="zero")
        mapped=relabel_layer_counts(normalized,"D99","BN",method="mean",cross_species=True,unknown_labels="drop")
        regions,ctypes=mapped["l1"].index,mapped["l1"].columns
        present,_=fetch_laminar_mask("external",ctypes,data_dir=a.data); mapped=reclose(mapped,norm,present)
        for relative in (False,True):
            key="relative" if relative else "absolute"
            if key not in thickness: thickness[key]=fetch_bigbrain_layer_thickness("BN",a.data,relative=relative,regions=regions)
            for clr in (False,True):
                branch=bname(norm,clr,relative)
                features,_=make_layer_subcompositions(mapped,present,transform="clr" if clr else "none",zero_method="multiplicative",invalid_rows="drop")
                layer_rows=[]; perf=[]
                for layer,label in zip(LAYER_KEYS,LAYER_LABELS):
                    X,y=features[layer].align(thickness[key][label],join="inner",axis=0)
                    valid=np.isfinite(X).all(axis=1)&np.isfinite(y); X,y=X.loc[valid],y.loc[valid]
                    vals=pd.DataFrame(0.0,index=X.index,columns=X.columns); pred=pd.Series(index=X.index,dtype=float); fold_a=[]
                    for train,test in KFold(5,shuffle=True,random_state=42).split(X):
                        sc=StandardScaler().fit(X.iloc[train]); xtr,xte=sc.transform(X.iloc[train]),sc.transform(X.iloc[test])
                        model=RidgeCV(alphas=alphas).fit(xtr,y.iloc[train]); ex=shap.LinearExplainer(model,xtr)
                        vals.iloc[test]=np.asarray(ex(xte).values); pred.iloc[test]=model.predict(xte); fold_a.append(float(model.alpha_))
                    row={"branch":branch,"normalization":norm,"clr":clr,"thickness_relative":relative,
                         "layer":layer,"layer_label":label,"n_roi":len(X),"n_features":X.shape[1],
                         "oof_r2":float(r2_score(y,pred)),"oof_pearson_r":float(np.corrcoef(y,pred)[0,1]),
                         "median_alpha":float(np.median(fold_a))}
                    all_layer.append(row); perf.append(row)
                    means=vals.abs().mean(axis=0)
                    layer_rows.extend({"layer":layer,"ctype":c,"mean_abs_shap":float(v)} for c,v in means.items())
                cdf=pd.DataFrame(layer_rows).groupby("ctype",as_index=False).mean(numeric_only=True).sort_values("mean_abs_shap",ascending=False)
                cdf["relative_contribution"]=cdf.mean_abs_shap/cdf.mean_abs_shap.sum(); cdf["class"]=cdf.ctype.map(class_map).fillna("Other")
                all_ctype.extend({"branch":branch,**r} for r in cdf.to_dict("records"))
                cls=cdf.groupby("class",as_index=False).agg(relative_contribution=("relative_contribution","sum"),n_ctype=("ctype","size"))
                all_class.extend({"branch":branch,**r} for r in cls.to_dict("records"))
                rel=cdf.relative_contribution.to_numpy(); top=cdf.head(5)
                all_summary.append({"branch":branch,"normalization":norm,"clr":clr,"thickness_relative":relative,
                    "mean_oof_r2":float(np.mean([r["oof_r2"] for r in perf])),"median_oof_r2":float(np.median([r["oof_r2"] for r in perf])),
                    "mean_oof_pearson_r":float(np.mean([r["oof_pearson_r"] for r in perf])),"top1_shap_share":float(rel.max()),
                    "top5_shap_share":float(top.relative_contribution.sum()),"shap_entropy":float(-(rel*np.log(rel+1e-15)).sum()),
                    "effective_n":float(np.exp(-(rel*np.log(rel+1e-15)).sum())),"top_ctype":str(cdf.iloc[0].ctype),
                    "top5_ctypes":"; ".join(top.ctype.tolist())})
    pd.DataFrame(all_layer).to_csv(a.output/"shap_all_branch_layer_performance.csv",index=False)
    pd.DataFrame(all_ctype).to_csv(a.output/"shap_all_branch_ctype.csv",index=False)
    pd.DataFrame(all_class).to_csv(a.output/"shap_all_branch_class.csv",index=False)
    summary=pd.DataFrame(all_summary).sort_values("mean_oof_r2",ascending=False); summary.to_csv(a.output/"shap_all_branch_summary.csv",index=False)
    (a.output/"shap_all_branch_metadata.json").write_text(json.dumps({"closure":"post-relabel mask-aware","model":"5-fold OOF RidgeCV","explainer":"shap.LinearExplainer","random_state":42,"mapping_unresolved":int(mapping.get("n_unresolved_types",0))},indent=2),encoding="utf-8")
    print(json.dumps({"status":"PASS","n_branches":len(summary)},ensure_ascii=False))

if __name__=="__main__": main()
