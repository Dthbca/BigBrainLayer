from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(os.environ.get("NONLAMINAR_ROOT", r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging"))
RES=ROOT/"results"; OUT=ROOT/"report"/"figure_draft"; OUT.mkdir(parents=True,exist_ok=True)
P={"Subclass":RES/"subclass_main_20260821"/"ratio_none_subclass",
   "Cluster":RES/"cluster_secondary_20260821"/"ratio_none_cluster"}
T=pd.read_csv(RES/"hansen_style_enigma_20260821"/"hansen_style_total_models.csv")
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"svg.fonttype":"none",
 "pdf.fonttype":42,"font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,
 "legend.frameon":False,"figure.facecolor":"white"})
names={"22q":"22q11.2 deletion","adhd":"ADHD","asd":"ASD","epilepsy_gge":"Generalized epilepsy",
 "epilepsy_rtle":"Right TLE","epilepsy_ltle":"Left TLE","depression":"Depression","ocd":"OCD",
 "schizophrenia":"Schizophrenia","bipolar":"Bipolar disorder","obesity":"Obesity","schizotypy":"Schizotypy","parkinson":"Parkinson"}
order=list(names); labels=[names[x] for x in order]; palette={"Subclass":"#527EB7","Cluster":"#CB624A"}

fig=plt.figure(figsize=(7.2,7.3),constrained_layout=True)
gs=fig.add_gridspec(2,2,height_ratios=[1.05,1.1],width_ratios=[1,1])

# a adjusted-R2 total spatial inference
ax=fig.add_subplot(gs[0,0]); yy=np.arange(len(order)); src=[]
for lev,dy in [("Subclass",-.15),("Cluster",.15)]:
    z=T[T.level.eq(lev.lower())].set_index("outcome").loc[order]
    ax.scatter(z.adjusted_r2,yy+dy,s=25,color=palette[lev],label=lev,zorder=3)
    for i,(v,q) in enumerate(zip(z.adjusted_r2,z.spin_q_bh_13_disorders)):
        if q<.05: ax.scatter(v,i+dy,s=48,facecolors="none",edgecolors="#222",lw=.7,zorder=4)
    for out,row in z.iterrows(): src.append({"level":lev,"outcome":out,**row.to_dict()})
ax.axvline(0,color="#555",lw=.7); ax.grid(axis="x",color="#e5e5e5",lw=.45); ax.set_axisbelow(True)
ax.set_yticks(yy,labels); ax.invert_yaxis(); ax.set_xlabel("Adjusted $R^2$")
ax.set_title("a  Spatial total-model inference",loc="left",fontweight="bold",fontsize=8)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2,fontsize=5.8)
ax.text(.5,-.23,"Outline: spin q < 0.05 across 13 disorders",ha="center",transform=ax.transAxes,fontsize=5.2,color="#555")

# b OOF generalization, explicitly separated from in-sample inference
ax=fig.add_subplot(gs[0,1]); perf=[]
for lev,p in P.items():
    z=pd.read_csv(p/"performance.csv"); z=z[z.phenotype.eq("enigma")].copy(); z["level"]=lev; perf.append(z)
perf=pd.concat(perf); piv=perf.pivot(index="outcome",columns="level",values="oof_r2").reindex(index=order,columns=["Subclass","Cluster"])
sns.heatmap(piv,cmap="RdBu_r",center=0,vmin=-.8,vmax=.8,linewidths=.3,linecolor="white",
            cbar_kws={"label":"Out-of-fold $R^2$","shrink":.72},ax=ax)
ax.set_yticklabels(labels,rotation=0,fontsize=5.5); ax.set_xticklabels(["Subclass","Cluster"],rotation=0)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("b  Lobe-wise generalization",loc="left",fontweight="bold",fontsize=8)

# c subclass dominance only for total-model-significant disorders
ax=fig.add_subplot(gs[1,0]); d=pd.read_csv(P["Subclass"]/"dominance.csv"); d=d[d.phenotype.eq("enigma")]
sig=T[(T.level.eq("subclass"))&(T.spin_q_bh_13_disorders<.05)].outcome.tolist()
tops=[]
for out in sig: tops.extend(d[d.outcome.eq(out)].nlargest(2,"relative_dominance").feature.tolist())
features=list(dict.fromkeys(tops)); mat=pd.DataFrame(index=features)
for out in sig: mat[names[out]]=d[d.outcome.eq(out)].set_index("feature").relative_dominance.reindex(features)*100
mat=mat.loc[mat.max(axis=1).sort_values(ascending=False).index]
sns.heatmap(mat,cmap="YlOrBr",vmin=0,linewidths=.25,linecolor="white",
            cbar_kws={"label":"Relative dominance (%)","shrink":.72},ax=ax)
ax.set_title("c  Subclass dominance after spatial screening",loc="left",fontweight="bold",fontsize=8)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticklabels(ax.get_xticklabels(),rotation=38,ha="right",fontsize=5.4)
ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=5.3)

# d cluster leading dominance, shown as exploratory concentration only
ax=fig.add_subplot(gs[1,1]); dc=pd.read_csv(P["Cluster"]/"dominance.csv"); dc=dc[dc.phenotype.eq("enigma")]
sigc=T[(T.level.eq("cluster"))&(T.spin_q_bh_13_disorders<.05)].outcome.tolist(); rows=[]
for out in sigc:
    z=dc[dc.outcome.eq(out)].sort_values("relative_dominance",ascending=False).iloc[0]
    rows.append((out,z.feature,max(0,100*float(z.relative_dominance))))
topc=pd.DataFrame(rows,columns=["outcome","feature","relative_dominance_pct"]).sort_values("relative_dominance_pct")
y=np.arange(len(topc)); ax.barh(y,topc.relative_dominance_pct,color="#E9A23B",height=.62)
ax.set_yticks(y,[names[x] for x in topc.outcome]); ax.set_xlim(0,max(100,topc.relative_dominance_pct.max()+18))
ax.set_xlabel("Leading cluster dominance (%)"); ax.set_title("d  Cluster-level exploratory concentration",loc="left",fontweight="bold",fontsize=8)
for i,row in topc.reset_index(drop=True).iterrows(): ax.text(row.relative_dominance_pct+1.2,i,f"{row.feature}  {row.relative_dominance_pct:.1f}%",va="center",fontsize=4.8)
ax.grid(axis="x",color="#e5e5e5",lw=.45); ax.set_axisbelow(True)

base=OUT/"nonlaminar_hansen_style_enigma_draft"
for ext,kw in [("png",{"dpi":300}),("svg",{}),("pdf",{}),("tiff",{"dpi":600})]: fig.savefig(f"{base}.{ext}",bbox_inches="tight",facecolor="white",**kw)
plt.close(fig)
pd.DataFrame(src).to_csv(OUT/"hansen_enigma_total_source.csv",index=False); piv.to_csv(OUT/"hansen_enigma_oof_source.csv")
mat.to_csv(OUT/"hansen_enigma_subclass_dominance_source.csv"); topc.to_csv(OUT/"hansen_enigma_cluster_top_source.csv",index=False)
caption=("Figure Z | Cell-composition models of ENIGMA cortical abnormalities following the neurotransmitter-mapping evidence hierarchy. "
"a, Adjusted R² from standardized ordinary linear models; outlined symbols pass Alexander–Bloch spin testing and Benjamini–Hochberg correction across 13 disorders. "
"b, Lobe-wise out-of-fold Ridge R², reported separately from spatial total-model inference. c, Relative subclass dominance only for disorders passing the spatial total-model screen. "
"d, Leading cluster contribution among spatially significant cluster models, shown as an exploratory concentration diagnostic. Dominance partitions shared in-sample fit and is not an independent or causal effect.")
(OUT/"hansen_enigma_caption.txt").write_text(caption,encoding="utf-8")
(OUT/"hansen_enigma_qa.json").write_text(json.dumps({"formats":{x:Path(f"{base}.{x}").stat().st_size for x in ["png","svg","pdf","tiff"]},"subclass_significant":sig,"cluster_significant":sigc},indent=2),encoding="utf-8")
print(base)
