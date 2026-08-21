from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(os.environ.get("NONLAMINAR_ROOT", r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging"))
OUT=ROOT/"report"/"figure_draft"; OUT.mkdir(parents=True,exist_ok=True)
PATHS={"Subclass":ROOT/"results"/"subclass_main_20260821"/"ratio_none_subclass",
       "Cluster":ROOT/"results"/"cluster_secondary_20260821"/"ratio_none_cluster"}
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.spines.top":False,
 "axes.spines.right":False,"axes.linewidth":.7,"legend.frameon":False,"figure.facecolor":"white"})

focus=[("meg","alpha","Alpha"),("meg","theta","Theta"),("enigma","adhd","ADHD"),
       ("enigma","asd","ASD"),("enigma","epilepsy_rtle","Right TLE"),("enigma","parkinson","Parkinson")]
labels=[x[2] for x in focus]
spin={k:pd.read_csv(p/"spin.csv") for k,p in PATHS.items()}
dom={k:pd.read_csv(p/"dominance.csv") for k,p in PATHS.items()}
total={k:pd.read_csv(p/"total_models.csv") for k,p in PATHS.items()}

def bh(p):
    p=np.asarray(p,float); o=np.argsort(p); q=np.empty(len(p));
    q[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    return np.minimum(q,1)

fig=plt.figure(figsize=(7.2,6.7),constrained_layout=True)
gs=fig.add_gridspec(2,2,height_ratios=[1.25,1],width_ratios=[1.45,.75])

# a: subclass univariate spin tests, all 23 mapped subclasses.
ax=fig.add_subplot(gs[0,0]); s=spin["Subclass"]
r=pd.DataFrame(index=sorted(s.feature.unique()),columns=labels,dtype=float)
q=r.copy()
for ph,out,lab in focus:
    z=s[(s.phenotype==ph)&(s.outcome==out)].set_index("feature")
    r[lab]=z.pearson_r; q[lab]=z.spin_q_bh
r=r.loc[r.abs().max(axis=1).sort_values(ascending=False).index]
q=q.reindex(r.index)
sns.heatmap(r,cmap="RdBu_r",center=0,vmin=-.9,vmax=.9,linewidths=.25,linecolor="white",
            cbar_kws={"label":"Pearson r","shrink":.72},ax=ax)
for iy,f in enumerate(r.index):
    for ix,lab in enumerate(r.columns):
        if np.isfinite(q.loc[f,lab]) and q.loc[f,lab]<.05:
            ax.plot(ix+.5,iy+.5,"o",ms=2.1,mfc="#1f1f1f",mec="none")
ax.set_title("a  Subclass spatial associations  (dots: spin q < 0.05)",loc="left",fontweight="bold",fontsize=8)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticklabels(ax.get_xticklabels(),rotation=35,ha="right")
ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=5.3)

# b: total model OOF and family-wise spatial significance.
ax=fig.add_subplot(gs[0,1]); rows=[]
for level in ["Subclass","Cluster"]:
    perf=pd.read_csv(PATHS[level]/"performance.csv")
    for ph,out,lab in focus:
        rr=perf[(perf.phenotype==ph)&(perf.outcome==out)].iloc[0]
        tt=total[level][total[level].phenotype.eq(ph)].copy(); tt["q_family"]=bh(tt.spin_p)
        rows.append((level,lab,float(rr.oof_r2),float(tt.loc[tt.outcome.eq(out),"q_family"].iloc[0])))
e=pd.DataFrame(rows,columns=["level","outcome","oof_r2","total_q"])
pal={"Subclass":"#527EB7","Cluster":"#CB624A"}; y=np.arange(len(labels))
for level,dy in [("Subclass",-.11),("Cluster",.11)]:
    z=e[e.level.eq(level)].set_index("outcome").loc[labels]
    ax.scatter(z.oof_r2,y+dy,s=27,color=pal[level],label=level,zorder=3)
    for yi,(v,qq) in enumerate(zip(z.oof_r2,z.total_q)):
        if qq<.05: ax.scatter(v,yi+dy,s=53,facecolors="none",edgecolors="#222",lw=.7,zorder=4)
ax.axvline(0,color="#555",lw=.7); ax.grid(axis="x",color="#e5e5e5",lw=.45)
ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel("Out-of-fold $R^2$")
ax.set_title("b  Total-model evidence",loc="left",fontweight="bold",fontsize=8)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2,fontsize=5.8)
ax.text(.5,-.25,"Outline: total-model spin q < 0.05",ha="center",transform=ax.transAxes,fontsize=5.2,color="#555")

# c: subclass dominance for a non-redundant union of top features.
ax=fig.add_subplot(gs[1,0]); d=dom["Subclass"]
tops=[]
for ph,out,lab in focus:
    z=d[(d.phenotype==ph)&(d.outcome==out)].nlargest(2,"relative_dominance")
    tops.extend(z.feature.tolist())
features=list(dict.fromkeys(tops))
mat=pd.DataFrame(index=features)
for ph,out,lab in focus:
    mat[lab]=d[(d.phenotype==ph)&(d.outcome==out)].set_index("feature").relative_dominance.reindex(features)*100
mat=mat.loc[mat.max(axis=1).sort_values(ascending=False).index]
sns.heatmap(mat,cmap="YlOrBr",vmin=0,linewidths=.25,linecolor="white",
            cbar_kws={"label":"Relative dominance (%)","shrink":.72},ax=ax)
ax.set_title("c  Subclass allocation of model $R^2$",loc="left",fontweight="bold",fontsize=8)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticklabels(ax.get_xticklabels(),rotation=35,ha="right")
ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=5.3)

# d: cluster concentration warns against reading fine clusters as independent effects.
ax=fig.add_subplot(gs[1,1]); rows=[]
for ph,out,lab in focus:
    z=dom["Cluster"][(dom["Cluster"].phenotype==ph)&(dom["Cluster"].outcome==out)].nlargest(1,"relative_dominance").iloc[0]
    rows.append((lab,z.feature,100*float(z.relative_dominance)))
topc=pd.DataFrame(rows,columns=["outcome","feature","relative_dominance_pct"])
yy=np.arange(len(topc)); ax.barh(yy,topc.relative_dominance_pct,color="#E9A23B",height=.58)
ax.set_yticks(yy,topc.outcome); ax.invert_yaxis(); ax.set_xlim(0,100); ax.set_xlabel("Leading cluster dominance (%)")
ax.set_title("d  Cluster concentration",loc="left",fontweight="bold",fontsize=8)
for yi,row in topc.iterrows():
    ax.text(min(row.relative_dominance_pct+2,94),yi,f"{row.feature}  {row.relative_dominance_pct:.1f}%",va="center",fontsize=5.2)
ax.grid(axis="x",color="#e5e5e5",lw=.45); ax.set_axisbelow(True)

base=OUT/"nonlaminar_ratio_spin_dominance_draft"
for ext,kw in [("png",{"dpi":300}),("svg",{}),("pdf",{}),("tiff",{"dpi":600})]:
    fig.savefig(f"{base}.{ext}",bbox_inches="tight",facecolor="white",**kw)
plt.close(fig)
r.to_csv(OUT/"spin_subclass_r_source.csv"); q.to_csv(OUT/"spin_subclass_q_source.csv")
e.to_csv(OUT/"total_model_evidence_source.csv",index=False); mat.to_csv(OUT/"dominance_subclass_source.csv")
topc.to_csv(OUT/"dominance_cluster_top_source.csv",index=False)
caption=("Figure Y | Spatial inference and dominance analysis for ratio-based cell-composition models. "
"a, Pearson correlations between mapped subclass proportions and selected MEG or ENIGMA maps; dots denote "
"Alexander–Bloch spin-test results passing Benjamini–Hochberg correction within each outcome. b, Lobe-wise "
"out-of-fold R² for subclass and cluster models; outlined symbols denote total models passing spatial spin testing "
"after correction within the MEG or ENIGMA family. c, Relative dominance allocation of in-sample model R² among "
"subclass predictors. d, The leading cluster contribution for each focused outcome, illustrating concentration at "
"fine resolution. Dominance partitions shared model fit and does not identify independent or causal effects.")
(OUT/"spin_dominance_caption.txt").write_text(caption,encoding="utf-8")
(OUT/"spin_dominance_qa.json").write_text(json.dumps({"formats":{x:Path(f"{base}.{x}").stat().st_size for x in ["png","svg","pdf","tiff"]},"n_spins":1000,"primary":"ratio","panels":4},indent=2),encoding="utf-8")
print(base)
