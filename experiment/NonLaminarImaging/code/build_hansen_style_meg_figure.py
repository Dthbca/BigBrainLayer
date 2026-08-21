from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

ROOT=Path(os.environ.get("NONLAMINAR_ROOT",r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging"))
OUT=ROOT/"report"/"figure_draft"; OUT.mkdir(parents=True,exist_ok=True)
RES=ROOT/"results"; H=RES/"hansen_style_meg_20260821"
PATHS={"Subclass":RES/"subclass_main_20260821"/"ratio_none_subclass",
       "Cluster":RES/"cluster_secondary_20260821"/"ratio_none_cluster"}
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"svg.fonttype":"none","pdf.fonttype":42,
 "font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,"legend.frameon":False,"figure.facecolor":"white"})
bands=["delta","theta","alpha","beta","gamma1","gamma2"]
names={"delta":"Delta","theta":"Theta","alpha":"Alpha","beta":"Beta","gamma1":"Gamma 1","gamma2":"Gamma 2"}
colors={"Subclass":"#527EB7","Cluster":"#CB624A"}
tot=pd.read_csv(H/"hansen_style_total_models.csv")

fig=plt.figure(figsize=(7.2,6.4),constrained_layout=True)
gs=fig.add_gridspec(2,2,height_ratios=[.82,1.18],width_ratios=[1.28,.72])

# a. Hansen-style full-model adjusted R2 and spatial inference.
ax=fig.add_subplot(gs[0,0]); x=np.arange(6); w=.34
for level,off in [("Subclass",-w/2),("Cluster",w/2)]:
    z=tot[tot.level.eq(level.lower())].set_index("outcome").loc[bands]
    bars=ax.bar(x+off,z.adjusted_r2,w,color=colors[level],label=level)
    for rect,q in zip(bars,z.spin_q_bh_six_bands):
        if q<.05: ax.plot(rect.get_x()+rect.get_width()/2,rect.get_height()-.035,"o",ms=2.8,color="#222",clip_on=False)
ax.set_xticks(x,[names[b] for b in bands]); ax.set_ylim(0,1.08); ax.set_ylabel("Adjusted $R^2$")
ax.set_title("a  Full-model fit with spatial inference",loc="left",fontweight="bold",fontsize=8)
ax.legend(ncol=2,loc="upper left",fontsize=6); ax.grid(axis="y",color="#e7e7e7",lw=.45); ax.set_axisbelow(True)
ax.text(.99,.03,"Black dots: spin-test q < 0.05 across six bands",ha="right",transform=ax.transAxes,fontsize=5.2,color="#555")

# b. OOF Ridge kept separate from full-model inference.
ax=fig.add_subplot(gs[0,1]); y=np.arange(6)
for level,dy in [("Subclass",-.11),("Cluster",.11)]:
    p=pd.read_csv(PATHS[level]/"performance.csv"); p=p[p.phenotype.eq("meg")].set_index("outcome").loc[bands]
    ax.scatter(p.oof_r2,y+dy,s=28,color=colors[level],label=level,zorder=3)
ax.axvline(0,color="#555",lw=.7); ax.set_yticks(y,[names[b] for b in bands]); ax.invert_yaxis()
ax.set_xlabel("Out-of-fold $R^2$"); ax.set_title("b  Held-out prediction",loc="left",fontweight="bold",fontsize=8)
ax.grid(axis="x",color="#e7e7e7",lw=.45); ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2,fontsize=5.8)

# c. Dominance only for spatially significant cluster models.
sig=tot[(tot.level.eq("cluster"))&(tot.spin_q_bh_six_bands<.05)].outcome.tolist()
d=pd.read_csv(PATHS["Cluster"]/"dominance.csv"); top=[]
for b in sig: top.extend(d[(d.phenotype.eq("meg"))&(d.outcome.eq(b))].nlargest(3,"relative_dominance").feature.tolist())
features=list(dict.fromkeys(top)); mat=pd.DataFrame(index=features)
for b in sig: mat[names[b]]=d[(d.phenotype.eq("meg"))&(d.outcome.eq(b))].set_index("feature").relative_dominance.reindex(features)*100
mat=mat.loc[mat.max(axis=1).sort_values(ascending=False).index]
ax=fig.add_subplot(gs[1,0]); sns.heatmap(mat,cmap="YlOrBr",vmin=0,linewidths=.3,linecolor="white",
 cbar_kws={"label":"Relative dominance (%)","shrink":.76},ax=ax)
ax.set_title("c  Cluster dominance for significant full models",loc="left",fontweight="bold",fontsize=8)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticklabels(ax.get_xticklabels(),rotation=0); ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=5.5)

# d. Agreement between in-sample dominance and held-out SHAP.
ax=fig.add_subplot(gs[1,1]); sh=pd.read_csv(PATHS["Cluster"]/"oof_shap.csv"); rows=[]
for b in sig:
    dd=d[(d.phenotype.eq("meg"))&(d.outcome.eq(b))].set_index("feature").relative_dominance
    ss=sh[(sh.phenotype.eq("meg"))&(sh.outcome.eq(b))].set_index("feature").relative_shap
    common=dd.index.intersection(ss.index); rho,p=spearmanr(dd.loc[common],ss.loc[common])
    rows.append((b,rho,p))
agree=pd.DataFrame(rows,columns=["outcome","spearman_rho","p"])
yy=np.arange(len(agree)); ax.barh(yy,agree.spearman_rho,color="#6D9F71",height=.58)
ax.set_yticks(yy,[names[b] for b in agree.outcome]); ax.invert_yaxis(); ax.set_xlim(-1,1); ax.axvline(0,color="#555",lw=.7)
ax.set_xlabel(r"Dominance–SHAP Spearman $\rho$"); ax.set_title("d  Attribution agreement",loc="left",fontweight="bold",fontsize=8)
ax.grid(axis="x",color="#e7e7e7",lw=.45); ax.set_axisbelow(True)
for i,row in agree.iterrows(): ax.text(row.spearman_rho+(.04 if row.spearman_rho>=0 else -.04),i,f"{row.spearman_rho:.2f}",ha="left" if row.spearman_rho>=0 else "right",va="center",fontsize=5.7)

base=OUT/"nonlaminar_hansen_style_meg_draft"
for ext,kw in [("png",{"dpi":300}),("svg",{}),("pdf",{}),("tiff",{"dpi":600})]: fig.savefig(f"{base}.{ext}",bbox_inches="tight",facecolor="white",**kw)
plt.close(fig)
tot.to_csv(OUT/"hansen_meg_total_source.csv",index=False); mat.to_csv(OUT/"hansen_meg_dominance_source.csv"); agree.to_csv(OUT/"hansen_meg_agreement_source.csv",index=False)
caption=("Figure Z | Cell-composition correlates of MEG power following a neurotransmitter-mapping analysis framework. "
"a, Adjusted R² of standardized ordinary least-squares models; asterisks denote Alexander–Bloch spin-test results passing Benjamini–Hochberg correction across six bands within resolution. "
"b, Lobe-wise out-of-fold R² from separately evaluated Ridge models. c, Relative dominance is shown only for cluster models passing the full-model spatial test. "
"d, Rank agreement between in-sample dominance and held-out SHAP attribution. Full-model fit, held-out prediction and attribution answer distinct questions and should not be interpreted interchangeably.")
(OUT/"hansen_meg_caption.txt").write_text(caption,encoding="utf-8")
(OUT/"hansen_meg_qa.json").write_text(json.dumps({"significant_cluster_bands":sig,"significant_subclass_bands":[],"n_spins":1000,"panels":4},indent=2),encoding="utf-8")
print(base)
