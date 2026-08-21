from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

ROOT=Path(os.environ.get("NONLAMINAR_ROOT", r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging"))
RES=ROOT/"results"; OUT=ROOT/"report"/"figure_draft"; OUT.mkdir(parents=True,exist_ok=True)
P={"subclass":RES/"subclass_main_20260821"/"ratio_none_subclass",
   "cluster":RES/"cluster_secondary_20260821"/"ratio_none_cluster"}
TM=pd.read_csv(RES/"hansen_style_meg_20260821"/"hansen_style_total_models.csv")
TE=pd.read_csv(RES/"hansen_style_enigma_20260821"/"hansen_style_total_models.csv")
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"svg.fonttype":"none",
 "pdf.fonttype":42,"font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,
 "legend.frameon":False,"figure.facecolor":"white"})
blue="#527EB7"; orange="#CB624A"; gold="#E9A23B"; dark="#304B5F"; grey="#65727A"

def perf(level,ph):
    x=pd.read_csv(P[level]/"performance.csv"); return x[x.phenotype.eq(ph)].set_index("outcome")
def top_dom(level,ph,out):
    x=pd.read_csv(P[level]/"dominance.csv"); x=x[(x.phenotype==ph)&(x.outcome==out)]
    return x.sort_values("relative_dominance",ascending=False).iloc[0]

fig=plt.figure(figsize=(7.2,8.8),constrained_layout=True)
gs=fig.add_gridspec(3,2,height_ratios=[.78,1.1,1.22],width_ratios=[.72,1.28])

# a — pipeline and comparison logic
ax=fig.add_subplot(gs[0,:]); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(-.012,.99,"a",fontweight="bold",fontsize=9,va="top")
boxes=[(.015,.35,.17,.42,"Macaque spatial\ntranscriptomics","226 plot classes"),
       (.235,.35,.17,.42,"Homologous\ncell mapping","191 mapped"),
       (.455,.35,.17,.42,"Human BN maps","23 subclasses · 71 clusters"),
       (.72,.35,.13,.42,"MEG","6 bands"),(.865,.35,.12,.42,"ENIGMA","13 disorders")]
cols=["#EAF2F6","#EAF2F6","#E3EFF4","#F8EEE9","#F8EEE9"]
for (x,y,w,h,t,s),c in zip(boxes,cols):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.01,rounding_size=.014",fc=c,ec="#52616B",lw=.75))
    ax.text(x+w/2,y+h*.61,t,ha="center",va="center",fontweight="bold",fontsize=6.6)
    ax.text(x+w/2,y+h*.20,s,ha="center",va="center",fontsize=5.5,color=grey)
for x1,x2 in [(.195,.225),(.415,.445),(.635,.71)]: ax.add_patch(FancyArrowPatch((x1,.56),(x2,.56),arrowstyle="-|>",mutation_scale=8,lw=.75,color="#52616B"))
ax.plot([.67,.67],[.56,.19],color="#52616B",lw=.75); ax.plot([.67,.79],[.19,.19],color="#52616B",lw=.75); ax.plot([.79,.79],[.19,.34],color="#52616B",lw=.75)
ax.plot([.67,.925],[.19,.19],color="#52616B",lw=.75); ax.plot([.925,.925],[.19,.34],color="#52616B",lw=.75)
ax.text(.5,.08,"Spatial total-model test  →  lobe-wise out-of-fold prediction  →  held-out cellular attribution",ha="center",fontsize=6.6,fontweight="bold",color=dark)

# b — hierarchy within HomoloMap; no cross-modality comparator
ax=fig.add_subplot(gs[1,0]); cats=["Mapped source\ncell types","Human-mapped\nclusters","Human-mapped\nsubclasses"]; vals=[191,71,23]
y=np.arange(3); ax.barh(y,vals,color=["#A8AFB4",orange,blue],height=.58)
ax.set_yticks(y,cats); ax.invert_yaxis(); ax.set_xlim(0,210); ax.set_xlabel("Cell-type maps")
for i,v in enumerate(vals): ax.text(v+1.5,i,str(v),va="center",fontweight="bold",fontsize=6.4)
ax.set_title("b  HomoloMap cell-type hierarchy",loc="left",fontweight="bold",fontsize=8)
ax.text(0,-.18,"Cluster and subclass are nested human-mapped resolutions.",transform=ax.transAxes,fontsize=5.1,color="#555")
ax.grid(axis="x",color="#e6e6e6",lw=.45); ax.set_axisbelow(True)

# c — MEG: cluster total spatial evidence, OOF, and screened dominance label
ax=fig.add_subplot(gs[1,1]); bands=["delta","theta","alpha","beta","gamma1","gamma2"]; labs=["Delta","Theta","Alpha","Beta","Gamma 1","Gamma 2"]
t=TM[TM.level.eq("cluster")].set_index("outcome").loc[bands]; p=perf("cluster","meg").loc[bands]
yy=np.arange(6); ax.scatter(t.adjusted_r2,yy-.13,s=28,color=orange,label="Adjusted $R^2$",zorder=3)
ax.scatter(p.oof_r2,yy+.13,s=28,color=blue,label="OOF $R^2$",zorder=3)
for i,(v,q) in enumerate(zip(t.adjusted_r2,t.spin_q_bh_six_bands)):
    if q<.05: ax.scatter(v,i-.13,s=55,facecolors="none",edgecolors="#222",lw=.7,zorder=4)
ax.axvline(0,color="#555",lw=.7); ax.set_yticks(yy,labs); ax.invert_yaxis(); ax.set_xlim(-.42,1.08); ax.set_xlabel("Model evidence")
ax.set_title("c  MEG: spatial fit and generalization",loc="left",fontweight="bold",fontsize=8)
ax.grid(axis="x",color="#e6e6e6",lw=.45); ax.legend(loc="upper center",bbox_to_anchor=(.5,-.13),ncol=2,fontsize=5.6)
for i,out in enumerate(bands):
    if t.loc[out,"spin_q_bh_six_bands"]<.05:
        z=top_dom("cluster","meg",out); ax.text(1.065,i,f"{z.feature}",ha="right",va="center",fontsize=4.8,color="#6D4B12")
ax.text(.99,.02,"right labels: leading dominance",transform=ax.transAxes,ha="right",fontsize=4.8,color="#6D4B12")

# d — ENIGMA all outcomes at subclass level, same evidence hierarchy
ax=fig.add_subplot(gs[2,0]); names={"22q":"22q11.2","adhd":"ADHD","asd":"ASD","epilepsy_gge":"Generalized epilepsy","epilepsy_rtle":"Right TLE","epilepsy_ltle":"Left TLE","depression":"Depression","ocd":"OCD","schizophrenia":"Schizophrenia","bipolar":"Bipolar","obesity":"Obesity","schizotypy":"Schizotypy","parkinson":"Parkinson"}; order=list(names)
t=TE[TE.level.eq("subclass")].set_index("outcome").loc[order]; p=perf("subclass","enigma").loc[order]; yy=np.arange(len(order))
ax.scatter(t.adjusted_r2,yy-.13,s=18,color=orange,label="Adjusted $R^2$",zorder=3); ax.scatter(p.oof_r2,yy+.13,s=18,color=blue,label="OOF $R^2$",zorder=3)
for i,(v,q) in enumerate(zip(t.adjusted_r2,t.spin_q_bh_13_disorders)):
    if q<.05: ax.scatter(v,i-.13,s=38,facecolors="none",edgecolors="#222",lw=.65,zorder=4)
ax.axvline(0,color="#555",lw=.7); ax.set_yticks(yy,[names[x] for x in order],fontsize=5.3); ax.invert_yaxis(); ax.set_xlim(-.85,1.02); ax.set_xlabel("Model evidence")
ax.set_title("d  ENIGMA: spatial fit and generalization",loc="left",fontweight="bold",fontsize=8); ax.grid(axis="x",color="#e6e6e6",lw=.45)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2,fontsize=5.4)

# e — fine cluster identities from held-out SHAP; main demonstration of added resolution
ax=fig.add_subplot(gs[2,1]); sh=pd.read_csv(P["cluster"]/"oof_shap.csv"); selected=[("meg","alpha","Alpha"),("enigma","adhd","ADHD"),("enigma","asd","ASD"),("enigma","epilepsy_rtle","Right TLE"),("enigma","parkinson","Parkinson")]
tops=[]
for ph,out,lab in selected: tops.extend(sh[(sh.phenotype==ph)&(sh.outcome==out)].nlargest(2,"relative_shap").feature.tolist())
features=list(dict.fromkeys(tops)); mat=pd.DataFrame(index=features)
for ph,out,lab in selected: mat[lab]=sh[(sh.phenotype==ph)&(sh.outcome==out)].set_index("feature").relative_shap.reindex(features)*100
mat=mat.loc[mat.max(axis=1).sort_values(ascending=False).index]
sns.heatmap(mat,cmap="YlOrBr",vmin=0,linewidths=.25,linecolor="white",cbar_kws={"label":"Relative held-out SHAP (%)","shrink":.72},ax=ax)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticklabels(ax.get_xticklabels(),rotation=35,ha="right",fontsize=5.5); ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=5.3)
ax.set_title("e  Fine cluster attribution in held-out regions",loc="left",fontweight="bold",fontsize=8)

base=OUT/"homolomap_nonlaminar_final_figure_draft"
for ext,kw in [("png",{"dpi":300}),("svg",{}),("pdf",{}),("tiff",{"dpi":600})]: fig.savefig(f"{base}.{ext}",bbox_inches="tight",facecolor="white",**kw)
plt.close(fig)
pd.DataFrame({"map_family":cats,"n_maps":vals}).to_csv(OUT/"final_panel_b_resolution.csv",index=False)
pd.DataFrame({"band":labs,"adjusted_r2":t.iloc[:0].index}).to_csv(OUT/"final_placeholder.csv",index=False) if False else None
TM[TM.level.eq("cluster")].to_csv(OUT/"final_panel_c_meg_total.csv",index=False); perf("cluster","meg").to_csv(OUT/"final_panel_c_meg_oof.csv")
TE[TE.level.eq("subclass")].to_csv(OUT/"final_panel_d_enigma_total.csv",index=False); perf("subclass","enigma").to_csv(OUT/"final_panel_d_enigma_oof.csv")
mat.to_csv(OUT/"final_panel_e_cluster_shap.csv")
caption=("Figure 1 | HomoloMap links homologous spatial-transcriptomic cell composition to human cortical dynamics and disease maps at cellular resolution. "
"a, Macaque spatial-transcriptomic cell classes were mapped to homologous human identities and relabelled to 105 Brainnetome regions before comparison with MEG and ENIGMA maps. "
"b, Of 226 source plot classes, 191 were mapped and summarized into nested human cell-type resolutions comprising 71 clusters and 23 subclasses. "
"c,d, Standardized total models were assessed with adjusted R² and 1,000 Alexander–Bloch rotations with complete refitting and family-wise Benjamini–Hochberg correction (outlined symbols), while lobe-wise out-of-fold R² independently assessed generalization. "
"Labels in c identify the leading dominance feature only for spatially significant models. e, Relative mean absolute held-out SHAP values identify fine cluster candidates in selected models. Dominance and SHAP partition model dependence and do not establish independent abundance effects or causality.")
(OUT/"homolomap_final_figure_caption.txt").write_text(caption,encoding="utf-8")
(OUT/"homolomap_final_figure_qa.json").write_text(json.dumps({"core_claim":"HomoloMap resolves homologous cell composition at subclass and cluster levels and links these maps to cortical phenotypes.","backend":"Python","formats":{x:Path(f"{base}.{x}").stat().st_size for x in ["png","svg","pdf","tiff"]},"n_regions":105,"n_spins":1000,"external_modality_comparator":False},indent=2),encoding="utf-8")
print(base)
