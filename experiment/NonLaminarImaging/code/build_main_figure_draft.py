from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

ROOT=Path(os.environ.get("NONLAMINAR_ROOT",r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging"))
RESULTS=ROOT/"results"; OUT=ROOT/"report"/"figure_draft"; OUT.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,"legend.frameon":False,"figure.facecolor":"white"})

PATHS={"subclass":RESULTS/"subclass_main_20260821"/"ratio_none_subclass","cluster":RESULTS/"cluster_secondary_20260821"/"ratio_none_cluster"}
def load(name):
    z=[]
    for level,p in PATHS.items():
        x=pd.read_csv(p/name); x["level"]=level; z.append(x)
    return pd.concat(z,ignore_index=True)
def bh(p):
    p=np.asarray(p,float); o=np.argsort(p); q=np.empty(len(p)); q[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; return np.minimum(q,1)

perf=load("performance.csv"); total=load("total_models.csv"); shap=load("oof_shap.csv")
focus=[("meg","alpha","Alpha"),("meg","theta","Theta"),("enigma","adhd","ADHD"),("enigma","asd","ASD"),("enigma","epilepsy_rtle","Right TLE"),("enigma","parkinson","Parkinson")]

fig=plt.figure(figsize=(7.2,8.25),constrained_layout=True)
gs=fig.add_gridspec(3,2,height_ratios=[.75,1.35,1.25],width_ratios=[1.05,.95])

# a. Workflow — deliberately concise and editable.
ax=fig.add_subplot(gs[0,:]); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(-.015,.99,"a",fontweight="bold",fontsize=9,va="top")
boxes=[(.02,.30,.18,.43,"Macaque spatial\ntranscriptomics","D99 · ratio"),(.27,.30,.18,.43,"Homologous\ncell mapping","Subclass / cluster"),(.52,.30,.18,.43,"Cross-species\nrelabeling","BN · 105 regions"),(.77,.30,.20,.43,"Human cortical maps","MEG · ENIGMA")]
colors=["#EAF2F6","#EAF2F6","#EAF2F6","#F8EEE9"]
for (x,y,w,h,title,sub),c in zip(boxes,colors):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=.015",fc=c,ec="#52616B",lw=.8))
    ax.text(x+w/2,y+h*.62,title,ha="center",va="center",fontweight="bold",fontsize=7)
    ax.text(x+w/2,y+h*.22,sub,ha="center",va="center",fontsize=6,color="#52616B")
for i in range(3):
    x1=boxes[i][0]+boxes[i][2]+.008; x2=boxes[i+1][0]-.008
    ax.add_patch(FancyArrowPatch((x1,.515),(x2,.515),arrowstyle="-|>",mutation_scale=9,lw=.8,color="#52616B"))
ax.text(.5,.11,"Spatial spin test  ·  lobe-wise out-of-fold prediction  ·  held-out SHAP",ha="center",va="center",fontsize=7,fontweight="bold",color="#304B5F")

# b. Primary ratio results across all outcomes.
ax=fig.add_subplot(gs[1,0])
order=[x[1] for x in focus]
other_meg=["gamma2","gamma1","beta","delta"]
other_enigma=["schizotypy","ocd","epilepsy_ltle","depression","epilepsy_gge","schizophrenia","bipolar","22q","obesity"]
row_order=order+other_meg+other_enigma
display={"gamma2":"Gamma 2","gamma1":"Gamma 1","beta":"Beta","delta":"Delta","schizotypy":"Schizotypy","ocd":"OCD","epilepsy_ltle":"Left TLE","depression":"Depression","epilepsy_gge":"Generalized epilepsy","schizophrenia":"Schizophrenia","bipolar":"Bipolar disorder","22q":"22q11.2 deletion","obesity":"Obesity"}
piv=perf.pivot(index="outcome",columns="level",values="oof_r2").reindex(index=row_order,columns=["subclass","cluster"])
sns.heatmap(piv,cmap="RdBu_r",center=0,vmin=-.8,vmax=.8,linewidths=.35,linecolor="white",cbar_kws={"label":"Out-of-fold $R^2$","shrink":.64},ax=ax)
ax.set_xticklabels(["Subclass","Cluster"],rotation=0); ax.set_yticklabels([dict((b,c) for _,b,c in focus).get(i,display.get(i,i)) for i in row_order],rotation=0,fontsize=5.5)
ax.hlines([6,10],*ax.get_xlim(),colors="#222",linewidth=.8)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("b  Ratio-based prediction across cortical maps",loc="left",fontweight="bold",fontsize=8)

# c. Evidence ladder for focused outcomes.
ax=fig.add_subplot(gs[1,1]); ev=[]
for ph,out,label in focus:
    for level in ["subclass","cluster"]:
        r=perf[(perf.phenotype==ph)&(perf.outcome==out)&(perf.level==level)].iloc[0]
        t=total[(total.phenotype==ph)&(total.level==level)].copy(); t["q_family"]=bh(t.spin_p)
        ev.append({"label":label,"level":level,"oof_r2":r.oof_r2,"q":float(t.loc[t.outcome.eq(out),"q_family"].iloc[0])})
ev=pd.DataFrame(ev); palette={"subclass":"#527EB7","cluster":"#CB624A"}; ypos=np.arange(len(focus))
for level,dy in [("subclass",-.11),("cluster",.11)]:
    x=ev[ev.level.eq(level)].set_index("label").loc[[z[2] for z in focus]]
    ax.scatter(x.oof_r2,ypos+dy,s=30,color=palette[level],label=level.capitalize(),zorder=3)
    for yi,(r,q) in enumerate(zip(x.oof_r2,x.q)):
        if q<.05: ax.scatter(r,yi+dy,s=58,facecolors="none",edgecolors="#20242a",lw=.7,zorder=4)
ax.axvline(0,color="#555",lw=.7); ax.set_yticks(ypos,[z[2] for z in focus]); ax.invert_yaxis(); ax.set_xlabel("Out-of-fold $R^2$"); ax.set_title("c  Focused evidence",loc="left",fontweight="bold",fontsize=8)
ax.grid(axis="x",color="#e4e4e4",lw=.5); ax.legend(loc="upper center",bbox_to_anchor=(.5,-.105),ncol=2,fontsize=6,title=None)
ax.text(.5,-.23,"Outlined symbols: total-model q < 0.05 within outcome family",ha="center",transform=ax.transAxes,fontsize=5.3,color="#555")

# d. Primary subclass interpretation; top union avoids cherry-picking per panel.
ax=fig.add_subplot(gs[2,:]); sub=shap[shap.level.eq("subclass")]
selected=[("meg","alpha","Alpha"),("enigma","adhd","ADHD"),("enigma","epilepsy_rtle","Right TLE"),("enigma","parkinson","Parkinson")]
top=[]
for ph,out,_ in selected: top.extend(sub[(sub.phenotype==ph)&(sub.outcome==out)].nlargest(3,"relative_shap").feature.tolist())
features=list(dict.fromkeys(top))
matrix=pd.DataFrame(index=features)
for ph,out,label in selected: matrix[label]=sub[(sub.phenotype==ph)&(sub.outcome==out)].set_index("feature").relative_shap.reindex(features)*100
matrix=matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
sns.heatmap(matrix,cmap="YlOrBr",vmin=0,linewidths=.35,linecolor="white",cbar_kws={"label":"Relative held-out SHAP (%)","shrink":.75},ax=ax)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticklabels(ax.get_xticklabels(),rotation=0); ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=6)
ax.set_title("d  Subclass contributions to held-out predictions",loc="left",fontweight="bold",fontsize=8)

base=OUT/"nonlaminar_ratio_main_figure_draft"
for ext,kw in [("png",{"dpi":300}),("svg",{}),("pdf",{}),("tiff",{"dpi":600})]: fig.savefig(f"{base}.{ext}",bbox_inches="tight",facecolor="white",**kw)
plt.close(fig)
piv.to_csv(OUT/"panel_b_source_data.csv"); ev.to_csv(OUT/"panel_c_source_data.csv",index=False); matrix.to_csv(OUT/"panel_d_source_data.csv")
caption="""Figure X | Homologous cell-composition maps predict selected human cortical phenotypes. a, Macaque spatial-transcriptomic cell proportions were mapped to homologous human subclass and cluster identities, relabelled from D99 to the Brainnetome atlas, and evaluated against MEG frequency-band and ENIGMA cortical-thickness effect maps. b, Lobe-wise out-of-fold R² for ratio-based subclass and cluster models across six MEG and thirteen ENIGMA outcomes. c, Out-of-fold R² for the six focused outcomes; outlined symbols indicate total Ridge models significant after Alexander–Bloch spin testing and Benjamini–Hochberg correction within the MEG or ENIGMA outcome family. d, Relative mean absolute held-out SHAP values for the union of the three leading subclass features in Alpha, ADHD, right temporal-lobe epilepsy and Parkinson disease models. All models used 105 BN regions, five lobe-wise folds, 1,000 spatial rotations and seed 42. SHAP values describe predictive attribution and do not establish independent abundance effects or causality."""
(OUT/"figure_caption.txt").write_text(caption,encoding="utf-8")
qa={"core_claim":"Ratio-based homologous cell maps predict selected MEG and ENIGMA cortical phenotypes at subclass and cluster resolution.","panels":{"a":"workflow","b":"all outcome OOF R2","c":"focused OOF plus total spin significance","d":"subclass held-out SHAP"},"primary_feature":"ratio","sensitivity_included":False,"formats":{e:Path(f"{base}.{e}").stat().st_size for e in ["png","svg","pdf","tiff"]},"source_data":["panel_b_source_data.csv","panel_c_source_data.csv","panel_d_source_data.csv"]}
(OUT/"figure_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
print(base)
