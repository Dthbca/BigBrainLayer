"""Standalone primary-result figures for the reclosed best pipeline."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
"svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.spines.right":False,"axes.spines.top":False,
"axes.linewidth":.8,"legend.frameon":False})
BEST="within_region_cross_layer__clr_false__thickness_relative"
LAYERS=["l1","l2","l3","l4","l5","l6"]

def save(fig,out,name):
    for ext,dpi in (("svg",None),("pdf",None),("png",300),("tiff",600)):
        kw={"bbox_inches":"tight","facecolor":"white"}
        if dpi: kw["dpi"]=dpi
        fig.savefig(out/f"{name}.{ext}",**kw)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    long=pd.read_csv(a.results/"source_data"/"all_branch_spin_results.csv")
    d=long[long.branch.eq(BEST)].copy(); r=d.pivot(index="layer",columns="ctype",values="correlation").reindex(LAYERS)
    q=d.pivot(index="layer",columns="ctype",values="p_adjusted").reindex(index=LAYERS,columns=r.columns)
    r.to_csv(a.output/"reclosed_best_r_matrix.csv"); q.to_csv(a.output/"reclosed_best_q_matrix.csv")
    fig,ax=plt.subplots(figsize=(7.2,2.8)); im=ax.imshow(r,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    ax.set_xticks(range(len(r.columns)),r.columns,rotation=65,ha="right",fontsize=6); ax.set_yticks(range(6),["I","II","III","IV","V","VI"])
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v=q.iloc[i,j]
            if np.isfinite(v) and v<.05: ax.text(j,i,"***" if v<.001 else "**" if v<.01 else "*",ha="center",va="center",fontsize=5,color="black")
    cb=fig.colorbar(im,ax=ax,pad=.015,shrink=.82); cb.set_label("Pearson r")
    ax.set_xlabel("Cell-type subclass"); ax.set_ylabel("Cortical layer"); fig.tight_layout(); save(fig,a.output,"figure_reclosed_best_heatmap")

    summary=pd.read_csv(a.results/"branch_summary.csv").set_index("branch").loc[BEST]
    null=pd.read_csv(a.results/"source_data"/f"{BEST}__whole_null.csv")["null"]
    fig,ax=plt.subplots(figsize=(5.2,3.4)); ax.hist(null,bins=28,color="#C8D0D5",edgecolor="white",linewidth=.5)
    ax.axvline(summary.whole_match_stat,color="#2C6C8A",lw=2,label=f"Observed = {summary.whole_match_stat:.4f}")
    ax.set(xlabel="Mean matched-layer Pearson r",ylabel="Layer-label permutations")
    ax.text(.03,.94,f"Exact one-sided p = {summary.whole_match_p:.6f}\n6! = {int(summary.whole_n_permutations)} permutations",transform=ax.transAxes,va="top")
    ax.legend(loc="upper right"); fig.tight_layout(); save(fig,a.output,"figure_reclosed_best_permutation")

    shap=pd.read_csv(a.results/"shap"/"shap_all_branch_ctype.csv"); s=shap[shap.branch.eq(BEST)].sort_values("relative_contribution",ascending=True)
    colors=s["class"].map({"Excitatory":"#5B84A6","Inhibitory":"#B66D73","Non-neuron":"#6F9A7A","Other":"#999999"})
    fig,ax=plt.subplots(figsize=(5.4,5.5)); ax.barh(s.ctype,s.relative_contribution*100,color=colors)
    ax.set_xlabel("Relative mean |SHAP| contribution (%)"); ax.set_ylabel("")
    handles=[plt.Line2D([0],[0],marker="s",ls="",color=c,label=k) for k,c in {"Excitatory":"#5B84A6","Inhibitory":"#B66D73","Non-neuron":"#6F9A7A"}.items()]
    ax.legend(handles=handles,loc="lower right"); fig.tight_layout(); save(fig,a.output,"figure_reclosed_shap_contribution")
    qa={"status":"PASS","backend":"Python/matplotlib","primary_branch":BEST,"figures":3,"n_tests":len(d),"n_sig":int(d.p_adjusted.lt(.05).sum()),"finite_r":bool(np.isfinite(d.correlation).all())}
    (a.output/"primary_figure_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8"); print(qa)
if __name__=="__main__": main()
