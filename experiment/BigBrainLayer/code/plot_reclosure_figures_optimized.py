"""Optimized standalone figures for the reclosure report (Python only)."""
from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans","sans-serif"],
"svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.spines.right":False,"axes.spines.top":False,
"axes.linewidth":.8,"legend.frameon":False,"xtick.major.width":.7,"ytick.major.width":.7})
BEST="within_region_cross_layer__clr_false__thickness_relative"; LAYERS=["l1","l2","l3","l4","l5","l6"]
BLUE="#245A73"; BLUE2="#5D8CA3"; ORANGE="#B7654D"; GREY="#9AA3AA"; LIGHT="#D7DEE2"; DARK="#24313A"

def lab(b): return ("Cross-layer" if b.startswith("within_region") else "Within-layer")+" | "+("CLR" if "clr_true" in b else "No CLR")+" | "+("Relative" if b.endswith("relative") else "Absolute")
def save(fig,out,name):
    fig.tight_layout(pad=1.2)
    for ext,dpi in (("svg",None),("pdf",None),("png",300),("tiff",600)):
        kw={"bbox_inches":"tight","facecolor":"white","pad_inches":.05}
        if dpi: kw["dpi"]=dpi
        fig.savefig(out/f"{name}.{ext}",**kw)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); p.add_argument("--unclosed",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    new=pd.read_csv(a.results/"branch_summary.csv"); comp=pd.read_csv(a.results/"reclosure_vs_unclosed_summary.csv")
    order=new.branch.tolist(); comp=comp.set_index("branch").loc[order].reset_index(); comp["label"]=comp.branch.map(lab); comp.to_csv(a.output/"optimized_reclosure_comparison_source.csv",index=False)

    # Effect magnitude and FDR discoveries.
    y=np.arange(len(comp)); fig,ax=plt.subplots(figsize=(7.1,4.1))
    for i,r in comp.iterrows(): ax.plot([r.mean_abs_r_unclosed,r.mean_abs_r_reclosed],[i,i],color=LIGHT,lw=1.4,zorder=1)
    ax.scatter(comp.mean_abs_r_unclosed,y,s=35,facecolor="white",edgecolor=GREY,lw=1,label="Unclosed sensitivity",zorder=3)
    ax.scatter(comp.mean_abs_r_reclosed,y,s=40,color=BLUE,edgecolor="white",lw=.5,label="Reclosed primary",zorder=4)
    for i,r in comp.iterrows(): ax.text(max(r.mean_abs_r_unclosed,r.mean_abs_r_reclosed)+.012,i,f"{int(r.n_sig_unclosed)} → {int(r.n_sig_reclosed)}",va="center",fontsize=6.5,color=DARK)
    ax.set_yticks(y,comp.label); ax.invert_yaxis(); ax.set_xlabel("Mean absolute spatial correlation, |r|")
    ax.set_xlim(.24,.53); ax.set_xticks(np.arange(.25,.51,.05)); ax.legend(loc="lower right",handletextpad=.5)
    ax.text(.985,.02,"labels: FDR-significant tests",transform=ax.transAxes,ha="right",va="bottom",fontsize=6,color=GREY)
    save(fig,a.output,"figure_reclosure_branch_effects_opt")

    # Pairwise effect concordance.
    pairs=pd.read_csv(a.results/"source_data"/"reclosure_vs_unclosed_pairs.csv")
    fig,ax=plt.subplots(figsize=(5.2,4.7)); families={True:BLUE,False:"#9B6A8A"}
    for cross,color in families.items():
        d=pairs[pairs.branch.str.startswith("within_region").eq(cross)]
        ax.scatter(d.correlation_unclosed,d.correlation_reclosed,s=11,alpha=.42,color=color,edgecolor="none",label="Cross-layer" if cross else "Within-layer")
    lo=math.floor(min(pairs.correlation_unclosed.min(),pairs.correlation_reclosed.min())*10)/10; hi=math.ceil(max(pairs.correlation_unclosed.max(),pairs.correlation_reclosed.max())*10)/10
    ax.plot([lo,hi],[lo,hi],ls="--",lw=1,color=DARK); ax.axhline(0,color=LIGHT,lw=.7); ax.axvline(0,color=LIGHT,lw=.7)
    ax.set(xlabel="Unclosed Pearson r",ylabel="Reclosed Pearson r",xlim=(lo,hi),ylim=(lo,hi)); ax.set_aspect("equal",adjustable="box"); ax.legend(loc="upper left")
    save(fig,a.output,"figure_reclosure_effect_consistency_opt")

    # Whole-layer inference.
    fig,ax=plt.subplots(figsize=(7.1,4.1)); xo=-np.log10(comp.whole_p_unclosed); xn=-np.log10(comp.whole_p_reclosed)
    for i in range(len(comp)): ax.plot([xo.iloc[i],xn.iloc[i]],[i,i],color=LIGHT,lw=1.4)
    ax.scatter(xo,y,s=35,facecolor="white",edgecolor=GREY,lw=1,label="Unclosed sensitivity",zorder=3)
    ax.scatter(xn,y,s=40,color=ORANGE,edgecolor="white",lw=.5,label="Reclosed primary",zorder=4)
    ax.axvline(-np.log10(.05),ls="--",lw=1,color=DARK); ax.text(-np.log10(.05)+.03,-.58,"p = 0.05",fontsize=6.5,va="center")
    ax.set_yticks(y,comp.label); ax.invert_yaxis(); ax.set_xlabel("−log₁₀ whole-layer permutation p")
    for i,r in comp.iterrows(): ax.text(max(xo.iloc[i],xn.iloc[i])+.045,i,f"J={r.sig_jaccard:.2f}",va="center",fontsize=6.5)
    ax.set_xlim(0,max(xo.max(),xn.max())+.42); ax.legend(loc="lower right")
    save(fig,a.output,"figure_reclosure_inference_change_opt")

    # Reclosed best heatmap with compact symmetric colorbar.
    long=pd.read_csv(a.results/"source_data"/"all_branch_spin_results.csv"); d=long[long.branch.eq(BEST)]
    r=d.pivot(index="layer",columns="ctype",values="correlation").reindex(LAYERS); q=d.pivot(index="layer",columns="ctype",values="p_adjusted").reindex(index=LAYERS,columns=r.columns)
    cmap=mpl.colormaps["RdBu_r"].copy(); cmap.set_bad("#E8ECEF"); vmax=math.ceil(np.nanmax(np.abs(r.to_numpy()))*10)/10
    norm=mpl.colors.TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    fig,ax=plt.subplots(figsize=(7.2,2.75)); im=ax.imshow(r,cmap=cmap,norm=norm,aspect="auto")
    ax.set_xticks(range(len(r.columns)),r.columns,rotation=62,ha="right",fontsize=6); ax.set_yticks(range(6),["I","II","III","IV","V","VI"])
    ax.tick_params(length=0); ax.set_xlabel("Cell-type subclass"); ax.set_ylabel("Cortical layer")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v=q.iloc[i,j]
            if np.isfinite(v) and v<.05:
                txt="***" if v<.001 else "**" if v<.01 else "*"; ax.text(j,i,txt,ha="center",va="center",fontsize=5.2,color="black",path_effects=[pe.withStroke(linewidth=1.1,foreground="white")])
    ticks=np.linspace(-vmax,vmax,7); cb=fig.colorbar(im,ax=ax,fraction=.022,pad=.012,aspect=28,ticks=ticks)
    cb.ax.set_yticklabels([f"{x:.1f}" if abs(x)>.001 else "0" for x in ticks]); cb.set_label("Spatial correlation (Pearson r)",labelpad=5); cb.outline.set_visible(False); cb.ax.tick_params(length=2,pad=2)
    save(fig,a.output,"figure_reclosed_best_heatmap_opt")

    # Exact permutation distribution.
    sm=new.set_index("branch").loc[BEST]; null=pd.read_csv(a.results/"source_data"/f"{BEST}__whole_null.csv")["null"]
    fig,ax=plt.subplots(figsize=(5.2,3.35)); ax.hist(null,bins=28,color="#C7D3D9",edgecolor="white",lw=.5)
    ax.axvline(sm.whole_match_stat,color=ORANGE,lw=2,label=f"Observed = {sm.whole_match_stat:.4f}")
    ax.set(xlabel="Mean matched-layer Pearson r",ylabel="Number of layer-label permutations"); ax.legend(loc="upper right")
    ax.text(.03,.94,f"Exact one-sided p = {sm.whole_match_p:.6f}\n6! = {int(sm.whole_n_permutations)} permutations",transform=ax.transAxes,va="top")
    save(fig,a.output,"figure_reclosed_best_permutation_opt")

    # Reclosed SHAP contributions.
    shap=pd.read_csv(a.results/"shap"/"shap_all_branch_ctype.csv"); s=shap[shap.branch.eq(BEST)].sort_values("relative_contribution",ascending=True)
    palette={"Excitatory":"#4F7F9F","Inhibitory":"#B06B74","Non-neuron":"#71947B","Other":GREY}; colors=s["class"].map(palette)
    fig,ax=plt.subplots(figsize=(5.3,5.4)); bars=ax.barh(s.ctype,s.relative_contribution*100,color=colors,edgecolor="white",lw=.4)
    ax.set_xlabel("Relative mean |SHAP| contribution (%)"); ax.set_ylabel(""); ax.set_xlim(0,s.relative_contribution.max()*100+1.25)
    for bar,val in zip(bars,s.relative_contribution*100):
        if val>=5.7: ax.text(val+.1,bar.get_y()+bar.get_height()/2,f"{val:.1f}",va="center",fontsize=6.2)
    handles=[mpl.lines.Line2D([0],[0],marker="s",ls="",color=c,label=k,markersize=6) for k,c in palette.items() if k!="Other"]
    ax.legend(handles=handles,loc="lower right",ncol=1); save(fig,a.output,"figure_reclosed_shap_contribution_opt")

    qa={"status":"PASS","backend":"Python/matplotlib","standalone_figures":6,"formats":["svg","pdf","png","tiff"],"heatmap_vlim":vmax,"colorbar_ticks":[float(x) for x in ticks],"n_tests":len(d),"n_sig":int(d.p_adjusted.lt(.05).sum()),"finite":bool(np.isfinite(d.correlation).all())}
    (a.output/"optimized_figure_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8"); print(qa)
if __name__=="__main__": main()
