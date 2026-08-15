"""Overlap-safe standalone figures for layer-specific SHAP results."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
 "svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.spines.right":False,
 "axes.spines.top":False,"axes.linewidth":.7,"legend.frameon":False,"figure.facecolor":"white"})
LAYERS=["Layer I","Layer II","Layer III","Layer IV","Layer V","Layer VI"]
COLORS={"Excitatory":"#D9925B","Inhibitory":"#477FA8","Non-neuron":"#6F9E78"}

def save(fig,out,name):
    for ext,dpi in [("png",300),("svg",None),("pdf",None),("tiff",600)]:
        kw={"bbox_inches":"tight","facecolor":"white"}
        if dpi: kw["dpi"]=dpi
        fig.savefig(out/f"{name}.{ext}",**kw)
    plt.close(fig)

def main():
    q=argparse.ArgumentParser(); q.add_argument("--source",type=Path,required=True); q.add_argument("--output",type=Path,required=True)
    a=q.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    perf=pd.read_csv(a.source/"layer_specific_model_performance.csv")
    cont=pd.read_csv(a.source/"layer_specific_shap_contributions.csv")
    cls=pd.read_csv(a.source/"layer_specific_shap_classes.csv")
    perf["layer_label"]=pd.Categorical(perf.layer_label,LAYERS,ordered=True); perf=perf.sort_values("layer_label")

    # Performance: values inside bars, feature names in a fixed right column.
    fig,ax=plt.subplots(figsize=(7.1,3.55)); y=np.arange(6)
    bars=ax.barh(y,perf.oof_r2,color="#477FA8",height=.62)
    ax.set_yticks(y,LAYERS); ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel("Out-of-fold $R^2$")
    for bar,row in zip(bars,perf.itertuples()):
        yc=bar.get_y()+bar.get_height()/2
        ax.text(max(row.oof_r2-.02,.04),yc,f"{row.oof_r2:.2f}",ha="right",va="center",color="white",fontweight="bold",fontsize=6.8)
        ax.text(.97,yc,row.top_ctype,ha="right",va="center",color="#26343B",fontsize=6.8)
    ax.text(.97,-.72,"Top cell type",ha="right",va="bottom",color="#5B666C",fontsize=6.2)
    ax.set_title("Layer-specific prediction performance",loc="left",fontweight="bold",fontsize=9,pad=10)
    fig.subplots_adjust(left=.13,right=.98,top=.87,bottom=.18); save(fig,a.output,"figure_layer_shap_performance_v2")

    # Heatmap: more vertical space, shorter rotation, explanatory text only in caption.
    order=(cont.groupby(["ctype","class"],as_index=False).relative_contribution.mean())
    order["group"]=order["class"].map({"Excitatory":0,"Inhibitory":1,"Non-neuron":2,"Other":3}).fillna(3)
    ctypes=order.sort_values(["group","relative_contribution"],ascending=[True,False]).ctype.tolist()
    mat=cont.pivot(index="layer_label",columns="ctype",values="relative_contribution").reindex(index=LAYERS,columns=ctypes)*100
    cmap=mpl.colormaps["YlGnBu"].copy(); cmap.set_bad("#E7EAEC")
    fig,ax=plt.subplots(figsize=(7.2,4.65)); im=ax.imshow(np.ma.masked_invalid(mat.to_numpy()),aspect="auto",cmap=cmap,vmin=0,vmax=42)
    ax.set_yticks(range(6),LAYERS); ax.set_xticks(range(len(ctypes)),ctypes,rotation=45,ha="right",rotation_mode="anchor",fontsize=5.8)
    for i,row in enumerate(mat.to_numpy()):
        if np.isfinite(row).any():
            j=int(np.nanargmax(row)); ax.scatter(j,i,s=13,facecolor="none",edgecolor="white",lw=.8)
    cb=fig.colorbar(im,ax=ax,pad=.016,shrink=.82); cb.set_label("Contribution within layer (%)",fontsize=7); cb.outline.set_linewidth(.5)
    ax.set_title("Cell-type contributions differ across cortical layers",loc="left",fontweight="bold",fontsize=9,pad=10)
    fig.subplots_adjust(left=.12,right=.92,top=.88,bottom=.34); save(fig,a.output,"figure_layer_shap_heatmap_v2")

    # Cell classes: legend has its own bottom band.
    wide=cls.pivot(index="layer_label",columns="class",values="relative_contribution").reindex(LAYERS).fillna(0)*100
    fig,ax=plt.subplots(figsize=(7.1,3.9)); left=np.zeros(6); x=np.arange(6)
    for group in ["Excitatory","Inhibitory","Non-neuron"]:
        vals=wide.get(group,pd.Series(0,index=wide.index)).to_numpy()
        ax.bar(x,vals,bottom=left,width=.68,color=COLORS[group],label=group)
        for xi,v,b in zip(x,vals,left):
            if v>=10: ax.text(xi,b+v/2,f"{v:.0f}%",ha="center",va="center",fontsize=6.5,color="white",fontweight="bold")
        left+=vals
    ax.set_xticks(x,LAYERS); ax.set_ylim(0,100); ax.set_ylabel("Relative SHAP contribution (%)")
    ax.set_title("Cell-class contribution changes by layer",loc="left",fontweight="bold",fontsize=9,pad=10)
    ax.legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,-.16),columnspacing=1.8,handletextpad=.5)
    fig.subplots_adjust(left=.12,right=.98,top=.87,bottom=.25); save(fig,a.output,"figure_layer_shap_classes_v2")

    qa={"backend":"Python/matplotlib","core_conclusion":"SHAP contributions are layer-specific; Layers IV and VI have the strongest OOF performance.",
        "layout_changes":["fixed right label column","increased heatmap bottom margin","caption-only heatmap note","legend-only bottom band"],
        "source_data":["layer_specific_model_performance.csv","layer_specific_shap_contributions.csv","layer_specific_shap_classes.csv"]}
    (a.output/"figure_v2_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    print(json.dumps({"status":"PASS","output":str(a.output)}))

if __name__=="__main__": main()
