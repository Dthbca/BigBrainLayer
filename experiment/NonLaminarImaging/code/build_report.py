from pathlib import Path
import html
import json
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(os.environ.get(
    "NONLAMINAR_ROOT",
    r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging"))
RESULTS = ROOT / "results"
OUT = ROOT / "report"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "figure.facecolor": "white", "axes.facecolor": "white",
})

BRANCHES = {
    "ratio_none_subclass": (RESULTS / "subclass_main_20260821" / "ratio_none_subclass", "Ratio · subclass"),
    "ratio_clr_subclass": (RESULTS / "subclass_main_20260821" / "ratio_clr_subclass", "Ratio–CLR · subclass"),
    "density_none_subclass": (RESULTS / "subclass_main_20260821" / "density_none_subclass", "Density · subclass"),
    "ratio_none_cluster": (RESULTS / "cluster_secondary_20260821" / "ratio_none_cluster", "Ratio · cluster"),
    "ratio_clr_cluster": (RESULTS / "cluster_secondary_20260821" / "ratio_clr_cluster", "Ratio–CLR · cluster"),
    "density_none_cluster": (RESULTS / "cluster_secondary_20260821" / "density_none_cluster", "Density · cluster"),
}
COLORS = {"ratio_none_subclass":"#3B6FB6", "ratio_clr_subclass":"#78A5D2",
          "density_none_subclass":"#7A6AAE", "ratio_none_cluster":"#C65D47",
          "ratio_clr_cluster":"#E39A82", "density_none_cluster":"#B47A4B"}

def save(fig, name, size=(7.2, 4.8)):
    fig.set_size_inches(*size)
    for ext, kw in [("png", {"dpi":300}), ("svg", {}), ("pdf", {}), ("tiff", {"dpi":600})]:
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)

def load_all(filename):
    out=[]
    for key,(path,label) in BRANCHES.items():
        x=pd.read_csv(path/filename); x["branch"]=key; x["branch_label"]=label; out.append(x)
    return pd.concat(out, ignore_index=True)

audit=pd.read_csv(RESULTS/"audit_summary.csv")
perf=load_all("performance.csv")
total=load_all("total_models.csv")
shap=load_all("oof_shap.csv")
dom=load_all("dominance.csv")
order=list(BRANCHES)
labels=[BRANCHES[k][1] for k in order]

# Figure 1: branch-level evidence hierarchy.
fig,axs=plt.subplots(1,3,gridspec_kw={"width_ratios":[1.3,1,1]},constrained_layout=True)
a=audit.set_index("branch").loc[order]
axs[0].barh(range(6),a.mean_oof_r2,color=[COLORS[k] for k in order])
axs[0].axvline(0,color="#333333",lw=.7); axs[0].set_yticks(range(6),labels); axs[0].invert_yaxis()
axs[0].set_xlabel("Mean out-of-fold $R^2$"); axs[0].set_title("a  Generalization",loc="left",fontweight="bold")
for i,v in enumerate(a.mean_oof_r2): axs[0].text(v+(.008 if v>=0 else .008),i,f"{v:.2f}",va="center",ha="left",fontsize=6)
axs[1].barh(range(6),a.n_positive_oof_r2,color=[COLORS[k] for k in order])
axs[1].set_yticks([]); axs[1].invert_yaxis(); axs[1].set_xlim(0,19); axs[1].set_xlabel("Outcomes with $R^2>0$")
axs[1].set_title("b  Positive predictions",loc="left",fontweight="bold")
for i,v in enumerate(a.n_positive_oof_r2): axs[1].text(v+.3,i,f"{int(v)}/19",va="center",fontsize=6)
axs[2].barh(np.arange(6)-.17,a.n_total_fdr,height=.32,color=[COLORS[k] for k in order],label="Total model")
axs[2].barh(np.arange(6)+.17,a.n_outcomes_pairwise_fdr,height=.32,color="#B8BEC7",label="Any cell type")
axs[2].set_yticks([]); axs[2].invert_yaxis(); axs[2].set_xlim(0,19); axs[2].set_xlabel("FDR-significant outcomes")
axs[2].set_title("c  Spatial inference",loc="left",fontweight="bold"); axs[2].legend(fontsize=6,loc="lower right")
save(fig,"figure1_branch_overview",(7.2,3.2))

# Figure 2: outcome-resolved OOF performance.
piv=perf.pivot(index="outcome",columns="branch",values="oof_r2").reindex(columns=order)
outcome_order=piv.max(axis=1).sort_values(ascending=False).index
piv=piv.loc[outcome_order]
fig,ax=plt.subplots(constrained_layout=True)
sns.heatmap(piv,cmap="RdBu_r",center=0,vmin=-.8,vmax=.8,linewidths=.25,linecolor="white",
            cbar_kws={"label":"Out-of-fold $R^2$","shrink":.75},ax=ax)
ax.set_xticklabels(labels,rotation=35,ha="right"); ax.set_ylabel(""); ax.set_xlabel("")
ax.set_title("Out-of-fold prediction across MEG and ENIGMA outcomes",loc="left",fontweight="bold")
save(fig,"figure2_oof_heatmap",(7.2,5.1))

# Figure 3: contribution profiles in selected reproducible outcomes, subclass main analysis.
selected=[("meg","alpha"),("enigma","adhd"),("enigma","parkinson")]
main=["ratio_none_subclass","ratio_clr_subclass","density_none_subclass"]
fig,axs=plt.subplots(1,3,constrained_layout=True)
for ax,(ph,outcome) in zip(axs,selected):
    subset=shap[(shap.phenotype==ph)&(shap.outcome==outcome)&(shap.branch.isin(main))]
    rank=subset.groupby("feature").relative_shap.mean().nlargest(8).index
    mat=subset[subset.feature.isin(rank)].pivot(index="feature",columns="branch",values="relative_shap").reindex(rank)
    mat=mat.reindex(columns=main)
    sns.heatmap(mat*100,cmap="YlOrBr",vmin=0,cbar=ax is axs[-1],linewidths=.25,linecolor="white",
                cbar_kws={"label":"Relative OOF SHAP (%)","shrink":.7},ax=ax)
    ax.set_xticklabels(["Ratio","Ratio–CLR","Density"],rotation=35,ha="right")
    ax.tick_params(axis="y", labelsize=6, pad=2)
    ax.set_ylabel(""); ax.set_xlabel(""); ax.set_title(outcome.replace("_"," ").upper() if outcome=="adhd" else outcome.capitalize(),fontweight="bold")
save(fig,"figure3_subclass_shap",(9.2,3.8))

# Source data behind plotted panels.
a.reset_index().to_csv(FIG/"figure1_source_data.csv",index=False)
piv.to_csv(FIG/"figure2_source_data.csv")
shap[(shap.branch.isin(main)) & shap.apply(lambda r:(r.phenotype,r.outcome) in selected,axis=1)].to_csv(FIG/"figure3_source_data.csv",index=False)

# Source-separated reporting figures. MEG and ENIGMA have different biological
# meanings and outcome-family sizes, so their summaries must not be pooled.
def phenotype_summary(ph):
    n_out = int(perf.loc[perf.phenotype.eq(ph), "outcome"].nunique())
    rows = []
    for key in order:
        pp = perf[(perf.phenotype == ph) & (perf.branch == key)]
        tt = total[(total.phenotype == ph) & (total.branch == key)]
        # Recalculate BH within the displayed phenotype family.
        from statsmodels.stats.multitest import multipletests
        tq = multipletests(tt.spin_p, method="fdr_bh")[1]
        rows.append({"branch": key, "mean_oof_r2": pp.oof_r2.mean(),
                     "n_positive": int((pp.oof_r2 > 0).sum()),
                     "n_total_fdr": int((tq < .05).sum()), "n_outcomes": n_out,
                     "best_outcome": pp.loc[pp.oof_r2.idxmax(), "outcome"],
                     "best_oof_r2": pp.oof_r2.max()})
    return pd.DataFrame(rows).set_index("branch")

def plot_ph_overview(ph, title, name):
    s = phenotype_summary(ph).loc[order]
    fig, axs = plt.subplots(1, 3, gridspec_kw={"width_ratios":[1.3,1,1]}, constrained_layout=True)
    axs[0].barh(range(6), s.mean_oof_r2, color=[COLORS[k] for k in order]); axs[0].axvline(0,color="#333",lw=.7)
    axs[0].set_yticks(range(6), labels); axs[0].invert_yaxis(); axs[0].set_xlabel("Mean out-of-fold $R^2$")
    axs[0].set_title(f"a  {title} generalization", loc="left", fontweight="bold")
    for i,v in enumerate(s.mean_oof_r2): axs[0].text(v+.008,i,f"{v:.2f}",va="center",fontsize=6)
    axs[1].barh(range(6),s.n_positive,color=[COLORS[k] for k in order]); axs[1].set_yticks([]); axs[1].invert_yaxis()
    axs[1].set_xlim(0,s.n_outcomes.iloc[0]); axs[1].set_xlabel("Outcomes with $R^2>0$"); axs[1].set_title("b  Positive predictions",loc="left",fontweight="bold")
    for i,v in enumerate(s.n_positive): axs[1].text(v+.15,i,f"{int(v)}/{int(s.n_outcomes.iloc[0])}",va="center",fontsize=6)
    axs[2].barh(range(6),s.n_total_fdr,color=[COLORS[k] for k in order]); axs[2].set_yticks([]); axs[2].invert_yaxis()
    axs[2].set_xlim(0,s.n_outcomes.iloc[0]); axs[2].set_xlabel("Total models with FDR < 0.05"); axs[2].set_title("c  Spatial inference",loc="left",fontweight="bold")
    for i,v in enumerate(s.n_total_fdr): axs[2].text(v+.15,i,f"{int(v)}/{int(s.n_outcomes.iloc[0])}",va="center",fontsize=6)
    save(fig,name,(7.2,3.2)); s.reset_index().to_csv(FIG/f"{name}_source_data.csv",index=False)
    return s

def plot_ph_heatmap(ph, title, name, height):
    x=perf[perf.phenotype.eq(ph)].pivot(index="outcome",columns="branch",values="oof_r2").reindex(columns=order)
    x=x.loc[x.max(axis=1).sort_values(ascending=False).index]
    fig,ax=plt.subplots(constrained_layout=True)
    sns.heatmap(x,cmap="RdBu_r",center=0,vmin=-.8,vmax=.8,linewidths=.25,linecolor="white",cbar_kws={"label":"Out-of-fold $R^2$","shrink":.75},ax=ax)
    ax.set_xticklabels(labels,rotation=35,ha="right"); ax.set_ylabel(""); ax.set_xlabel(""); ax.set_title(title,loc="left",fontweight="bold")
    save(fig,name,(7.2,height)); x.to_csv(FIG/f"{name}_source_data.csv")

def plot_shap_selected(selected, title, name, width):
    fig,axs=plt.subplots(1,len(selected),constrained_layout=True)
    axs=np.atleast_1d(axs)
    for ax,(ph,outcome) in zip(axs,selected):
        subset=shap[(shap.phenotype==ph)&(shap.outcome==outcome)&(shap.branch.isin(main))]
        rank=subset.groupby("feature").relative_shap.mean().nlargest(8).index
        mat=subset[subset.feature.isin(rank)].pivot(index="feature",columns="branch",values="relative_shap").reindex(rank).reindex(columns=main)
        sns.heatmap(mat*100,cmap="YlOrBr",vmin=0,cbar=ax is axs[-1],linewidths=.25,linecolor="white",cbar_kws={"label":"Relative OOF SHAP (%)","shrink":.7},ax=ax)
        ax.set_xticklabels(["Ratio","Ratio–CLR","Density"],rotation=35,ha="right"); ax.tick_params(axis="y",labelsize=6,pad=2)
        ax.set_ylabel(""); ax.set_xlabel(""); ax.set_title(outcome.upper() if outcome=="adhd" else outcome.capitalize(),fontweight="bold")
    fig.suptitle(title,fontweight="bold",y=1.02); save(fig,name,(width,3.8))
    shap[(shap.branch.isin(main)) & shap.apply(lambda r:(r.phenotype,r.outcome) in selected,axis=1)].to_csv(FIG/f"{name}_source_data.csv",index=False)

meg_summary=plot_ph_overview("meg","MEG","figure_meg_branch_overview")
plot_ph_heatmap("meg","MEG frequency-band prediction","figure_meg_oof_heatmap",3.4)
plot_shap_selected([("meg","alpha")],"MEG subclass contributions","figure_meg_shap",3.4)
enigma_summary=plot_ph_overview("enigma","ENIGMA","figure_enigma_branch_overview")
plot_ph_heatmap("enigma","ENIGMA disease-map prediction","figure_enigma_oof_heatmap",4.5)
plot_shap_selected([("enigma","adhd"),("enigma","parkinson")],"ENIGMA subclass contributions","figure_enigma_shap",6.4)

sig_total=total[total.spin_q_bh_across_outcomes<.05]
best=perf.sort_values("oof_r2",ascending=False).groupby("branch",as_index=False).first()
rows="".join(f"<tr><td>{html.escape(BRANCHES[r.branch][1])}</td><td>{r.n_features}</td><td>{r.mean_oof_r2:.3f}</td><td>{int(r.n_positive_oof_r2)}/19</td><td>{int(r.n_total_fdr)}</td><td>{html.escape(str(r.best_oof_outcome))} ({r.best_oof_r2:.3f})</td></tr>" for r in a.reset_index().itertuples())
def summary_rows(s):
    return "".join(
        f"<tr><td>{html.escape(BRANCHES[r.branch][1])}</td><td>{r.mean_oof_r2:.3f}</td>"
        f"<td>{int(r.n_positive)}/{int(r.n_outcomes)}</td><td>{int(r.n_total_fdr)}/{int(r.n_outcomes)}</td>"
        f"<td>{html.escape(str(r.best_outcome))} ({r.best_oof_r2:.3f})</td></tr>"
        for r in s.reset_index().itertuples())
meg_rows=summary_rows(meg_summary)
enigma_rows=summary_rows(enigma_summary)

report=f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Non-laminar cell-type imaging analysis</title>
<style>body{{font:16px/1.65 Arial,sans-serif;color:#20242a;max-width:1100px;margin:36px auto;padding:0 28px}}h1,h2{{line-height:1.25}}h1{{font-size:30px}}h2{{margin-top:38px;border-bottom:1px solid #ddd;padding-bottom:6px}}.lead{{font-size:18px;color:#354052}}.note{{background:#f4f7f9;border-left:4px solid #597b9d;padding:12px 16px}}img{{width:100%;height:auto;margin:12px 0 6px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f4f5f6}}code{{background:#eee;padding:2px 4px}}.caption{{font-size:13px;color:#555}}</style></head><body>
<h1>非分层细胞组成与脑影像空间表型的关联</h1>
<p class="lead">猕猴空间转录组细胞特征经转录同源映射和跨物种图谱重标记后，与人类MEG频段及ENIGMA疾病皮层厚度效应进行统一BN空间分析。</p>
<div class="note"><b>核心结论。</b> MEG与ENIGMA分别分析后均显示cluster特征具有更高的分区外预测潜力，但具体优势结局不同。subclass维度更低，更适合主要生物学解释；CLR保留部分核心信号，但改变个别细胞类型的贡献排序。</div>
<h2>分析设计</h2><p>分析包含细胞比例、CLR转换后的细胞比例及细胞密度，并分别在subclass与cluster层级评估。所有数据映射到105个BN区域。单变量关联使用双侧Pearson相关与1000次Alexander–Bloch空间旋转，并在每个表型的细胞特征集合内进行Benjamini–Hochberg校正。总模型采用Ridge回归及空间旋转检验；本报告将MEG的6个频段与ENIGMA的13种疾病分别定义为两个多重比较家族。预测性能来自按BN脑叶分组的五折分区外预测；SHAP仅在held-out区域计算。</p>
<h2>MEG振荡频率结果</h2>
<img src="figures/figure_meg_branch_overview.png"><p class="caption"><b>图1。</b> 六个MEG频段内，各分支的平均分区外R²、正预测频段数及经过MEG频段内部FDR校正的显著总模型数。</p>
<table><thead><tr><th>分支</th><th>平均OOF R²</th><th>正预测</th><th>总模型FDR显著</th><th>最佳频段</th></tr></thead><tbody>{meg_rows}</tbody></table>
<img src="figures/figure_meg_oof_heatmap.png"><p class="caption"><b>图2。</b> 六个MEG频段的分区外预测表现。色标以零为中心，正值表示模型在held-out脑区优于均值基线。</p>
<img src="figures/figure_meg_shap.png" style="max-width:560px"><p class="caption"><b>图3。</b> Alpha频段subclass模型的前8个相对held-out SHAP贡献，对比原始比例、CLR比例和密度。</p>
<p>MEG中最稳定的信号集中于alpha频段：ratio–cluster的分区外R²为0.639，ratio–CLR–cluster为0.577。ratio与CLR均保持较高表现，说明alpha相关空间信息并非完全由组成闭合约束造成。其他频段的表现更加依赖特征定义，应作为次级结果。</p>
<h2>ENIGMA疾病皮层厚度结果</h2>
<img src="figures/figure_enigma_branch_overview.png"><p class="caption"><b>图4。</b> 13种ENIGMA疾病表型内，各分支的平均分区外R²、正预测疾病数及经过疾病家族内部FDR校正的显著总模型数。</p>
<table><thead><tr><th>分支</th><th>平均OOF R²</th><th>正预测</th><th>总模型FDR显著</th><th>最佳疾病</th></tr></thead><tbody>{enigma_rows}</tbody></table>
<img src="figures/figure_enigma_oof_heatmap.png"><p class="caption"><b>图5。</b> 13种疾病皮层厚度效应图的分区外预测表现。疾病之间的模型泛化差异明显。</p>
<img src="figures/figure_enigma_shap.png"><p class="caption"><b>图6。</b> ADHD与帕金森病subclass模型的前8个相对held-out SHAP贡献。各列在完整特征集合内归一。</p>
<p>ENIGMA中，density–cluster对帕金森病的分区外R²最高（0.717）；ratio–cluster对右颞叶癫痫的R²为0.618。ADHD在多个特征策略中保持正预测，提示其空间效应与细胞组成之间的关系相对稳健。相反，精神分裂症、双相障碍和22q等结局在部分分支中为明显负R²，说明空间显著性不能替代泛化验证。</p>
<h2>细胞类型贡献的解释</h2>
<p>SHAP描述模型在未参与训练的脑区中依赖哪些细胞类型信息；优势分析则描述全数据模型中各变量对解释度的平均分配。二者回答的问题不同，排名一致时证据更强，不一致时应避免将某一细胞类型解释为独立驱动因素。</p>
<h2>空间推断与预测的区别</h2><p>多个ENIGMA结局的总Ridge模型通过空间旋转检验，但subclass平均分区外R²仍为负。这说明模型能够捕获与空间自相关不同的总体结构，却未必稳定预测新脑区。正式解释应优先依据分区外表现，并把全数据R²、dominance及总模型spin检验作为描述性或辅助证据。</p>
<h2>数据边界</h2><ul><li>比例数据精确映射191/226个原始特征，密度数据映射212/257个；未映射部分约占总质量11.2%。</li><li>比例在删除未映射项后于D99及BN空间重新闭合；CLR使用乘法零值替换。</li><li>D99标签106、118、194不属于有效源图谱，已在重标记前剔除并记录。</li><li>cluster分析的变量数接近区域数，仍需外部数据或重复采样验证贡献排序。</li></ul>
<h2>建议</h2><ol><li>主文以subclass呈现细胞类型解释，cluster作为精细定位和补充分析。</li><li>优先报告同时具有总模型空间显著性、正分区外R²、且ratio与CLR方向一致的结局。</li><li>对重点结局增加预测值–观测值图、跨折不确定性和SHAP排名稳定性。</li><li>在映射作者确认缺失细胞类型的新版对应关系后，重新运行完整映射敏感性分析。</li></ol>
<p class="caption">生成参数：BN 105区域；1000次空间旋转；五折脑叶分组预测；随机种子42。源数据和可编辑图件位于本报告同级目录。</p></body></html>'''
(OUT/"nonlaminar_imaging_report.html").write_text(report,encoding="utf-8")

qa={"figures":{},"html":str(OUT/"nonlaminar_imaging_report.html")}
for name in ["figure_meg_branch_overview","figure_meg_oof_heatmap","figure_meg_shap","figure_enigma_branch_overview","figure_enigma_oof_heatmap","figure_enigma_shap"]:
    qa["figures"][name]={ext:(FIG/f"{name}.{ext}").stat().st_size for ext in ["png","svg","pdf","tiff"]}
(OUT/"qa_manifest.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
print(OUT/"nonlaminar_imaging_report.html")
