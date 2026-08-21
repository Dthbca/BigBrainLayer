from pathlib import Path
import os, json, html
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(os.environ.get("NONLAMINAR_ROOT",r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging"))
RESULTS=ROOT/"results"; REPORT=ROOT/"report"; FIG=REPORT/"figures"; FIG.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,"figure.facecolor":"white"})

BRANCHES={
"ratio_none_subclass":(RESULTS/"subclass_main_20260821"/"ratio_none_subclass","Ratio · subclass"),
"ratio_clr_subclass":(RESULTS/"subclass_main_20260821"/"ratio_clr_subclass","Ratio–CLR · subclass"),
"density_none_subclass":(RESULTS/"subclass_main_20260821"/"density_none_subclass","Density · subclass"),
"ratio_none_cluster":(RESULTS/"cluster_secondary_20260821"/"ratio_none_cluster","Ratio · cluster"),
"ratio_clr_cluster":(RESULTS/"cluster_secondary_20260821"/"ratio_clr_cluster","Ratio–CLR · cluster"),
"density_none_cluster":(RESULTS/"cluster_secondary_20260821"/"density_none_cluster","Density · cluster")}

def load(name):
    z=[]
    for key,(p,label) in BRANCHES.items():
        x=pd.read_csv(p/name); x["branch"]=key; x["branch_label"]=label; z.append(x)
    return pd.concat(z,ignore_index=True)

def save(fig,name,size):
    fig.set_size_inches(*size)
    for ext,kw in [("png",{"dpi":300}),("svg",{}),("pdf",{}),("tiff",{"dpi":600})]: fig.savefig(FIG/f"{name}.{ext}",bbox_inches="tight",facecolor="white",**kw)
    plt.close(fig)

perf=load("performance.csv"); total=load("total_models.csv"); spin=load("spin.csv"); shap=load("oof_shap.csv"); dom=load("dominance.csv")

# Qualitative convergence is a literature synthesis, not a statistical score.
families=["Upper-layer\nexcitatory","Deep-layer\nprojection","SST/PVALB/VIP\ninterneurons","Oligodendroglial\nlineage","Vascular /\nother glia"]
conv=pd.DataFrame([
    [2,2,1,1,1], # ADHD
    [1,2,2,2,1], # TLE
    [2,1,2,0,1], # ASD
    [0,1,1,2,1], # PD
    [0,0,2,1,1], # alpha
    [0,1,2,0,0], # theta
],index=["ADHD","Right temporal\nlobe epilepsy","Autism spectrum\ndisorder","Parkinson disease","Alpha oscillation","Theta oscillation"],columns=families)
rationale={
"ADHD":"Analysis: L2/3, L4 and L5 IT clusters; literature: excitatory-neuron and broader neuronal enrichment.",
"Right temporal lobe epilepsy":"Analysis: L5 ET, L6 CT, SST and oligodendroglia; literature: L5/6 and L2/3 principal neurons, SST/PVALB and glial changes.",
"Autism spectrum disorder":"Analysis: SST and IT clusters; literature: upper-layer excitatory, SST and VIP vulnerability.",
"Parkinson disease":"Analysis: repeated L5 IT contribution and distributed glial signals; literature: cortical neuronal states and oligodendroglial genetic/transcriptomic involvement.",
"Alpha oscillation":"Analysis: SST/VIP/Sncg plus OPC/microvascular clusters; literature: cell-type-specific inhibitory synchronization and VIP–SST responses at theta/alpha frequencies.",
"Theta oscillation":"Analysis: SST/VIP/Lamp5/L6b; literature: frequency-specific VIP and SST circuit responses."}
conv.reset_index(names="outcome").to_csv(FIG/"figure_literature_convergence_source_data.csv",index=False)
pd.DataFrame({"outcome":list(rationale),"rationale":list(rationale.values())}).to_csv(FIG/"figure_literature_convergence_rationale.csv",index=False)

fig,ax=plt.subplots(constrained_layout=True)
cmap=mpl.colors.ListedColormap(["#F1F2F3","#A9C4D9","#2F6F9F"])
sns.heatmap(conv,cmap=cmap,vmin=0,vmax=2,linewidths=1,linecolor="white",cbar=False,ax=ax,annot=False)
for i in range(conv.shape[0]):
    for j in range(conv.shape[1]):
        v=conv.iloc[i,j]; ax.text(j+.5,i+.5,["—","Partial","Strong"][v],ha="center",va="center",fontsize=6,color="white" if v==2 else "#20242a")
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Convergence between imaging–cell mapping and prior cell-type evidence",loc="left",fontweight="bold")
ax.tick_params(axis="x",rotation=0); ax.tick_params(axis="y",rotation=0)
save(fig,"figure_literature_convergence",(7.2,3.7))

# Agreement of two attribution definitions. Low agreement is an explicit risk flag.
selected=["alpha","theta","adhd","asd","epilepsy_rtle","parkinson"]
merged=shap.merge(dom,on=["phenotype","outcome","feature","branch","branch_label"],suffixes=("_shap","_dom"))
concord=[]
for (branch,outcome),x in merged[merged.outcome.isin(selected)].groupby(["branch","outcome"]):
    rho=x.relative_shap.corr(x.relative_dominance,method="spearman")
    concord.append({"branch":branch,"outcome":outcome,"spearman_rho":rho,"n_features":len(x)})
concord=pd.DataFrame(concord)
mat=concord.pivot(index="outcome",columns="branch",values="spearman_rho").reindex(index=selected,columns=list(BRANCHES))
mat.to_csv(FIG/"figure_attribution_concordance_source_data.csv")
fig,ax=plt.subplots(constrained_layout=True)
sns.heatmap(mat,cmap="RdBu_r",center=0,vmin=-1,vmax=1,annot=True,fmt=".2f",annot_kws={"fontsize":6},linewidths=.5,linecolor="white",cbar_kws={"label":"Spearman rank correlation","shrink":.75},ax=ax)
ax.set_xticklabels([BRANCHES[k][1] for k in mat.columns],rotation=35,ha="right"); ax.set_yticklabels(["Alpha","Theta","ADHD","ASD","Right TLE","Parkinson"],rotation=0)
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Agreement between held-out SHAP and dominance rankings",loc="left",fontweight="bold")
save(fig,"figure_attribution_concordance",(7.2,3.7))

# Evidence table uses family-specific BH for reporting, while raw 19-outcome q remains in source files.
def bh(p):
    p=np.asarray(p,float); o=np.argsort(p); q=np.empty(len(p)); q[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; return np.minimum(q,1)

focus=[("meg","alpha","Alpha"),("meg","theta","Theta"),("enigma","adhd","ADHD"),("enigma","asd","ASD"),("enigma","epilepsy_rtle","Right temporal lobe epilepsy"),("enigma","parkinson","Parkinson disease")]
evidence=[]
for ph,out,label in focus:
    for branch in ["ratio_none_cluster","ratio_clr_cluster","density_none_cluster"]:
        pp=perf[(perf.phenotype==ph)&(perf.outcome==out)&(perf.branch==branch)].iloc[0]
        tt=total[(total.phenotype==ph)&(total.branch==branch)].copy(); tt["q_family"]=bh(tt.spin_p)
        q=float(tt.loc[tt.outcome.eq(out),"q_family"].iloc[0])
        n_pair=int(((spin.phenotype==ph)&(spin.outcome==out)&(spin.branch==branch)&(spin.spin_q_bh<.05)).sum())
        evidence.append({"outcome":label,"branch":BRANCHES[branch][1],"oof_r2":pp.oof_r2,"total_q_family":q,"n_pairwise_fdr":n_pair})
evidence=pd.DataFrame(evidence); evidence.to_csv(FIG/"focused_evidence_table.csv",index=False)

def mini_table(outcome):
    x=evidence[evidence.outcome.eq(outcome)]
    return "".join(f"<tr><td>{html.escape(r.branch)}</td><td>{r.oof_r2:.3f}</td><td>{r.total_q_family:.4f}</td><td>{int(r.n_pairwise_fdr)}</td></tr>" for r in x.itertuples())

links={
"adhd":"https://pmc.ncbi.nlm.nih.gov/articles/PMC9458853/",
"adhd_gwas":"https://pmc.ncbi.nlm.nih.gov/articles/10914347/",
"tle":"https://pmc.ncbi.nlm.nih.gov/articles/PMC7541486/",
"tle_glia":"https://pmc.ncbi.nlm.nih.gov/articles/PMC9590125/",
"asd":"https://pmc.ncbi.nlm.nih.gov/articles/PMC7678724/",
"pd":"https://www.nature.com/articles/s41467-024-47867-4",
"pd_genetics":"https://www.nature.com/articles/s41588-020-0610-9",
"osc":"https://pubmed.ncbi.nlm.nih.gov/36865311/",
"sync":"https://www.nature.com/articles/s41467-019-10498-1"}

doc=f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Cell-type architecture of MEG and ENIGMA cortical maps</title><style>
body{{font:16px/1.7 Arial,sans-serif;color:#20242a;max-width:1120px;margin:36px auto;padding:0 30px}}h1{{font-size:31px;line-height:1.2}}h2{{margin-top:42px;border-bottom:1px solid #d8dce0;padding-bottom:7px}}h3{{margin-top:28px}}img{{width:100%;height:auto;margin:13px 0 5px}}.lead{{font-size:18px;color:#3b4652}}.key{{background:#eef4f7;border-left:4px solid #386f91;padding:14px 17px}}.warn{{background:#fff6e8;border-left:4px solid #c4862f;padding:12px 16px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{border-bottom:1px solid #ddd;padding:7px;text-align:left}}th{{background:#f3f5f6}}.caption{{font-size:13px;color:#555}}a{{color:#245c84}}code{{background:#eee;padding:2px 4px}}</style></head><body>
<h1>非分层细胞空间结构与MEG及ENIGMA皮层表型</h1><p class="lead">本分析将猕猴空间转录组细胞特征映射为人类细胞群并重标记到统一皮层图谱，以检验细胞组成能否解释人类神经振荡和疾病相关皮层厚度效应。</p>
<div class="key"><b>总体结论。</b> cluster层级保留了subclass聚合后丢失的预测信息。MEG主要表现为Alpha和Theta与抑制性回路空间组成的关系；ENIGMA中ADHD、右颞叶癫痫和帕金森病具有较稳定的多变量预测，ASD则在比例模型中显示SST相关收敛。文献一致性支持这些结果作为细胞系统层面的解释，但不足以证明单个cluster的因果作用。</div>
<h2>1. 证据框架与分析边界</h2><p>报告将证据分为三层：首先以按脑叶划分的五折分区外R²评估空间泛化；其次以1000次Alexander–Bloch旋转检验总模型是否超出空间自相关背景；最后使用held-out SHAP和dominance描述细胞群贡献。MEG的6个频段与ENIGMA的13种疾病分别进行多重比较校正。</p><div class="warn"><b>解释边界：</b>全数据总模型显著不等于能够预测新脑区；SHAP和dominance排名不一致时，只报告细胞系统而不指定唯一驱动cluster。</div>
<h2>2. MEG：Alpha与Theta形成主要振荡结果</h2><img src="figures/figure_meg_branch_overview.png"><img src="figures/figure_meg_oof_heatmap.png"><p class="caption"><b>图1。</b> MEG六频段的分支总体表现和结局级分区外R²。正R²表示优于held-out脑区的均值基线。</p>
<h3>Alpha</h3><table><tr><th>Cluster特征</th><th>OOF R²</th><th>MEG内部总模型q</th><th>显著单cluster数</th></tr>{mini_table("Alpha")}</table><p>Alpha在ratio和CLR模型中保持较高预测，主要贡献涉及SST、VIP、Sncg及部分OPC/微血管相关cluster。VIP与SST细胞对theta/alpha频率输入表现出不同反应，SST亚型也具有不同的网络同步耦合特性，因此结果与抑制性回路调节节律活动的实验发现相符（<a href="{links['osc']}">频率特异性VIP–SST回路</a>；<a href="{links['sync']}">细胞亚型与皮层同步</a>）。但原始ratio总模型仅接近MEG家族FDR阈值，具体cluster归因仍应视为候选。</p>
<h3>Theta</h3><table><tr><th>Cluster特征</th><th>OOF R²</th><th>MEG内部总模型q</th><th>显著单cluster数</th></tr>{mini_table("Theta")}</table><p>Theta的预测幅度低于Alpha，但ratio与CLR总模型均通过空间检验。SST、VIP、Lamp5与L6b反复出现，支持其与抑制性调控及跨层回路有关；不同解释方法的cluster排名变化提示该结果更适合在细胞类别层面讨论。</p><img src="figures/figure_meg_shap.png" style="max-width:570px"><p class="caption"><b>图2。</b> Alpha的subclass held-out SHAP贡献。CLR列是相对于完整组成的对数比贡献。</p>
<h2>3. ENIGMA：疾病间存在明显异质性</h2><img src="figures/figure_enigma_branch_overview.png"><img src="figures/figure_enigma_oof_heatmap.png"><p class="caption"><b>图3。</b> ENIGMA 13种疾病的分支总体表现和结局级分区外R²。</p>
<h3>ADHD：跨策略最稳健</h3><table><tr><th>Cluster特征</th><th>OOF R²</th><th>ENIGMA内部总模型q</th><th>显著单cluster数</th></tr>{mini_table("ADHD")}</table><p>ADHD在ratio、CLR和density中均获得正预测及显著总模型，贡献涉及L2/3、L4和L5 IT细胞群。该结果与ADHD风险基因富集于皮层兴奋性神经元、特别是上层和深层兴奋性神经元的研究一致（<a href="{links['adhd']}">细胞类型富集</a>；<a href="{links['adhd_gwas']}">大型GWAS</a>）。胶质cluster贡献可作为联合机制候选，但证据弱于兴奋性神经元。</p>
<h3>右颞叶癫痫：深层投射回路与胶质支持</h3><table><tr><th>Cluster特征</th><th>OOF R²</th><th>ENIGMA内部总模型q</th><th>显著单cluster数</th></tr>{mini_table("Right temporal lobe epilepsy")}</table><p>L5 ET、L6 CT、SST和Oligo相关cluster共同出现，与人类颞叶癫痫中L5/6投射神经元、L2/3兴奋性神经元及SST/PVALB改变相符（<a href="{links['tle']}">神经元亚型研究</a>），也与OPC和少突胶质状态改变的研究一致（<a href="{links['tle_glia']}">胶质转录组研究</a>）。</p>
<h3>ASD：比例模型中的SST收敛</h3><table><tr><th>Cluster特征</th><th>OOF R²</th><th>ENIGMA内部总模型q</th><th>显著单cluster数</th></tr>{mini_table("ASD")}</table><p>Sst_5在ratio与CLR模型中较稳定，并伴有IT细胞群贡献。人类ASD单核研究同样指出上层兴奋性神经元以及SST/VIP抑制性神经元受影响（<a href="{links['asd']}">ASD单细胞研究</a>）。由于density模型泛化较弱，该结果应限定为比例组成层面的收敛。</p>
<h3>帕金森病：强多变量预测，但缺乏单cluster显著性</h3><table><tr><th>Cluster特征</th><th>OOF R²</th><th>ENIGMA内部总模型q</th><th>显著单cluster数</th></tr>{mini_table("Parkinson disease")}</table><p>Density模型达到最高预测表现，L5 IT在多种策略的dominance中反复出现，但没有单cluster通过pairwise FDR。既有研究支持皮层神经元状态及少突胶质细胞参与帕金森病（<a href="{links['pd']}">皮层单核研究</a>；<a href="{links['pd_genetics']}">细胞类型遗传研究</a>），但尚不能确认L5 IT是特异疾病细胞群。</p><img src="figures/figure_enigma_shap.png"><p class="caption"><b>图4。</b> ADHD与帕金森病的subclass held-out SHAP贡献。</p>
<h2>4. 与既有细胞类型文献的收敛</h2><img src="figures/figure_literature_convergence.png"><p class="caption"><b>图5。</b> 定性文献收敛矩阵。“Strong”表示本分析与至少一项直接细胞类型研究在同一细胞系统上收敛；“Partial”表示仅在较宽细胞类别或跨物种回路层面一致。该矩阵不是统计检验。</p>
<h2>5. 贡献解释的稳定性</h2><img src="figures/figure_attribution_concordance.png"><p class="caption"><b>图6。</b> held-out SHAP与dominance细胞群排名的Spearman相关。高一致性支持相对稳定的特征排序；低一致性说明归因依赖解释定义。</p><p>贡献排名的一致性在疾病、频段和特征表示之间变化明显。因此，正文优先报告跨ratio/CLR/density重复出现的细胞系统；单个cluster名称适合作为后续实验候选，而不应被描述为已确立的病理细胞。</p>
<h2>6. 推荐的正文与补充材料结构</h2><ol><li><b>正文主结果：</b>ADHD、右颞叶癫痫、帕金森病、Alpha及Theta；ASD作为比例组成支持结果。</li><li><b>正文解释层级：</b>以subclass和宽细胞系统解释，cluster用于定位候选亚群。</li><li><b>补充材料：</b>全部13种疾病、6个频段、所有pairwise spin结果、完整SHAP和dominance排名。</li><li><b>后续验证：</b>外部细胞映射、不同空间折叠、映射缺失补全及cluster排名的bootstrap稳定性。</li></ol>
<h2>7. 数据限制</h2><ul><li>ratio精确映射191/226个原始特征，density映射212/257个；约11.2%的原始质量未映射。</li><li>ratio删除未映射项后在D99和BN空间重新闭合；CLR使用乘法零值替换。</li><li>cluster模型含71–72个变量，而BN仅105个区域；正则化和分区外验证降低但不能消除高维不稳定性。</li><li>文献证据来自不同物种、脑区、疾病阶段和测量尺度，只支持收敛解释，不构成因果验证。</li></ul>
<p class="caption">分析参数：BN 105区域；1000次空间旋转；按脑叶五折分区外预测；随机种子42。全部定量图均提供源数据、SVG、PDF、PNG和600 dpi TIFF。</p></body></html>'''
(REPORT/"nonlaminar_imaging_integrated_literature_report.html").write_text(doc,encoding="utf-8")
qa={"html":"nonlaminar_imaging_integrated_literature_report.html","new_figures":{},"evidence_rows":len(evidence),"literature_matrix_is_qualitative":True}
for n in ["figure_literature_convergence","figure_attribution_concordance"]: qa["new_figures"][n]={e:(FIG/f"{n}.{e}").stat().st_size for e in ["png","svg","pdf","tiff"]}
(REPORT/"integrated_literature_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
print(REPORT/"nonlaminar_imaging_integrated_literature_report.html")
