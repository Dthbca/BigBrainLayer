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

PATHS={
"ratio_none_subclass":RESULTS/"subclass_main_20260821"/"ratio_none_subclass",
"ratio_none_cluster":RESULTS/"cluster_secondary_20260821"/"ratio_none_cluster",
"ratio_clr_subclass":RESULTS/"subclass_main_20260821"/"ratio_clr_subclass",
"ratio_clr_cluster":RESULTS/"cluster_secondary_20260821"/"ratio_clr_cluster",
"density_none_subclass":RESULTS/"subclass_main_20260821"/"density_none_subclass",
"density_none_cluster":RESULTS/"cluster_secondary_20260821"/"density_none_cluster"}
LABEL={"ratio_none_subclass":"Subclass","ratio_none_cluster":"Cluster"}

def read(branch,name):
    x=pd.read_csv(PATHS[branch]/name); x["branch"]=branch; return x
def save(fig,name,size):
    fig.set_size_inches(*size)
    for e,k in [("png",{"dpi":300}),("svg",{}),("pdf",{}),("tiff",{"dpi":600})]: fig.savefig(FIG/f"{name}.{e}",bbox_inches="tight",facecolor="white",**k)
    plt.close(fig)
def bh(p):
    p=np.asarray(p,float); o=np.argsort(p); q=np.empty(len(p)); q[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; return np.minimum(q,1)

primary=["ratio_none_subclass","ratio_none_cluster"]
perf=pd.concat([read(b,"performance.csv") for b in primary],ignore_index=True)
total=pd.concat([read(b,"total_models.csv") for b in primary],ignore_index=True)
shap=pd.concat([read(b,"oof_shap.csv") for b in primary],ignore_index=True)
dom=pd.concat([read(b,"dominance.csv") for b in primary],ignore_index=True)

# Main Figure 1: outcome-resolved results only; no transform/density comparison.
fig,axs=plt.subplots(1,2,gridspec_kw={"width_ratios":[.75,1.35]},constrained_layout=True)
for ax,ph,title in zip(axs,["meg","enigma"],["a  MEG frequency bands","b  ENIGMA cortical-thickness maps"]):
    x=perf[perf.phenotype.eq(ph)].pivot(index="outcome",columns="branch",values="oof_r2").reindex(columns=primary)
    x=x.loc[x.max(axis=1).sort_values(ascending=False).index]
    sns.heatmap(x,cmap="RdBu_r",center=0,vmin=-.8,vmax=.8,linewidths=.4,linecolor="white",cbar=ax is axs[-1],cbar_kws={"label":"Out-of-fold $R^2$","shrink":.72},ax=ax)
    ax.set_xticklabels([LABEL[c] for c in x.columns],rotation=0); ax.set_ylabel(""); ax.set_xlabel(""); ax.set_title(title,loc="left",fontweight="bold")
save(fig,"figure_main_ratio_prediction",(7.2,4.8))
perf.to_csv(FIG/"figure_main_ratio_prediction_source_data.csv",index=False)

# Main Figure 2: focused evidence ladder for the pre-specified interpretable outcomes.
focus=[("meg","alpha","Alpha"),("meg","theta","Theta"),("enigma","adhd","ADHD"),("enigma","asd","ASD"),("enigma","epilepsy_rtle","Right TLE"),("enigma","parkinson","Parkinson")]
ev=[]
for ph,out,label in focus:
    for b in primary:
        r=perf[(perf.phenotype==ph)&(perf.outcome==out)&(perf.branch==b)].iloc[0]
        t=total[(total.phenotype==ph)&(total.branch==b)].copy(); t["q_family"]=bh(t.spin_p)
        ev.append({"phenotype":ph,"outcome":out,"label":label,"branch":b,"level":LABEL[b],"oof_r2":r.oof_r2,"q_family":float(t.loc[t.outcome.eq(out),"q_family"].iloc[0])})
ev=pd.DataFrame(ev)
fig,axs=plt.subplots(1,2,constrained_layout=True)
palette={"Subclass":"#527EB7","Cluster":"#CB624A"}
for level,dx in [("Subclass",-.11),("Cluster",.11)]:
    x=ev[ev.level.eq(level)].set_index("label").loc[[f[2] for f in focus]]
    axs[0].scatter(np.arange(6)+dx,x.oof_r2,s=34,color=palette[level],label=level,zorder=3)
    axs[1].scatter(np.arange(6)+dx,-np.log10(x.q_family),s=34,color=palette[level],label=level,zorder=3)
axs[0].axhline(0,color="#555",lw=.7); axs[0].set_ylabel("Out-of-fold $R^2$"); axs[0].set_title("a  Generalization",loc="left",fontweight="bold")
axs[1].axhline(-np.log10(.05),color="#555",lw=.7,ls="--"); axs[1].set_ylabel("$-\log_{10}$(family-specific q)"); axs[1].set_title("b  Spatial inference",loc="left",fontweight="bold"); axs[1].legend(fontsize=6)
for ax in axs: ax.set_xticks(np.arange(6),[f[2] for f in focus],rotation=35,ha="right"); ax.grid(axis="y",color="#e5e5e5",lw=.5,zorder=0)
save(fig,"figure_main_focused_evidence",(7.2,3.3)); ev.to_csv(FIG/"figure_main_focused_evidence_source_data.csv",index=False)

# Main Figure 3: ratio-only subclass contributions. Cluster candidates remain tabulated.
selected=[("meg","alpha","Alpha"),("enigma","adhd","ADHD"),("enigma","epilepsy_rtle","Right TLE"),("enigma","parkinson","Parkinson")]
sub=shap[shap.branch.eq("ratio_none_subclass")]
fig,axs=plt.subplots(1,4,constrained_layout=True)
top_tables={}
for ax,(ph,out,title) in zip(axs,selected):
    x=sub[(sub.phenotype==ph)&(sub.outcome==out)].nlargest(8,"relative_shap").sort_values("relative_shap")
    ax.barh(x.feature,x.relative_shap*100,color="#527EB7"); ax.set_title(title,fontweight="bold"); ax.set_xlabel("Relative OOF SHAP (%)"); ax.tick_params(axis="y",labelsize=6)
    c=shap[(shap.branch.eq("ratio_none_cluster"))&(shap.phenotype.eq(ph))&(shap.outcome.eq(out))].nlargest(5,"relative_shap")
    top_tables[title]=c[["feature","relative_shap"]].to_dict("records")
save(fig,"figure_main_ratio_subclass_shap",(9.2,3.5)); sub[sub.apply(lambda r:(r.phenotype,r.outcome) in [(x[0],x[1]) for x in selected],axis=1)].to_csv(FIG/"figure_main_ratio_subclass_shap_source_data.csv",index=False)

def cluster_candidates(title):
    return ", ".join(f"{html.escape(x['feature'])} ({100*x['relative_shap']:.1f}%)" for x in top_tables[title])

doc=f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Cell-type architecture of MEG and ENIGMA cortical maps</title><style>body{{font:16px/1.7 Arial,sans-serif;color:#20242a;max-width:1120px;margin:36px auto;padding:0 30px}}h1{{font-size:31px;line-height:1.2}}h2{{margin-top:42px;border-bottom:1px solid #d8dce0;padding-bottom:7px}}h3{{margin-top:27px}}img{{width:100%;height:auto;margin:13px 0 5px}}.lead{{font-size:18px;color:#3b4652}}.key{{background:#eef4f7;border-left:4px solid #386f91;padding:14px 17px}}.supp{{background:#f5f5f5;padding:13px 17px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{border-bottom:1px solid #ddd;padding:7px;text-align:left}}th{{background:#f3f5f6}}.caption{{font-size:13px;color:#555}}a{{color:#245c84}}</style></head><body>
<h1>HomoloMap实现细胞分辨率的人脑皮层影像解释</h1><p class="lead">HomoloMap将猕猴空间转录组细胞组成映射为人脑同源细胞图谱，并在统一的Brainnetome空间中连接MEG振荡和ENIGMA疾病皮层厚度图。主分析使用重标记后重新闭合的细胞比例，分别在23个subclass和71个cluster层级进行验证。</p>
<div class="key"><b>主结论。</b> HomoloMap的优势来自转录同源映射产生的层级化细胞图谱：191个成功映射的原始细胞类型可在71个cluster和23个subclass两个尺度上解释人脑影像。cluster提供更精细的候选细胞定位，subclass提供较低维的系统级解释；两者均以空间整体检验、脑叶分区外预测和held-out SHAP逐层限制过度解释。MEG中Alpha的cluster结果最稳定；ENIGMA中ADHD、右颞叶癫痫和帕金森病主要获得subclass支持，ASD获得cluster层面的空间与泛化支持。</div>
<h2>1. 主分析设计</h2><p>原始细胞比例经精确转录同源映射、D99至BN跨物种重标记，并在重标记后重新闭合。subclass提供较低维、较稳健的细胞系统解释；cluster保留更细的候选亚群信息。空间总体拟合与泛化验证被明确分开：前者使用标准化普通多元线性模型、调整后R²以及每次旋转后的完整重拟合；后者使用按脑叶划分的五折分区外Ridge预测和held-out SHAP。MEG和ENIGMA均只对通过整体空间检验的模型展示优势贡献；六个频段和十三种疾病分别进行多重比较校正。</p>
<h2>2. HomoloMap的层级化细胞解释</h2><img src="figure_draft/homolomap_nonlaminar_final_figure_draft.png"><p class="caption"><b>图1。</b> HomoloMap跨物种映射、内部细胞类型层级及MEG/ENIGMA证据链。226个原始plot类型中191个获得精确同源映射，并分别汇总为71个cluster和23个subclass；这里不存在与神经递质图谱的性能或数量对照。</p><p>HomoloMap的关键价值是把猕猴空间转录组观测转化为具有明确人类同源标签的区域细胞组成图。subclass和cluster不是两个相互竞争的数据来源，而是同一映射结果的嵌套解释尺度：subclass减少维度并强调主要细胞系统，cluster保留更精细的转录亚群差异。因此，宏观MEG或疾病皮层图不仅可以关联到宽泛的兴奋性、抑制性和非神经元系统，还可进一步定位到Chandelier、SST、VIP、L5/L6 IT及胶质相关cluster等候选群。</p><p>神经递质—MEG/ENIGMA研究仅用于参考整体模型、空间旋转和优势分析的统计组织方式，不构成本研究的基线或比较对象。为避免高维cluster模型产生虚高拟合，本报告要求整体空间检验和分区外预测分别成立，并将优势分析与held-out SHAP视为互补归因。</p>
<h2>3. Ratio主结果概览</h2><img src="figures/figure_main_ratio_prediction.png"><p class="caption"><b>图2。</b> Ratio主分析在subclass和cluster层级的分区外R²。红色表示优于held-out脑区均值基线，蓝色表示泛化不足。</p><img src="figures/figure_main_focused_evidence.png"><p class="caption"><b>图3。</b> 六个重点结局的分区外预测和总模型空间推断。虚线表示家族内部FDR q=0.05。</p>
<h2>4. MEG主结果：神经递质研究框架复核</h2><img src="figure_draft/nonlaminar_hansen_style_meg_draft.png"><p class="caption"><b>图4。</b> 标准化普通线性模型的调整后R²和空间检验、独立的脑叶分区外预测、仅对整体显著模型展示的cluster优势贡献，以及优势与held-out SHAP的排序一致性。</p><h3>空间总体拟合</h3><p>subclass的六个频段均未通过跨频段校正。cluster模型的Delta、Theta、Alpha和Gamma 1通过校正，Beta和Gamma 2未通过。该结果说明细粒度细胞组成包含与频段空间拓扑一致的信息，但完整模型拟合度不能直接代表外部泛化。</p><h3>泛化和归因</h3><p>Alpha的分区外R²最高（cluster 0.639；subclass 0.516），Theta仅在cluster层面达到中等预测；Delta虽然整体空间检验显著，但分区外R²为负，不能作为可泛化结果。优势与held-out SHAP的排序相关仅为0.12–0.30，因此Sncg、SST、Chandelier、VIP等cluster应视为候选解释，而不是稳定独立效应。</p>
<h2>5. ENIGMA主结果：神经递质研究框架复核</h2><img src="figure_draft/nonlaminar_hansen_style_enigma_draft.png"><p class="caption"><b>图5。</b> 标准化普通线性模型的调整后R²与完整重拟合空间检验、独立的脑叶分区外预测，以及仅对整体显著疾病展示的优势贡献。外圈表示13种疾病内FDR q&lt;0.05。</p><h3>空间总体拟合</h3><p>subclass层面，ADHD、ASD、左/右颞叶癫痫、抑郁、OCD、肥胖和帕金森病通过整体空间检验。cluster层面，仅22q11.2缺失、ASD、左颞叶癫痫、OCD和肥胖通过检验。优势热图据此筛选，而不是对全部疾病无条件解释。</p><h3>空间一致性与泛化的交集</h3><p>ADHD-subclass（q=0.005，OOF R²=0.552）、右颞叶癫痫-subclass（q=0.005，OOF R²=0.271）、帕金森病-subclass（q=0.015，OOF R²=0.358）以及ASD-cluster（q=0.013，OOF R²=0.355）同时具有整体空间证据和正向泛化。相反，部分模型虽具有较高调整后R²，却在分区外预测中为负，说明其空间拟合不能直接转化为可推广的细胞解释。</p><h3>优势贡献的解释边界</h3><p>subclass优势分析用于描述整体显著模型内的共享拟合分配。cluster优势采用高维增量近似，出现单一cluster占据较高比例的情况，仅作为候选集中度诊断；必须结合held-out SHAP与分区外R²，不解释为独立或因果效应。</p>
<h2>6. Ratio主分析的细胞贡献</h2><img src="figures/figure_main_ratio_subclass_shap.png"><p class="caption"><b>图6。</b> Ratio–subclass模型的held-out SHAP贡献。该层级用于正文的主要细胞系统解释。</p><table><tr><th>结局</th><th>Ratio–cluster前五个SHAP候选</th></tr><tr><td>Alpha</td><td>{cluster_candidates('Alpha')}</td></tr><tr><td>ADHD</td><td>{cluster_candidates('ADHD')}</td></tr><tr><td>右颞叶癫痫</td><td>{cluster_candidates('Right TLE')}</td></tr><tr><td>帕金森病</td><td>{cluster_candidates('Parkinson')}</td></tr></table><p>cluster候选用于提出后续实验假设，而不是替代subclass层面的稳健解释。SHAP表示模型对held-out区域预测的依赖，不代表独立细胞丰度效应。</p>
<h2>7. 文献收敛与解释边界</h2><img src="figures/figure_literature_convergence.png"><p class="caption"><b>图7。</b> Ratio主分析与既有细胞类型研究的定性收敛。该图不是统计检验。</p><p>本研究参考既有皮层图谱研究的空间整体模型、旋转检验和优势分配方法，但研究对象和生物学解释独立：HomoloMap关注的是跨物种同源细胞组成。其贡献是把宏观相关解析为可检验的细胞类型候选，而不是替代神经递质机制或证明细胞丰度具有因果作用。</p>
<h2>8. 补充敏感性分析</h2><div class="supp"><p><b>CLR：</b>仅用于检验比例闭合约束是否主导结果。Alpha、Theta、ADHD、右颞叶癫痫、ASD和帕金森病的CLR结果及贡献变化不参与主结果排序。</p><p><b>Density：</b>作为不同分母定义的连续特征敏感性分析，不与ratio共同定义主结论。density对帕金森病预测较强，但仅作为支持性发现。</p></div><img src="figures/figure_meg_branch_overview.png"><img src="figures/figure_meg_oof_heatmap.png"><img src="figures/figure_enigma_branch_overview.png"><img src="figures/figure_enigma_oof_heatmap.png"><img src="figures/figure_attribution_concordance.png"><p class="caption"><b>补充图S1–S5。</b> Ratio、CLR和density的完整敏感性比较，以及不同特征策略下SHAP与dominance排名的一致性。</p>
<h2>9. 结论与限制</h2><p>综合来看，HomoloMap使空间转录组来源的跨物种细胞组成能够在subclass和cluster两个嵌套尺度上进入人脑皮层影像分析，并从宏观空间关联中提出更精细的候选细胞类型。最可信的结果来自空间整体检验与分区外预测的交集，而非完整模型的高拟合度。该框架适合生成可验证的细胞类型假设，但当前证据仍是跨数据集空间关联。</p><ul><li>ratio精确映射191/226个原始特征，约11.3%的原始组成质量未映射；删除后进行了重新闭合。</li><li>cluster模型包含71个变量，而BN只有105个区域；正则化和分区外验证不能完全消除高维不稳定性。</li><li>subclass与cluster是同一映射结果的不同聚合尺度，不能将特征数量差异本身解释为性能优势。</li><li>文献来自不同物种、脑区和测量尺度，只支持细胞系统层面的收敛。</li></ul><p class="caption">参数：BN 105区域；1000次空间旋转；五折脑叶分组预测；随机种子42。图件提供SVG、PDF、PNG、600 dpi TIFF及源数据。</p></body></html>'''
(REPORT/"nonlaminar_imaging_report.html").write_text(doc,encoding="utf-8")
qa={"html":"nonlaminar_imaging_report.html","primary_feature":"ratio","primary_levels":["subclass","cluster"],"sensitivity":["clr","density"],"new_figures":{}}
for n in ["figure_main_ratio_prediction","figure_main_focused_evidence","figure_main_ratio_subclass_shap"]: qa["new_figures"][n]={e:(FIG/f"{n}.{e}").stat().st_size for e in ["png","svg","pdf","tiff"]}
final_base=REPORT/"figure_draft"/"homolomap_nonlaminar_final_figure_draft"
qa["final_figure"]={e:Path(f"{final_base}.{e}").stat().st_size for e in ["png","svg","pdf","tiff"]}
image_refs=["figures/figure_main_ratio_prediction.png","figures/figure_main_focused_evidence.png","figure_draft/homolomap_nonlaminar_final_figure_draft.png","figure_draft/nonlaminar_hansen_style_meg_draft.png","figure_draft/nonlaminar_hansen_style_enigma_draft.png","figures/figure_main_ratio_subclass_shap.png","figures/figure_literature_convergence.png","figures/figure_meg_branch_overview.png","figures/figure_meg_oof_heatmap.png","figures/figure_enigma_branch_overview.png","figures/figure_enigma_oof_heatmap.png","figures/figure_attribution_concordance.png"]
qa["html_image_links"]={x:(REPORT/x).exists() for x in image_refs}; qa["all_html_images_exist"]=all(qa["html_image_links"].values())
(REPORT/"primary_ratio_report_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
print(REPORT/"nonlaminar_imaging_report.html")
