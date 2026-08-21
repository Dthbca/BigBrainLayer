from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns


ROOT = Path(os.environ.get(
    "NONLAMINAR_ROOT",
    r"D:\HomoloMap\projects\imaging_integration\NonLaminarImaging",
))
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
OUT = REPORT / "figure_draft"
OUT.mkdir(parents=True, exist_ok=True)

BRANCHES = {
    "subclass": RESULTS / "subclass_main_20260821" / "ratio_none_subclass",
    "cluster": RESULTS / "cluster_secondary_20260821" / "ratio_none_cluster",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
})

BLUE = "#3977A8"
ORANGE = "#D2674A"
TEAL = "#3B8C88"
GREY = "#66737C"
LIGHT = "#EEF3F5"


def read(level, filename):
    frame = pd.read_csv(BRANCHES[level] / filename)
    frame["level"] = level
    return frame


total = pd.concat([read(level, "total_models.csv") for level in BRANCHES])
perf = pd.concat([read(level, "performance.csv") for level in BRANCHES])
shap = pd.concat([read(level, "oof_shap.csv") for level in BRANCHES])
evidence = total.merge(
    perf,
    on=["phenotype", "outcome", "level"],
    validate="one_to_one",
)
evidence["neglog10_q"] = -np.log10(evidence["spin_q_bh_across_outcomes"].clip(lower=1e-12))
evidence["convergent"] = (
    (evidence["spin_q_bh_across_outcomes"] < 0.05)
    & (evidence["oof_r2"] > 0)
)


def pretty_outcome(value):
    labels = {
        "delta": "Delta", "theta": "Theta", "alpha": "Alpha",
        "beta": "Beta", "gamma1": "Gamma 1", "gamma2": "Gamma 2",
        "22q": "22q11.2", "adhd": "ADHD", "asd": "ASD",
        "epilepsy_gge": "Generalized epilepsy",
        "epilepsy_rtle": "Right TLE", "epilepsy_ltle": "Left TLE",
        "depression": "Depression", "ocd": "OCD",
        "schizophrenia": "Schizophrenia", "bipolar": "Bipolar",
        "obesity": "Obesity", "schizotypy": "Schizotypy",
        "parkinson": "Parkinson",
    }
    return labels.get(value, value)


def evidence_panel(ax, phenotype, title):
    data = evidence[evidence.phenotype.eq(phenotype)].copy()
    colors = {"subclass": BLUE, "cluster": ORANGE}
    markers = {"subclass": "o", "cluster": "s"}
    for level in ["subclass", "cluster"]:
        x = data[data.level.eq(level)]
        ax.scatter(
            x.oof_r2, x.neglog10_q,
            s=30, color=colors[level], marker=markers[level],
            alpha=0.84, label=level.capitalize(), zorder=3,
        )
        hit = x[x.convergent]
        ax.scatter(
            hit.oof_r2, hit.neglog10_q,
            s=62, facecolors="none", edgecolors="#1F2529",
            marker=markers[level], linewidths=0.8, zorder=4,
        )
    # Direct labels are restricted to the strongest non-duplicated predictive
    # outcomes; the source-data table retains every screened model.
    label_limit = 3 if phenotype == "enigma" else 4
    labelled = (
        data[data.convergent]
        .sort_values("oof_r2", ascending=False)
        .drop_duplicates("outcome")
        .head(label_limit)
    )
    offsets = [(4, 6), (4, -12), (-5, 7), (-5, -11)]
    for (idx, row), offset in zip(labelled.iterrows(), offsets):
        ax.annotate(
            pretty_outcome(row.outcome),
            (row.oof_r2, row.neglog10_q),
            xytext=offset, textcoords="offset points", fontsize=5.2,
            ha="left" if offset[0] > 0 else "right",
        )
    ax.axvline(0, color="#69757D", lw=0.7)
    ax.axhline(-np.log10(0.05), color="#69757D", lw=0.7, ls="--")
    ax.set_xlabel("Lobe-wise out-of-fold $R^2$")
    ax.set_ylabel(r"Spatial evidence, $-\log_{10}(q)$")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8)
    ax.grid(color="#E4E8EA", lw=0.45, zorder=0)


fig = plt.figure(figsize=(7.2, 7.0), constrained_layout=True)
grid = fig.add_gridspec(3, 2, height_ratios=[0.72, 1.0, 1.15])

# a | compact study logic
ax = fig.add_subplot(grid[0, :])
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.0, 0.98, "a", fontweight="bold", fontsize=9, va="top")
boxes = [
    (0.04, 0.38, 0.20, 0.40, "Macaque spatial\ntranscriptomics", "226 source classes"),
    (0.29, 0.38, 0.20, 0.40, "Homologous mapping", "191 mapped"),
    (0.54, 0.38, 0.20, 0.40, "Human BN maps", "23 subclasses | 71 clusters"),
    (0.80, 0.55, 0.16, 0.23, "MEG", "6 frequency bands"),
    (0.80, 0.22, 0.16, 0.23, "ENIGMA", "13 disorders"),
]
for i, (x, y, w, h, title, subtitle) in enumerate(boxes):
    face = "#E7F0F4" if i < 3 else "#F7EEE9"
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.014",
        fc=face, ec="#53636D", lw=0.75,
    ))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontweight="bold", fontsize=6.4)
    ax.text(x + w / 2, y + h * 0.22, subtitle, ha="center", va="center",
            color=GREY, fontsize=5.2)
for start, end in [((0.245, 0.58), (0.285, 0.58)), ((0.495, 0.58), (0.535, 0.58))]:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                 color="#53636D", lw=0.75))
ax.plot([0.745, 0.77], [0.58, 0.58], color="#53636D", lw=0.75)
ax.plot([0.77, 0.77], [0.335, 0.665], color="#53636D", lw=0.75)
for y in [0.335, 0.665]:
    ax.add_patch(FancyArrowPatch((0.77, y), (0.795, y), arrowstyle="-|>",
                                 mutation_scale=8, color="#53636D", lw=0.75))
ax.text(
    0.42, 0.10,
    "Spatially controlled association  →  blocked prediction  →  held-out attribution",
    ha="center", fontsize=6.2, fontweight="bold", color="#304B5F",
)

# b-c | discovery and generalization on the same coordinate system
ax_meg = fig.add_subplot(grid[1, 0])
evidence_panel(ax_meg, "meg", "b  MEG: spatial evidence and generalization")
ax_enigma = fig.add_subplot(grid[1, 1])
evidence_panel(ax_enigma, "enigma", "c  ENIGMA: spatial evidence and generalization")
handles, labels = ax_enigma.get_legend_handles_labels()
ax_enigma.legend(handles, labels, loc="upper left", fontsize=5.6)

# d | attribution only for models satisfying both gates
ax = fig.add_subplot(grid[2, :])
selected = evidence[evidence.convergent & evidence.level.eq("cluster")].sort_values(
    ["phenotype", "oof_r2"], ascending=[True, False]
)
selected = selected.groupby("phenotype", group_keys=False).head(4)
columns = []
feature_pool = []
for row in selected.itertuples():
    label = f"{pretty_outcome(row.outcome)}\n({row.level})"
    columns.append((row.phenotype, row.outcome, row.level, label))
    subset = shap[
        shap.phenotype.eq(row.phenotype)
        & shap.outcome.eq(row.outcome)
        & shap.level.eq(row.level)
    ]
    feature_pool.extend(subset.nlargest(3, "relative_shap").feature.tolist())
features = list(dict.fromkeys(feature_pool))
matrix = pd.DataFrame(index=features)
for phenotype, outcome, level, label in columns:
    subset = shap[
        shap.phenotype.eq(phenotype)
        & shap.outcome.eq(outcome)
        & shap.level.eq(level)
    ].set_index("feature")
    matrix[label] = subset.relative_shap.reindex(features) * 100
matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
matrix = matrix.head(18)
plot_matrix = matrix.T
sns.heatmap(
    plot_matrix, ax=ax, cmap="YlGnBu", vmin=0, linewidths=0.3, linecolor="white",
    cbar_kws={"label": "Relative held-out SHAP (%)", "shrink": 0.75},
)
ax.set_title("d  Fine cluster attribution after spatial and predictive screening",
             loc="left", fontweight="bold", fontsize=8)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=42, ha="right", fontsize=5.0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5.2)

base = OUT / "homolomap_evidence_figure_v2"
fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{base}.svg", bbox_inches="tight", facecolor="white")
fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)

evidence.to_csv(OUT / "homolomap_evidence_source_data.csv", index=False)
matrix.to_csv(OUT / "homolomap_attribution_source_data.csv")

convergent = evidence[evidence.convergent].copy()
convergent["outcome_label"] = convergent.outcome.map(pretty_outcome)
meg_hits = convergent[convergent.phenotype.eq("meg")]
enigma_hits = convergent[convergent.phenotype.eq("enigma")]

def hit_rows(frame):
    if frame.empty:
        return "<tr><td colspan='5'>No model passed both gates.</td></tr>"
    rows = []
    for row in frame.sort_values("oof_r2", ascending=False).itertuples():
        rows.append(
            f"<tr><td>{row.outcome_label}</td><td>{row.level}</td>"
            f"<td>{row.oof_r2:.3f}</td><td>{row.spin_q_bh_across_outcomes:.4f}</td>"
            f"<td>{row.oof_pearson:.3f}</td></tr>"
        )
    return "".join(rows)


def complete_rows(phenotype):
    rows = []
    frame = evidence[evidence.phenotype.eq(phenotype)].copy()
    frame["outcome_label"] = frame.outcome.map(pretty_outcome)
    for row in frame.sort_values(["outcome", "level"]).itertuples():
        decision = "进入归因" if row.convergent else "未通过双门槛"
        rows.append(
            f"<tr><td>{row.outcome_label}</td><td>{row.level}</td>"
            f"<td>{row.full_r2:.3f}</td><td>{row.spin_p:.4f}</td>"
            f"<td>{row.spin_q_bh_across_outcomes:.4f}</td><td>{row.oof_r2:.3f}</td>"
            f"<td>{row.oof_pearson:.3f}</td><td>{decision}</td></tr>"
        )
    return "".join(rows)


spin_frames = []
for level in BRANCHES:
    frame = read(level, "spin.csv")
    spin_frames.append(frame)
spin_all = pd.concat(spin_frames, ignore_index=True)
pairwise_summary = (
    spin_all.assign(significant=spin_all.spin_q_bh < 0.05)
    .groupby(["phenotype", "level"], as_index=False)
    .agg(n_tests=("feature", "size"), n_fdr_significant=("significant", "sum"))
)


def pairwise_rows():
    rows = []
    for row in pairwise_summary.itertuples():
        rows.append(
            f"<tr><td>{row.phenotype.upper()}</td><td>{row.level}</td>"
            f"<td>{row.n_tests}</td><td>{row.n_fdr_significant}</td></tr>"
        )
    return "".join(rows)


audit_path = RESULTS / "audit_summary.csv"
audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()


def audit_rows():
    if audit.empty:
        return "<tr><td colspan='7'>Audit summary unavailable.</td></tr>"
    rows = []
    for row in audit.itertuples():
        rows.append(
            f"<tr><td>{row.branch}</td><td>{int(row.n_features)}</td>"
            f"<td>{row.mapping_fraction:.3f}</td><td>{row.unmapped_mass_fraction:.3f}</td>"
            f"<td>{row.mean_oof_r2:.3f}</td><td>{int(row.n_positive_oof_r2)}</td>"
            f"<td>{int(row.n_total_fdr)}</td></tr>"
        )
    return "".join(rows)

caption = (
    "Figure 1 | HomoloMap links homologous cortical cell composition to human MEG and disease-related cortical maps. "
    "a, Macaque spatial-transcriptomic cell classes were mapped to homologous human cell identities and relabelled to 105 Brainnetome regions. "
    "b,c, Each point combines lobe-wise out-of-fold prediction with the Benjamini–Hochberg-adjusted spatial spin-test result for the same ridge model. "
    "Open symbols identify models with q<0.05 and out-of-fold R²>0; dashed and solid lines mark these two gates. "
    "d, Relative mean absolute SHAP values for the fine cluster resolution were calculated only from held-out regions and are shown only for screened models. "
    "Subclass and cluster are nested resolutions of the same homologous mapping. SHAP quantifies predictive dependence and does not establish an independent abundance effect or causality."
)
(OUT / "homolomap_evidence_figure_v2_caption.txt").write_text(caption, encoding="utf-8")

html = f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'>
<title>HomoloMap与人脑皮层影像的细胞组成关联</title>
<style>
body{{font:16px/1.72 Arial,'Microsoft YaHei',sans-serif;color:#20262b;max-width:1100px;margin:36px auto;padding:0 30px}}
h1{{font-size:31px;line-height:1.22;margin-bottom:10px}}h2{{font-size:23px;margin-top:40px;border-bottom:1px solid #d7dde0;padding-bottom:7px}}
h3{{font-size:18px;margin-top:26px}}img{{width:100%;height:auto;margin:12px 0 5px}}.lead{{font-size:18px;color:#3e4a52}}
.key{{background:#edf4f7;border-left:4px solid #3977a8;padding:14px 18px;margin:18px 0}}
.note{{background:#f4f5f5;padding:12px 16px}}table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{border-bottom:1px solid #dfe3e5;padding:7px;text-align:left}}th{{background:#f1f4f5}}.caption{{font-size:13px;color:#565f65}}
</style></head><body>
<h1>HomoloMap揭示人脑皮层影像的同源细胞组成结构</h1>
<p class='lead'>本分析检验由猕猴空间转录组推断的人类同源细胞组成，是否能够解释正常皮层振荡和疾病相关皮层厚度变化。主分析使用重标记后重新闭合的细胞比例；23个subclass用于较低维的系统解释，71个cluster用于精细候选定位。</p>
<div class='key'><b>核心结论。</b> HomoloMap产生的细胞图谱包含与MEG和ENIGMA空间模式一致的信息，但可信解释必须同时满足两个条件：总体模型在保留空间自相关的旋转检验中显著，并且对未参与训练的脑叶区域具有正向预测能力。细胞归因仅用于通过这两个筛选条件的模型。</div>

<h2>1. 研究问题与证据链</h2>
<p>研究对象是HomoloMap细胞图谱在两个嵌套分辨率上的解释能力。分析依次回答三个问题：细胞组成与影像表型是否具有超出空间平滑结构的整体关联；这种关系能否推广到留出的脑区；在可推广模型中，哪些细胞类群对预测贡献最大。</p>
<img src='figure_draft/homolomap_evidence_figure_v2.png'>
<p class='caption'><b>图1 | HomoloMap连接同源皮层细胞组成与人类MEG及疾病相关皮层图。</b> a，猕猴空间转录组细胞类型经转录同源映射并重标记到105个Brainnetome皮层区域。b、c，每个点同时表示同一Ridge模型的脑叶分区外预测R²和Benjamini–Hochberg校正后的空间旋转检验结果；空心外圈表示q&lt;0.05且分区外R²&gt;0，虚线与实线分别表示两个筛选阈值。d，仅对通过双门槛的精细cluster模型展示held-out区域的平均绝对SHAP相对贡献。subclass和cluster是同一同源映射的嵌套分辨率。SHAP表示预测依赖，不构成独立丰度效应或因果证据。</p>

<h2>2. 数据处理与统计设计</h2>
<h3>2.1 特征构建与空间对齐</h3><p>226个来源细胞类型中，191个依据转录同源关系获得精确映射，并聚合为23个subclass和71个cluster。未映射成分约占原始比例质量的11.3%；主分析删除未映射列后先在D99来源空间重新闭合，再执行D99至Brainnetome的跨物种区域重标记，并在目标空间再次闭合。每个Brainnetome区域的mapped ratio因此和为1。该变量表示“已成功映射细胞类型之间的相对组成”，不是完整组织中所有细胞的绝对比例。</p>
<h3>2.2 单变量空间关联</h3><p>每个细胞类型—影像结局组合计算Pearson相关。零假设不是普通的独立同分布随机性，而是“在保留皮层空间自相关的情况下不存在位置对应关系”。为此使用Alexander–Bloch球面旋转产生1,000个空间替代图。每个分析分支内，对同一影像结局的全部细胞类型p值使用Benjamini–Hochberg方法校正；因此该q值回答的是该结局内部所有细胞类型比较形成的发现率控制。</p>
<h3>2.3 总体模型空间检验</h3><p>subclass或cluster同时进入带截距Ridge模型。观测统计量为完整数据模型R²；正则化强度由交叉验证选择。空间零分布通过旋转影像结局获得，并在每次旋转中使用固定分析流程重新拟合模型。经验p值按(extreme+1)/(1,000+1)计算，避免得到零p值。随后分别在每个分支的6个MEG频段或13个ENIGMA疾病之间实施Benjamini–Hochberg校正。该检验回答“整个细胞组成是否携带超出空间平滑结构的联合信息”，不用于评估未见区域的预测能力。</p>
<h3>2.4 分区外预测</h3><p>泛化能力通过五折脑叶分组交叉验证评估：同一脑叶的区域整体进入训练或测试集合，降低相邻区域同时出现在两侧造成的空间泄漏。每一外层训练折内独立估计标准化参数，并在训练数据内部选择Ridge正则化强度；测试折不参与任何预处理或参数选择。汇总全部held-out预测后计算OOF R²、Pearson相关和平均绝对误差。OOF R²&gt;0表示优于以测试观测总体均值为基准的平方误差预测，负值则表示泛化不足。</p>
<h3>2.5 归因筛选与多重比较</h3><p>只有总体模型空间q&lt;0.05且OOF R²&gt;0的组合进入正文归因。线性SHAP在每个外层模型的held-out区域计算，随后汇总绝对贡献并标准化为相对比例；因此不存在使用同一观察同时训练和解释的训练内SHAP。优势分析用于将完整模型R²在相关特征之间分配：subclass维度可作较稳定的系统解释，71维cluster的增量近似只作为候选排序。MEG与ENIGMA是两个独立多重比较家族，不合并校正；subclass与cluster也作为预先定义的不同解释分辨率分别校正。</p>
<div class='note'><b>为什么不把普通最小二乘调整后R²放在主图：</b> cluster模型具有71个变量而只有约105个区域，训练内拟合容易偏高。调整后R²可作为方法复核，但不能替代空间零模型或留出泛化证据。</div>

<h2>3. MEG：细胞组成与皮层振荡</h2>
<p>MEG分析检验六个频段的区域功率图。按主分析的正则化总体模型，cluster层面有多个频段通过空间检验，其中Theta、Beta和Gamma 1同时获得正向脑叶分区外预测；subclass模型未形成相同强度的总体空间证据。该模式提示细粒度转录亚群能够分辨部分频率特异的空间组织，但并非所有训练内拟合较高的频段都具有泛化能力。</p>
<table><tr><th>频段</th><th>细胞分辨率</th><th>分区外R²</th><th>空间q</th><th>预测相关</th></tr>{hit_rows(meg_hits)}</table>
<p>这些结果支持“频率相关的细胞组成结构”这一空间统计解释，但不说明细胞比例直接产生某一振荡频率。SHAP所标记的SST、VIP、Chandelier或其他精细cluster是模型依赖的候选来源，需要独立数据验证。</p>

<h2>4. ENIGMA：细胞组成与疾病相关皮层厚度</h2>
<p>ENIGMA分析以13种疾病的皮层厚度效应图作为结局。ADHD、颞叶癫痫、帕金森病及部分其他表型在至少一个细胞分辨率上同时获得空间证据和正向分区外预测。subclass结果适合描述较宽的细胞系统；cluster结果则进一步区分同一系统内部的转录亚群。</p>
<table><tr><th>疾病</th><th>细胞分辨率</th><th>分区外R²</th><th>空间q</th><th>预测相关</th></tr>{hit_rows(enigma_hits)}</table>
<p>疾病效应图来自独立的大规模病例—对照研究，因此这里的关系应解释为不同数据源之间的空间收敛。它能够提出与疾病皮层易感性相关的细胞候选，但不能区分细胞组成是病因、结果还是共同空间梯度的标记。</p>

<h2>5. 细胞贡献的解释原则</h2>
<p>总体优势分析回答完整模型的解释度如何在相关特征之间分配；held-out SHAP回答每个细胞特征对未见脑区预测的平均贡献。两者受到特征共线性和组成约束影响，均不等同于某一细胞类型的独立生物学效应。正文优先报告subclass层面的稳定系统模式，并将cluster作为更精细、可实验验证的候选列表。</p>

<h2>6. 敏感性分析与结果边界</h2>
<ul>
<li><b>CLR：</b>检验闭合约束是否主导结果，放入补充材料；CLR坐标的贡献必须解释为相对其余组成的对数比依赖。</li>
<li><b>Density：</b>采用不同分母定义，作为连续特征敏感性分析，不与比例结果合并形成主结论。</li>
<li><b>分辨率：</b>subclass与cluster来自同一映射层级，特征数量更多不等于解释更优；比较应依据空间证据、留出预测和贡献稳定性。</li>
<li><b>样本与空间：</b>约105个区域限制了71变量模型的稳定性。后续应增加重复空间分区、报告预测区间，并在独立细胞图谱或影像数据中复现。</li>
</ul>

<h2>7. 可能的文献支持证据</h2>
<p><b>MEG空间组织。</b> Hansen等整合多种神经递质受体图谱后发现，皮层受体空间分布能够解释六个经典MEG频段的区域功率，并使用空间旋转、距离约束交叉验证和优势分析限制空间自相关与共线性的影响。这一工作支持“皮层微观分子或细胞构成与振荡拓扑相关”的一般框架，但其预测变量是受体密度而不是HomoloMap细胞比例，因此只能支持分析策略和跨尺度假设，不能视为本研究结果的直接复现（<a href='https://www.nature.com/articles/s41593-022-01186-3'>Hansen et al., Nature Neuroscience, 2022</a>）。独立MEG研究还显示，静息态峰值频率沿皮层层级呈系统性空间梯度，并与皮层厚度梯度相关，提示本研究必须采用保留空间结构的零模型，而不能使用普通参数检验（<a href='https://pubmed.ncbi.nlm.nih.gov/32820722/'>Mahjoory et al., eLife, 2020</a>）。</p>
<p><b>ENIGMA疾病皮层厚度。</b> ENIGMA的virtual histology研究发现，六类精神疾病的区域皮层厚度差异与锥体细胞、星形胶质细胞和小胶质细胞相关基因表达图具有空间对应关系，三类细胞表达共同解释了25%–54%的区域差异。这为疾病皮层形态具有细胞类型相关空间组织提供支持，也与本研究在神经元和胶质cluster中观察到候选贡献相容；但该研究使用基因表达代理而非直接细胞比例，而且分析疾病集合与本研究并不完全相同（<a href='https://pubmed.ncbi.nlm.nih.gov/32857118/'>Patel et al., JAMA Psychiatry, 2021</a>）。22q11.2缺失综合征的影像转录组研究进一步展示了区域皮层异常与基因表达空间收敛可用于候选基因优先级排序，支持把当前结果定位为候选生成而非病因证明（<a href='https://pubmed.ncbi.nlm.nih.gov/33638978/'>Forsyth et al., Molecular Psychiatry, 2021</a>）。</p>
<p><b>跨物种同源映射。</b> 多灵长类单核转录组研究表明，大多数转录定义的细胞亚型跨人、黑猩猩、猕猴和狨猴具有可识别的保守性，同时也存在物种特异亚型和同源类型内部的分子差异。这同时支持HomoloMap利用转录同源关系进行映射的生物学基础，并界定其边界：同源标签不意味着空间比例或全部分子状态在人与猕猴之间完全相同（<a href='https://pubmed.ncbi.nlm.nih.gov/36007006/'>Ma et al., Science, 2022</a>）。猕猴空间多组学研究也证明分子状态可以在皮层区域中定位，并揭示人—猕猴胶质轨迹的相似性与差异，为空间细胞图谱的跨物种比较提供进一步依据（<a href='https://pubmed.ncbi.nlm.nih.gov/36347848/'>Zhu et al., Nature Neuroscience, 2022</a>）。</p>
<div class='note'><b>证据强度边界：</b>上述文献支持研究问题的合理性、空间统计方法和细胞类型解释的可行性，但没有独立验证本研究的具体cluster排序。正文应使用“与……一致”“支持……假设”或“提供生物学背景”，避免使用“证实机制”或“重复发现”。</div>

<h1 style='margin-top:60px'>Part 2｜完整中间结果与补充材料</h1>
<h2>P2.1 映射、闭合与分支审计</h2>
<table><tr><th>分支</th><th>特征数</th><th>映射比例</th><th>未映射质量</th><th>平均OOF R²</th><th>正OOF结局数</th><th>总体FDR显著数</th></tr>{audit_rows()}</table>
<p>该表用于确认每个分支使用的映射覆盖、特征维度和整体行为。ratio-none是正文主分析；ratio-CLR与density不参与正文结果排序。</p>

<h2>P2.2 Ratio主分析的全部总体模型结果</h2>
<img src='figures/figure_main_ratio_prediction.png'><p class='caption'><b>补充图S1。</b> Ratio主分析中MEG与ENIGMA全部结局的空间总体检验和脑叶分区外预测。图形同时显示空间q值、OOF R²及进入归因的双门槛结果。</p>
<img src='figures/figure_main_focused_evidence.png'><p class='caption'><b>补充图S2。</b> 通过双门槛的主要结局及其预测证据，用于区分空间显著但不能推广的模型与具有留出预测能力的模型。</p>
<details><summary>展开MEG全部频段数值</summary><table><tr><th>结局</th><th>分辨率</th><th>完整R²</th><th>spin p</th><th>家族q</th><th>OOF R²</th><th>OOF r</th><th>决策</th></tr>{complete_rows('meg')}</table></details>
<details><summary>展开ENIGMA全部疾病数值</summary><table><tr><th>结局</th><th>分辨率</th><th>完整R²</th><th>spin p</th><th>家族q</th><th>OOF R²</th><th>OOF r</th><th>决策</th></tr>{complete_rows('enigma')}</table></details>
<p>“进入归因”严格表示总体q&lt;0.05且OOF R²&gt;0，并不等于因果性或临床预测有效性。</p>

<h2>P2.3 单细胞类型空间关联的完整性摘要</h2>
<table><tr><th>表型家族</th><th>分辨率</th><th>检验数</th><th>结局内FDR显著数</th></tr>{pairwise_rows()}</table>
<img src='figures/figure_main_ratio_subclass_shap.png'><p class='caption'><b>补充图S3。</b> Ratio主分析在subclass层面的held-out SHAP贡献。仅展示通过空间总体检验与正向留出预测双门槛的模型，贡献表示模型对未见脑区预测的依赖。</p>
<p>每个单变量结果的相关系数、spin p、结局内BH q和有效旋转次数保存在各主分支的 <code>spin.csv</code>。单变量显著性用于描述空间对应，不能替代总体模型或留出预测。</p>

<h2>P2.4 CLR与density敏感性</h2>
<img src='figures/figure_meg_branch_overview.png'><p class='caption'><b>补充图S4。</b> MEG在ratio、CLR和density分支中的总体与预测结果概览。</p>
<img src='figures/figure_enigma_branch_overview.png'><p class='caption'><b>补充图S5。</b> ENIGMA在ratio、CLR和density分支中的总体与预测结果概览。</p>
<p>CLR回答结果是否依赖闭合组成的绝对坐标；density回答更接近细胞丰度尺度的连续特征是否产生相似结果。二者的分母和解释单位不同，因此不与ratio合并计算总体显著性，也不根据三种表示中最小p值选择结论。</p>

<h2>P2.5 预测、优势分析与SHAP完整结果</h2>
<img src='figures/figure_attribution_concordance.png'><p class='caption'><b>补充图S6。</b> 不同分析分支中优势贡献与held-out SHAP排序的一致性。低一致性提示相关特征之间的贡献分配具有方法依赖性。</p>
<p>每个分支均保存 <code>performance.csv</code>、<code>folds.csv</code>、<code>oof_predictions.csv</code>、<code>total_models.csv</code>、<code>dominance.csv</code>和<code>oof_shap.csv</code>。正文只展示双门槛通过的归因；完整特征排序应作为Source Data或补充数据表提供。</p>

<h2>P2.6 建议的补充材料清单</h2>
<table><tr><th>补充项目</th><th>内容</th><th>目的</th></tr>
<tr><td>Supplementary Methods</td><td>映射、闭合、空间旋转、多重比较与分区外预测</td><td>集中说明完整统计流程</td></tr>
<tr><td>Supplementary Fig. S1–S6</td><td>全部结局、CLR/density敏感性及归因一致性</td><td>用图呈现完整性与稳健性</td></tr>
<tr><td>Supplementary Data 1</td><td>单变量spin、总体模型、OOF预测及fold定义</td><td>支持结果与统计复核</td></tr>
<tr><td>Supplementary Data 2</td><td>全部dominance与held-out SHAP贡献</td><td>支持细胞候选复核</td></tr></table>
<p class='caption'>分析范围：Brainnetome 105个皮层区域；1,000次空间旋转；五折脑叶分组预测；主特征为重标记后重新闭合的mapped ratio。结果是空间关联和预测性解释，不是因果推断。</p>
</body></html>"""

(REPORT / "nonlaminar_imaging_report.html").write_text(html, encoding="utf-8")
qa = {
    "core_conclusion": "HomoloMap cell composition is interpretable only where spatial inference and blocked prediction converge.",
    "backend": "Python",
    "main_figure": {ext: Path(f"{base}.{ext}").stat().st_size for ext in ["png", "svg", "pdf", "tiff"]},
    "source_data": ["homolomap_evidence_source_data.csv", "homolomap_attribution_source_data.csv"],
    "n_convergent_meg": int(len(meg_hits)),
    "n_convergent_enigma": int(len(enigma_hits)),
    "html_images_exist": (OUT / "homolomap_evidence_figure_v2.png").exists(),
    "statistics": {
        "spatial": "1,000 Alexander-Bloch spins; BH within six MEG or thirteen ENIGMA outcomes per branch",
        "prediction": "five-fold lobe-wise out-of-fold Ridge",
        "attribution": "mean absolute held-out linear SHAP",
    },
    "review_risks": [
        "105 regions relative to 71 cluster predictors",
        "approximately 11.3% source ratio mass was unmapped",
        "spatial association and SHAP are not causal effects",
    ],
}
(REPORT / "evidence_led_report_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(REPORT / "nonlaminar_imaging_report.html")
print(base)
