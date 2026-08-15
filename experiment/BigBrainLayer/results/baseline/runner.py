import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from HomoloMap.datasets.layers import (load_layer_counts, normalize_layer_composition,
    relabel_layer_counts, fetch_bigbrain_layer_thickness, fetch_laminar_mask,
    LAYER_KEYS, LAYER_LABELS)
from HomoloMap.transforms.layers import make_layer_subcompositions
from HomoloMap.stats import SpinTest, layer_spin_correlation, layer_match_permutation

DATA_ROOT=Path('/share/user_data/dthbca/public/experiment/BigBrainLayer/dataset')
OUT=Path(os.environ['LAYER_COMPARE_OUT']); OUT.mkdir(parents=True,exist_ok=True)
SOURCE=OUT/'source_data'; SOURCE.mkdir(exist_ok=True)
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
SEED=42; N_SPINS=1000; N_JOBS=int(os.environ.get('LAYER_N_JOBS','8'))
ALIASES=dict(zip(LAYER_KEYS,LAYER_LABELS))

def log(msg): print(time.strftime('%F %T'),msg,flush=True)
def savefig(fig,name):
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',dpi=600,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.tiff',dpi=600,bbox_inches='tight')
    plt.close(fig)

params={'seed':SEED,'n_spins':N_SPINS,'spin_method':'Alexander-Bloch','n_jobs':N_JOBS,
        'metric':'pearsonr','correction':'fdr_bh','mapping_column':'subclass','unmapped':'drop',
        'source_atlas':'D99','target_atlas':'BN','continuous_relabel_method':'mean',
        'normalizations':['within_layer','within_region_cross_layer'],
        'use_clr':[False,True],'thickness':['absolute','relative']}
(OUT/'parameters.json').write_text(json.dumps(params,indent=2,ensure_ascii=False),encoding='utf-8')

try:
    log('LOAD raw D99 counts')
    raw,mapping=load_layer_counts(DATA_ROOT,source_atlas='D99',mapping_column='subclass',unmapped='drop',return_mapping=True)
    branches_data={}
    relabel_audits={}
    for norm in ['within_layer','within_region_cross_layer']:
        mode='within_layer' if norm=='within_layer' else 'within_region'
        normalized=normalize_layer_composition(raw,mode=mode,zero_policy='zero')
        mapped,audit=relabel_layer_counts(normalized,'D99','BN',method='mean',cross_species=True,
                                          unknown_labels='drop',return_audit=True)
        branches_data[norm]=mapped; relabel_audits[norm]=audit
        log(f'PREP {norm} shapes={ {k:v.shape for k,v in mapped.items()} } dropped={audit["dropped_labels"]}')
    regions=next(iter(branches_data.values()))['l1'].index
    ctypes=next(iter(branches_data.values()))['l1'].columns
    present,mask_audit=fetch_laminar_mask('external',ctypes,data_dir=DATA_ROOT)
    thick_abs=fetch_bigbrain_layer_thickness('BN',DATA_ROOT,False,regions=regions)
    thick_rel=fetch_bigbrain_layer_thickness('BN',DATA_ROOT,True,regions=regions)
    thick_abs.to_csv(SOURCE/'thickness_absolute.csv'); thick_rel.to_csv(SOURCE/'thickness_relative.csv')
    present.to_csv(SOURCE/'present_mask.csv')
    audit_json={'mapping':{k:(v.tolist() if hasattr(v,'tolist') else str(v)) for k,v in mapping.items() if k!='mapping_coverage'},
                'relabel':relabel_audits,'mask':mask_audit}
    def conv(o):
        if isinstance(o,(np.integer,np.floating)): return o.item()
        if isinstance(o,np.ndarray): return o.tolist()
        if isinstance(o,pd.Series): return o.to_dict()
        if isinstance(o,pd.DataFrame): return o.to_dict(orient='records')
        if isinstance(o,Path): return str(o)
        raise TypeError(type(o).__name__)
    (OUT/'audit.json').write_text(json.dumps(audit_json,default=conv,indent=2,ensure_ascii=False),encoding='utf-8')
    log('BUILD shared spinner n=1000')
    spinner=SpinTest(atlas='BN',n_spins=N_SPINS,method='Alexander-Bloch',seed=SEED)
    all_results={}; summaries=[]; match_rows=[]
    for norm in params['normalizations']:
      base=branches_data[norm]
      for clr in [False,True]:
        features,_=make_layer_subcompositions(base,present,transform='clr' if clr else 'none',
                                               zero_method='multiplicative')
        for tname,thickness in [('absolute',thick_abs),('relative',thick_rel)]:
          branch=f'{norm}__clr_{str(clr).lower()}__thickness_{tname}'
          log(f'BRANCH START {branch}')
          res=layer_spin_correlation(features,thickness,spinner,present_mask=present,
                                     metric='pearsonr',correction='fdr_bh',n_jobs=N_JOBS)
          res.insert(0,'branch',branch); res.to_csv(OUT/f'{branch}.csv',index=False)
          all_results[branch]=res
          whole=layer_match_permutation(features,thickness,scheme='whole',alternative='greater',
                                        random_state=SEED,n_permutations=None)
          mismatch=layer_match_permutation(features,thickness,scheme='mismatch',alternative='greater',
                                           correction='fdr_bh',random_state=SEED,n_permutations=None)
          mismatch.insert(0,'branch',branch); mismatch.to_csv(OUT/f'{branch}__mismatch.csv',index=False)
          pd.DataFrame({'null':whole['null_distribution']}).to_csv(SOURCE/f'{branch}__whole_null.csv',index=False)
          r=res.correlation.dropna(); sig=res[res.p_adjusted<.05]
          summaries.append({'branch':branch,'normalization':norm,'use_clr':clr,'thickness':tname,
             'n_tested':len(res),'n_fdr_sig':len(sig),'median_abs_r':r.abs().median(),
             'mean_abs_r':r.abs().mean(),'mean_signed_r':r.mean(),'max_abs_r':r.abs().max(),
             'whole_match_stat':whole['observed_stat'],'whole_match_p':whole['p_value'],
             'whole_n_permutations':whole['n_permutations'],'mismatch_n_fdr_sig':int((mismatch.p_adjusted<.05).sum())})
          match_rows.append(mismatch)
          top=res.assign(abs_r=res.correlation.abs()).sort_values('abs_r',ascending=False).head(15)
          top.to_csv(OUT/f'{branch}__top.csv',index=False)
          log(f'BRANCH DONE {branch} tested={len(res)} sig={len(sig)} meanabs={r.abs().mean():.4f} whole_p={whole["p_value"]:.5f}')
    summary=pd.DataFrame(summaries).sort_values(['n_fdr_sig','mean_abs_r'],ascending=False)
    summary.to_csv(OUT/'branch_summary.csv',index=False)
    long=pd.concat(all_results.values(),ignore_index=True); long.to_csv(SOURCE/'all_branch_spin_results.csv',index=False)
    pd.concat(match_rows,ignore_index=True).to_csv(SOURCE/'all_branch_mismatch_results.csv',index=False)
    # Robustness matrices on common layer/ctype keys
    wide=long.pivot_table(index=['layer','ctype'],columns='branch',values='correlation')
    rho=wide.corr(method='spearman'); rho.to_csv(SOURCE/'branch_spearman_r.csv')
    branches=list(summary.branch); jac=pd.DataFrame(index=branches,columns=branches,dtype=float)
    sigsets={b:set(map(tuple,all_results[b].loc[all_results[b].p_adjusted<.05,['layer','ctype']].values)) for b in branches}
    for a in branches:
      for b in branches:
        union=sigsets[a]|sigsets[b]; jac.loc[a,b]=len(sigsets[a]&sigsets[b])/len(union) if union else 1.0
    jac.to_csv(SOURCE/'branch_significant_jaccard.csv')
    # Figures
    sns.set_theme(style='white',context='paper')
    fig,axs=plt.subplots(2,2,figsize=(12,8)); order=summary.branch.tolist(); s=summary.set_index('branch').loc[order]
    sns.barplot(x=s.index,y=s.n_fdr_sig,ax=axs[0,0],color='#4477AA'); axs[0,0].set_title('FDR-significant tests');
    sns.barplot(x=s.index,y=s.mean_abs_r,ax=axs[0,1],color='#CC6677'); axs[0,1].set_title('Mean |r|')
    sns.barplot(x=s.index,y=s.whole_match_stat,ax=axs[1,0],hue=(s.whole_match_p<.05),palette={True:'#228833',False:'#BBBBBB'},legend=False); axs[1,0].set_title('Whole-match statistic')
    sns.barplot(x=s.index,y=-np.log10(s.whole_match_p),ax=axs[1,1],color='#AA3377'); axs[1,1].axhline(-np.log10(.05),ls='--',c='k',lw=.8); axs[1,1].set_title('-log10 whole-match p')
    for ax in axs.flat: ax.tick_params(axis='x',rotation=75,labelsize=6); ax.set_xlabel('')
    fig.tight_layout(); savefig(fig,'hero_branch_comparison')
    layer_metric=long.assign(abs_r=long.correlation.abs()).pivot_table(index='branch',columns='layer',values='abs_r',aggfunc='mean').loc[order]
    layer_metric.to_csv(SOURCE/'branch_layer_mean_abs_r.csv')
    fig,ax=plt.subplots(figsize=(8,5)); sns.heatmap(layer_metric,cmap='mako',annot=True,fmt='.2f',ax=ax); ax.set_title('Mean |r| by branch and layer'); fig.tight_layout(); savefig(fig,'branch_layer_heatmap')
    fig,ax=plt.subplots(figsize=(8,7)); sns.heatmap(rho.loc[order,order],vmin=-1,vmax=1,cmap='vlag',square=True,ax=ax); ax.set_title('Spearman stability of effects'); fig.tight_layout(); savefig(fig,'branch_spearman_matrix')
    best=summary.iloc[0].branch; br=all_results[best]; rmat=br.pivot(index='layer',columns='ctype',values='correlation'); pmat=br.pivot(index='layer',columns='ctype',values='p_adjusted')
    rmat.to_csv(SOURCE/'best_branch_r_matrix.csv'); pmat.to_csv(SOURCE/'best_branch_p_matrix.csv')
    fig,ax=plt.subplots(figsize=(13,4)); sns.heatmap(rmat,cmap='vlag',vmin=-1,vmax=1,center=0,ax=ax,cbar_kws={'label':'Pearson r'})
    for i in range(rmat.shape[0]):
      for j in range(rmat.shape[1]):
        p=pmat.iloc[i,j]
        if np.isfinite(p) and p<.05: ax.text(j+.5,i+.5,'***' if p<.001 else '**' if p<.01 else '*',ha='center',va='center',fontsize=6)
    ax.set_title(f'Best branch: {best}'); fig.tight_layout(); savefig(fig,'best_branch_layer_ctype')
    # Chinese report
    bestrow=summary.iloc[0]
    report=f'''# HomoloMap 层分支比较报告\n\n## 方法\n\n数据在 D99 同一空间先归一，再以 `method="mean"` 将连续特征映射至 BN（未 round，未用计数 sum）。比较 2 种归一、CLR 开关和绝对/相对厚度，共 8 分支。空间置换为 Alexander-Bloch，1000 spins，seed=42，Pearson，FDR-BH。whole/mismatch 均精确枚举 6!=720 层排列。\n\n## 数据与审计\n\n- 原始层数：6；映射后 BN ROI：{len(regions)}；cell types：{len(ctypes)}。\n- mapping unresolved：{mapping.get('n_unresolved_types')}。\n- D99→BN 显式丢弃 atlas 外 ROI：106、118、194；详见 `audit.json`。\n- external present mask 有效组合：{int(present.values.sum())}。\n\n## 结果\n\n最佳分支（先按 FDR 显著数，再按 mean|r|）：`{bestrow.branch}`。n_tested={bestrow.n_tested}，n_fdr_sig={bestrow.n_fdr_sig}，mean|r|={bestrow.mean_abs_r:.4f}，median|r|={bestrow.median_abs_r:.4f}，max|r|={bestrow.max_abs_r:.4f}，whole-match statistic={bestrow.whole_match_stat:.4f}，p={bestrow.whole_match_p:.6f}。\n\n完整结果见 `branch_summary.csv`、各分支 CSV 和 `source_data/`。\n\n## 稳健性\n\n分支间效应 Spearman 相关见 `source_data/branch_spearman_r.csv`；FDR 显著集合 Jaccard 见 `source_data/branch_significant_jaccard.csv`。\n\n## 限制\n\n结果依赖 D99→BN 最近质心/体素映射、外部 laminar mask、cell-type 同源映射和空间 null。CLR 系数代表层内 subcomposition 的相对 log-ratio；非 CLR 分支不应解释为独立绝对效应。8 分支属于敏感性分析，不能以最佳分支替代预注册主分析。\n'''
    (OUT/'report_zh.md').write_text(report,encoding='utf-8')
    (OUT/'SUCCESS').write_text(time.strftime('%F %T'),encoding='utf-8')
    log(f'ALL DONE best={best}')
except Exception:
    (OUT/'FAILED').write_text(traceback.format_exc(),encoding='utf-8')
    traceback.print_exc(); raise
