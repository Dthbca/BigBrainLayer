import json, os, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PKG=Path('/share/user_data/dthbca/public/experiment/BigBrainLayer/staging/HomoloMap_20260811_001313')
DATA=Path('/share/user_data/dthbca/public/experiment/BigBrainLayer/dataset')
ROOT=Path('/share/user_data/dthbca/public/experiment/BigBrainLayer/results/homolomap_layer_branch_compare_20260812_060115')
OUT=ROOT/'validation'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(PKG.parent))
from HomoloMap.datasets.layers import load_layer_counts, normalize_layer_composition, relabel_layer_counts, fetch_bigbrain_layer_thickness
from HomoloMap.datasets import fetch_annot
from HomoloMap.transforms.atlas import load_volume_atlas

raw,mapping=load_layer_counts(DATA,mapping_column='subclass',unmapped='drop',return_mapping=True)
wl=normalize_layer_composition(raw,mode='within_layer',zero_policy='zero')
wr=normalize_layer_composition(raw,mode='within_region',zero_policy='zero')
raw_stack=np.stack([raw[k].to_numpy(float) for k in raw])
wr_stack=np.stack([wr[k].to_numpy(float) for k in wr])
wl_sums=np.stack([wl[k].sum(axis=1).to_numpy(float) for k in wl])
wl_positive=np.stack([raw[k].sum(axis=1).to_numpy(float)>0 for k in raw])
cross_sums=wr_stack.sum(axis=0); cross_positive=raw_stack.sum(axis=0)>0
wl_err=float(np.max(np.abs(wl_sums[wl_positive]-1)))
cross_err=float(np.max(np.abs(cross_sums[cross_positive]-1)))
wlm,wl_audit=relabel_layer_counts(wl,'D99','BN',method='mean',return_audit=True)
wrm,wr_audit=relabel_layer_counts(wr,'D99','BN',method='mean',return_audit=True)
counts_sum,count_audit=relabel_layer_counts(raw,'D99','BN',method='sum',return_audit=True)
bn_path,bn_info=fetch_annot(atlas='BN',annot=True)
bn=load_volume_atlas(bn_path,bn_info,hemisphere='left')
regions=wlm['l1'].index
ta=fetch_bigbrain_layer_thickness('BN',DATA,False,regions=regions)
tr=fetch_bigbrain_layer_thickness('BN',DATA,True,regions=regions)
params=json.loads((ROOT/'parameters.json').read_text())
summary=pd.read_csv(ROOT/'branch_summary.csv')
branch_csv=sorted(p for p in ROOT.glob('*.csv') if '__thickness_' in p.name and '__' not in p.stem.split('__thickness_',1)[1])
expected=[]
for n in ['within_layer','within_region_cross_layer']:
 for c in ['false','true']:
  for t in ['absolute','relative']: expected.append(f'{n}__clr_{c}__thickness_{t}')
rows={p.stem:len(pd.read_csv(p)) for p in branch_csv}
all_long=pd.read_csv(ROOT/'source_data'/'all_branch_spin_results.csv')
rho=pd.read_csv(ROOT/'source_data'/'branch_spearman_r.csv',index_col=0)
jac=pd.read_csv(ROOT/'source_data'/'branch_significant_jaccard.csv',index_col=0)
fig_required=[]
for stem in ['hero_branch_comparison','branch_layer_heatmap','branch_spearman_matrix','best_branch_layer_ctype']:
 for ext in ['png','pdf','svg','tiff']: fig_required.append(ROOT/'figures'/f'{stem}.{ext}')
files=[]
for p in sorted(ROOT.rglob('*')):
 if p.is_file() and 'validation' not in p.parts:
  files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size})
audit={
 'status':'PASS','existing_run_reused':True,'no_spin_rerun':True,
 'package':str(PKG),'python':sys.executable,
 'branch_keys_expected':expected,'branch_keys_summary':summary.branch.tolist(),
 'branch_keys_match':set(expected)==set(summary.branch),'per_branch_rows':rows,
 'all_branches_97_tests':set(rows)==set(expected) and all(v==97 for v in rows.values()),
 'long_rows':len(all_long),'long_expected':8*97,
 'raw_shapes':{k:list(v.shape) for k,v in raw.items()},
 'within_layer_max_abs_error_positive_denominator':wl_err,
 'cross_layer_per_roi_ctype_max_abs_error_positive_denominator':cross_err,
 'normalization_invariants_pass':wl_err<1e-12 and cross_err<1e-12,
 'mapped_shapes_mean':{'within_layer':{k:list(v.shape) for k,v in wlm.items()},'cross_layer':{k:list(v.shape) for k,v in wrm.items()}},
 'mapped_finite':all(np.isfinite(v.to_numpy()).all() for z in [wlm,wrm] for v in z.values()),
 'mapped_common_roi_exact':all(v.index.equals(regions) for z in [wlm,wrm,counts_sum] for v in z.values()),
 'n_effective_roi':len(regions),'n_celltypes':len(wlm['l1'].columns),
 'bn_left_n_labels':len(bn['roi_labels']),'bn_left_labels_exact':np.array_equal(regions.to_numpy(),np.asarray(bn['roi_labels'])),
 'd99_dropped_labels':wl_audit['dropped_labels'],
 'counts_sum_audit':{'method':count_audit['method'],'shapes':{k:list(v.shape) for k,v in counts_sum.items()},'finite':all(np.isfinite(v.to_numpy()).all() for v in counts_sum.values())},
 'thickness_absolute':{'shape':list(ta.shape),'finite':bool(np.isfinite(ta.to_numpy()).all()),'row_sum_mean':float(ta.sum(axis=1).mean()),'relative_flag':False},
 'thickness_relative':{'shape':list(tr.shape),'finite':bool(np.isfinite(tr.to_numpy()).all()),'row_sum_max_abs_error':float(np.max(np.abs(tr.sum(axis=1)-1))),'relative_flag':True},
 'spin_parameters':{k:params[k] for k in ['n_spins','seed','spin_method','n_jobs','metric','correction']},
 'spin_parameter_pass':params['n_spins']==1000 and params['seed']==42 and params['spin_method']=='Alexander-Bloch',
 'summary_n_fdr_sig':dict(zip(summary.branch,summary.n_fdr_sig.astype(int))),
 'summary_mean_abs_r':dict(zip(summary.branch,summary.mean_abs_r)),
 'summary_whole_match_p':dict(zip(summary.branch,summary.whole_match_p)),
 'branch_spearman_offdiag':{'min':float(rho.to_numpy()[~np.eye(len(rho),dtype=bool)].min()),'median':float(np.median(rho.to_numpy()[~np.eye(len(rho),dtype=bool)])),'max':float(rho.to_numpy()[~np.eye(len(rho),dtype=bool)].max())},
 'significant_jaccard_offdiag':{'min':float(jac.to_numpy()[~np.eye(len(jac),dtype=bool)].min()),'median':float(np.median(jac.to_numpy()[~np.eye(len(jac),dtype=bool)])),'max':float(jac.to_numpy()[~np.eye(len(jac),dtype=bool)].max())},
 'report_exists_nonempty':(ROOT/'report_zh.md').stat().st_size>0,
 'figures_required_complete':all(p.exists() and p.stat().st_size>0 for p in fig_required),
 'success_marker':(ROOT/'SUCCESS').exists(),'failed_marker':(ROOT/'FAILED').exists(),
 'files':files
}
checks=['branch_keys_match','all_branches_97_tests','normalization_invariants_pass','mapped_finite','mapped_common_roi_exact','bn_left_labels_exact','spin_parameter_pass','figures_required_complete','success_marker']
audit['status']='PASS' if all(audit[x] for x in checks) and not audit['failed_marker'] else 'FAIL'
(OUT/'validation_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
pd.DataFrame(files).to_csv(OUT/'file_manifest.csv',index=False)
pd.DataFrame({'branch':summary.branch,'n_fdr_sig':summary.n_fdr_sig,'mean_abs_r':summary.mean_abs_r,'whole_match_p':summary.whole_match_p}).to_csv(OUT/'quantitative_summary.csv',index=False)
print(json.dumps({k:audit[k] for k in ['status','branch_keys_match','all_branches_97_tests','normalization_invariants_pass','within_layer_max_abs_error_positive_denominator','cross_layer_per_roi_ctype_max_abs_error_positive_denominator','mapped_finite','n_effective_roi','n_celltypes','bn_left_labels_exact','spin_parameters','figures_required_complete','success_marker']},indent=2))
