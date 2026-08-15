"""Correct exact whole-match and unique-target mismatch permutation p-values."""
from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def main():
    p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(a.results/'branch_summary.csv')
    mismatch=pd.read_csv(a.results/'source_data'/'all_branch_mismatch_results.csv')
    corrected=[]; whole_rows=[]
    for row in old.itertuples(index=False):
        f=a.results/'source_data'/f'{row.branch}__whole_null.csv'
        null=pd.read_csv(f).iloc[:,0].dropna().to_numpy(float)
        observed=float(row.whole_match_stat)
        extreme=int(np.sum(null>=observed)); n=len(null)
        p_exact=extreme/n
        whole_rows.append({'branch':row.branch,'observed_stat':observed,'n_exact':n,
                           'extreme_count':extreme,'p_old_plus1':float(row.whole_match_p),
                           'p_correct_exact':p_exact})
    whole=pd.DataFrame(whole_rows)
    out=old.merge(whole[['branch','p_correct_exact']],on='branch')
    out['whole_match_p_original']=out.whole_match_p
    out['whole_match_p']=out.pop('p_correct_exact')

    mm=mismatch.copy(); mm['p_value_original']=mm.p_value
    corrected_parts=[]; mismatch_audit=[]
    n_layers=6; repeated_per_target=math.factorial(n_layers-1)
    for branch,d in mm.groupby('branch',sort=False):
        d=d.copy()
        if not (d.n_perm_used==repeated_per_target*(n_layers-1)).all():
            raise RuntimeError(f'unexpected mismatch permutation count in {branch}')
        numerator=d.p_value_original*(d.n_perm_used+1)-1
        k=np.rint(numerator/repeated_per_target).astype(int)
        if not np.allclose(numerator,k*repeated_per_target,atol=1e-7):
            raise RuntimeError(f'cannot recover unique extremes in {branch}')
        d['unique_mismatch_extreme_count']=k
        d['p_value']=(k+1)/n_layers
        d['n_unique_mismatch']=n_layers-1
        reject,q,_,_=multipletests(d.p_value,method='fdr_bh')
        d['p_adjusted']=q; d['reject_H0']=reject
        corrected_parts.append(d)
        mismatch_audit.append({'branch':branch,'n_tests':len(d),'min_exact_p':float(d.p_value.min()),
                               'n_fdr_original':int((d.p_adjusted_original<.05).sum()) if 'p_adjusted_original' in d else None,
                               'n_fdr_corrected':int(reject.sum())})
    mmc=pd.concat(corrected_parts,ignore_index=True)
    counts=mmc.groupby('branch').reject_H0.sum().astype(int)
    out['mismatch_n_fdr_sig_original']=out.mismatch_n_fdr_sig
    out['mismatch_n_fdr_sig']=out.branch.map(counts).fillna(0).astype(int)
    out.to_csv(a.output/'branch_summary_corrected.csv',index=False)
    whole.to_csv(a.output/'whole_match_correction.csv',index=False)
    mmc.to_csv(a.output/'mismatch_results_corrected.csv',index=False)
    audit={'issue_whole':'Exact enumeration included identity but also used +1 correction.',
           'issue_mismatch':'600 permutations represented only five unique mismatch targets repeated 120 times.',
           'whole_formula':'extreme / 720','mismatch_formula':'(1 + unique mismatches as/extreme as observed) / 6',
           'mismatch_min_p':1/6,'mismatch':mismatch_audit}
    (a.output/'permutation_correction_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(whole.to_string(index=False)); print(out[['branch','whole_match_p_original','whole_match_p','mismatch_n_fdr_sig_original','mismatch_n_fdr_sig']].to_string(index=False))

if __name__=='__main__': main()
