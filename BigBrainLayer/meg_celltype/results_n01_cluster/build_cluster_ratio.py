import numpy as np
import pandas as pd
from CellAlign.datasets.atlas import ctype_ratio_agg
from CellAlign.parcellation import surf_relabel

FGC_RAW = "/data100/home/dthbca/project/CellAlign/CellAlign/datasets/features/SpatialTranscriptomics/ctype_ratio_plot_FGC.csv"
CMAP = "/data100/home/dthbca/project/Macaque_ST/notebook/cluster_mapping_dict.csv"
SUBCLASS_REF = "/data100/home/dthbca/project/CellAlign/tmp/subclass_ratio.csv"
OUT = "/data100/home/dthbca/project/CellAlign/tmp/cluster_ratio.csv"

fgc = pd.read_csv(FGC_RAW, index_col=0)
print("FGC raw:", fgc.shape, "index[:5]:", list(fgc.index[:5]))

# ---- 1. VERIFY recipe: reproduce subclass_ratio.csv via FGC->BN ----
sub_fgc = ctype_ratio_agg(fgc, key='subclass')       # uses ctype_map.csv
sub_bn = surf_relabel(src='FGC', trg='BN', data=sub_fgc, method='mean')
ref = pd.read_csv(SUBCLASS_REF, index_col=0)
print("\n[VERIFY] reproduced subclass BN:", sub_bn.shape, "ref:", ref.shape)
print("index match:", list(sub_bn.index) == list(ref.index))
common = [c for c in ref.columns if c in sub_bn.columns]
diffs = []
for c in common:
    a = sub_bn[c].reindex(ref.index).values
    b = ref[c].values
    m = np.isfinite(a) & np.isfinite(b)
    r = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 2 else np.nan
    maxabs = np.nanmax(np.abs(a[m] - b[m])) if m.sum() else np.nan
    diffs.append((c, r, maxabs))
print("cols reproduced:", len(common), "/", ref.shape[1])
print("per-col corr min/mean:", np.nanmin([d[1] for d in diffs]), np.nanmean([d[1] for d in diffs]))
print("per-col maxabsdiff max:", np.nanmax([d[2] for d in diffs]))

# ---- 2. BUILD cluster-level BN ratio with user's cluster_mapping_dict ----
cmap = pd.read_csv(CMAP)
print("\ncluster_mapping_dict cols:", list(cmap.columns), "shape:", cmap.shape)
cmap_idx = cmap.set_index('plot')
cluster_fgc = ctype_ratio_agg(fgc, map_df=cmap_idx, key='cluster')
print("cluster FGC:", cluster_fgc.shape, "n clusters:", cluster_fgc.shape[1])
cluster_bn = surf_relabel(src='FGC', trg='BN', data=cluster_fgc, method='mean')
print("cluster BN:", cluster_bn.shape, "index match subclass:", list(cluster_bn.index) == list(ref.index))
print("cluster cols:", list(cluster_bn.columns))

# sanity: rows are compositions -> row sums (over mapped clusters)
rs = cluster_bn.sum(axis=1)
print("row-sum min/max:", float(rs.min()), float(rs.max()))
print("n NaN cells:", int(cluster_bn.isna().sum().sum()))

cluster_bn.to_csv(OUT)
print("\nWROTE", OUT)
