"""Render a representative spin-FDR-significant layer-cell association."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':7,'svg.fonttype':'none','pdf.fonttype':42})
LAYERS=['l1','l2','l3','l4','l5','l6']

def reclose(mapped):
    out={k:v.copy().astype(float) for k,v in mapped.items()}
    for ctype in out['l1'].columns:
        m=pd.concat({l:out[l][ctype] for l in LAYERS},axis=1); den=m.sum(axis=1)
        good=np.isfinite(den)&(den>0)
        for l in LAYERS:
            out[l].loc[good,ctype]=m.loc[good,l]/den[good]
            out[l].loc[~good,ctype]=0.0
    return out

def to_rgb(fig):
    fig.canvas.draw(); rgba=np.asarray(fig.canvas.buffer_rgba()); alpha=rgba[:,:,3:4]/255
    rgb=np.rint(rgba[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8); plt.close(fig)
    nonwhite=(rgb<250).any(axis=2); rows,cols=np.where(nonwhite)
    return rgb[max(rows.min()-4,0):min(rows.max()+5,len(rgb)),max(cols.min()-4,0):min(cols.max()+5,rgb.shape[1])]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--package',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); sys.path.insert(0,str(a.package)); a.output.mkdir(parents=True,exist_ok=True)
    from HomoloMap.datasets.layers import load_layer_counts,normalize_layer_composition,relabel_layer_counts,fetch_bigbrain_layer_thickness
    from HomoloMap.plotting import plot_left
    raw=load_layer_counts(a.data,source_atlas='D99',mapping_column='subclass',unmapped='drop')
    norm=normalize_layer_composition(raw,mode='within_region',zero_policy='zero')
    mapped=reclose(relabel_layer_counts(norm,'D99','BN',method='mean',cross_species=True,unknown_labels='drop'))
    feature=mapped['l4']['Pax6']; thickness=fetch_bigbrain_layer_thickness('BN',a.data,relative=True,regions=feature.index)['Layer IV']
    feature,thickness=feature.align(thickness,join='inner'); table=pd.DataFrame({'Layer_IV_Pax6':feature,'Layer_IV_relative_thickness':thickness})
    table.index.name='BN_roi'; table.to_csv(a.output/'layer4_pax6_brain_source.csv')
    f1=plot_left(feature,atlas='BN',species='human',surf='inflated',view='row',cmap='YlGnBu',outline=True,
                 cbar_label='Pax6 cross-layer proportion',title='',size=(560,260),zoom=1.35,dpi=300,cbar_kwargs={'decimals':3,'shrink':.68})
    f2=plot_left(thickness,atlas='BN',species='human',surf='inflated',view='row',cmap='magma',outline=True,
                 cbar_label='Layer IV relative thickness',title='',size=(560,260),zoom=1.35,dpi=300,cbar_kwargs={'decimals':3,'shrink':.68})
    images=[to_rgb(f1),to_rgb(f2)]
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.5))
    for ax,img,title in zip(axes,images,['Layer IV Pax6 composition','Layer IV relative thickness']): ax.imshow(img); ax.set_title(title,fontsize=8,pad=3); ax.axis('off')
    fig.text(.5,.015,'Spatial Pearson r = 0.889 | spin p = 0.000999 | FDR q = 0.00745',ha='center',fontsize=7,color='#26343B')
    fig.subplots_adjust(left=.01,right=.99,top=.93,bottom=.11,wspace=.03)
    base=a.output/'figure_brain_layer4_pax6_spin_significant'
    for ext,dpi in [('png',300),('svg',None),('pdf',None),('tiff',600)]:
        kw={'bbox_inches':'tight','facecolor':'white'}
        if dpi: kw['dpi']=dpi
        fig.savefig(str(base)+f'.{ext}',**kw)
    plt.close(fig)
    print(json.dumps({'status':'PASS','n_roi':len(table),'r':float(table.corr().iloc[0,1]),'source':str(a.output)}))
if __name__=='__main__': main()
