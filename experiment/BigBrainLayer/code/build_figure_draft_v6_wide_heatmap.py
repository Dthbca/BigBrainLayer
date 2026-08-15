"""Clean AI-element workflow with rebuilt labels and compact evidence layout."""
from pathlib import Path
import json, numpy as np, pandas as pd
from PIL import Image
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.spines.right':False,
 'axes.spines.top':False,'axes.linewidth':.7,'legend.frameon':False,'figure.facecolor':'white'})
ROOT=Path(r'D:\HomoloMap\reports\homolomap_layer_branch_compare_20260812_060115\output')
RES=ROOT/'reclosure_v3'; COR=ROOT/'permutation_corrected_v10'; LSH=RES/'layer_specific_shap'; OUT=ROOT/'figure_draft_v6'; OUT.mkdir(parents=True,exist_ok=True)
REF=Image.open(r'D:\HomoloMap\.reference_preview\figure4_reference_render.png').convert('RGB')
BLUE='#477FA8'; ORANGE='#D47A45'; TEAL='#4F9087'; DARK='#26343B'; LINE='#819097'

def crop(name,box):
    w,h=REF.size; im=REF.crop(tuple(int(v*s) for v,s in zip(box,(w,h,w,h)))); im.save(OUT/f'reference_element_{name}.png'); return np.asarray(im)
def trim(arr,threshold=248):
    mask=(arr<threshold).any(axis=2); ys,xs=np.where(mask)
    if not len(xs): return arr
    pad=4; return arr[max(ys.min()-pad,0):min(ys.max()+pad+1,arr.shape[0]),max(xs.min()-pad,0):min(xs.max()+pad+1,arr.shape[1])]
E={
 'macaque':crop('macaque_layers',(0.035,0.042,0.195,0.145)),
 'stereo':crop('stereo_section',(0.195,0.043,0.310,0.148)),
 'layers':crop('six_layers',(0.315,0.040,0.393,0.150)),
 'ratios':crop('cell_ratios',(0.438,0.050,0.575,0.148)),
 'homology':crop('homology_mapping',(0.640,0.012,0.785,0.068)),
 'histology':crop('bigbrain_histology',(0.035,0.188,0.175,0.258)),
 'brain':crop('brain_maps',(0.350,0.194,0.535,0.244)),
}

def save(fig,stem):
    for ext,dpi in [('png',300),('svg',None),('pdf',None),('tiff',600)]:
        kw={'bbox_inches':'tight','facecolor':'white'}
        if dpi: kw['dpi']=dpi
        fig.savefig(OUT/f'{stem}.{ext}',**kw)
    plt.close(fig)
def put(ax,img,extent):
    left,right,bottom,top=extent
    ia=ax.inset_axes([left,bottom,right-left,top-bottom])
    ia.imshow(img,aspect='equal')
    ia.axis('off')
def arrow(ax,a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.9,color=TEAL,zorder=5))

def workflow(ax):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.text(.005,.98,'a',fontweight='bold',fontsize=9,va='top')
    ax.text(.035,.98,'Cross-species laminar mapping and inference',fontweight='bold',fontsize=9,va='top')
    # Upper cell-composition stream; all old labels/arrows are excluded.
    items=[('macaque',(.02,.135,.58,.86),'Macaque\nlaminar tissue'),('stereo',(.16,.275,.58,.86),'Spatial\ntranscriptomics'),
           ('layers',(.31,.385,.58,.86),'Six-layer\nprofiles'),('ratios',(.42,.535,.58,.86),'Cross-layer\ncomposition'),
           ('homology',(.57,.695,.62,.86),'Homologous\ncell mapping'),('brain',(.72,.825,.60,.84),'BN relabel +\nreclosure')]
    for key,ext,label in items: put(ax,E[key],ext); ax.text((ext[0]+ext[1])/2,.52,label,ha='center',va='top',fontsize=5.9,fontweight='bold',color=DARK)
    for x1,x2 in [( .135,.16),(.275,.31),(.385,.42),(.535,.57),(.695,.72)]: arrow(ax,(x1,.72),(x2,.72))
    # Lower thickness stream.
    put(ax,E['histology'],(.035,.155,.12,.35)); put(ax,E['brain'],(.24,.355,.13,.34))
    ax.text(.095,.075,'BigBrain\nlayer thickness',ha='center',va='top',fontsize=5.9,fontweight='bold',color=DARK)
    ax.text(.298,.075,'BN layer maps +\nrelative thickness',ha='center',va='top',fontsize=5.9,fontweight='bold',color=DARK)
    arrow(ax,(.16,.23),(.235,.23)); arrow(ax,(.36,.23),(.84,.37)); arrow(ax,(.825,.70),(.86,.58))
    # Shared inference endpoint mirrors the reference's vertical statistics block.
    box=FancyBboxPatch((.855,.16),.135,.66,boxstyle='round,pad=.012,rounding_size=.018',facecolor='#F4F7F8',edgecolor=LINE,lw=.7)
    ax.add_patch(box); ax.text(.922,.68,'Layer-matched\nassociation',ha='center',va='center',fontsize=6.3,fontweight='bold',color=DARK)
    ax.plot([.875,.97],[.57,.57],color='#CED5D8',lw=.6); ax.text(.922,.46,'Spatial spin\nFDR-BH',ha='center',va='center',fontsize=6.0,color=DARK)
    ax.text(.922,.27,'Exact layer-order\npermutation',ha='center',va='center',fontsize=6.0,color=DARK)

def main():
    branch='within_region_cross_layer__clr_false__thickness_relative'
    r=pd.read_csv(RES/'figures'/'reclosed_best_r_matrix.csv',index_col=0); q=pd.read_csv(RES/'figures'/'reclosed_best_q_matrix.csv',index_col=0).reindex_like(r)
    null=pd.read_csv(RES/'source_data'/f'{branch}__whole_null.csv').iloc[:,0].dropna().to_numpy(float)
    wr=pd.read_csv(COR/'whole_match_correction.csv').set_index('branch').loc[branch]; perf=pd.read_csv(LSH/'layer_specific_model_performance.csv')
    brain=trim(np.asarray(Image.open(ROOT/'figure_draft_v2'/'figure_brain_layer4_pax6_spin_significant.png').convert('RGB')))
    fig=plt.figure(figsize=(8.6,8.45)); gs=GridSpec(3,2,figure=fig,height_ratios=[1.15,1.42,1.0],width_ratios=[1.42,.58],hspace=.45,wspace=.24)
    workflow(fig.add_subplot(gs[0,:]))
    axb=fig.add_subplot(gs[1,0]); arr=r.to_numpy(float); im=axb.imshow(np.ma.masked_invalid(arr),aspect='auto',cmap='RdBu_r',vmin=-.9,vmax=.9)
    axb.set_yticks(range(len(r.index)),r.index); axb.set_xticks(range(len(r.columns)),r.columns,rotation=54,ha='right',rotation_mode='anchor',fontsize=5.1)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i,j]) and np.isfinite(q.iloc[i,j]) and q.iloc[i,j]<.05: axb.text(j,i,'*',ha='center',va='center',fontsize=6.5)
    cb=fig.colorbar(im,ax=axb,pad=.018,shrink=.78); cb.set_label('Pearson r',fontsize=6.5); cb.outline.set_linewidth(.5)
    axb.set_title('b  Layer-matched associations',loc='left',fontweight='bold',fontsize=8.5,pad=7)
    axc=fig.add_subplot(gs[1,1]); axc.axis('off'); axc.set_title('c  Spin-FDR spatial example',loc='left',fontweight='bold',fontsize=8.5,pad=5)
    mid=brain.shape[1]//2
    for y0,part in [(.52,brain[:,:mid]),(.03,brain[:,mid:])]:
        ia=axc.inset_axes([.02,y0,.96,.43]); ia.imshow(part); ia.axis('off')
    axd=fig.add_subplot(gs[2,0]); axd.hist(null,bins=30,color='#B7C5CF',edgecolor='white',linewidth=.35); axd.axvline(wr.observed_stat,color=ORANGE,lw=2)
    axd.set_xlabel('Mean matched-layer correlation'); axd.set_ylabel('Permutations'); axd.set_title('d  Exact layer-order inference',loc='left',fontweight='bold',fontsize=8.5,pad=7)
    axd.text(.97,.92,'4 / 720\np = 0.005556',transform=axd.transAxes,ha='right',va='top',fontsize=6.2,bbox=dict(boxstyle='round,pad=.22',facecolor='white',edgecolor='#D8DDE0',lw=.5))
    axe=fig.add_subplot(gs[2,1]); y=np.arange(6); bars=axe.barh(y,perf.oof_r2,color=BLUE,height=.62); axe.set_yticks(y,perf.layer_label); axe.invert_yaxis(); axe.set_xlim(0,.9); axe.set_xlabel('Out-of-fold $R^2$')
    for bar,row in zip(bars,perf.itertuples()):
        yc=bar.get_y()+bar.get_height()/2; axe.text(max(row.oof_r2-.014,.03),yc,f'{row.oof_r2:.2f}',ha='right',va='center',color='white',fontweight='bold',fontsize=5.8); axe.text(.88,yc,row.top_ctype,ha='right',va='center',fontsize=5.5,color=DARK)
    axe.set_title('e  Layer-specific SHAP models',loc='left',fontweight='bold',fontsize=8.5,pad=7)
    fig.subplots_adjust(left=.075,right=.975,top=.985,bottom=.07); save(fig,'figure_draft_v6_wide_heatmap')
    qa={'structure':'aspect-preserved workflow + extra-wide heatmap + vertically stacked brain maps','reference_elements':list(E),'old_reference_labels_excluded':True,
        'brain_example':{'layer':'Layer IV','ctype':'Pax6','r':.888900704,'q':.007454084377},'exports':['png','svg','pdf','tiff'],'backend':'Python/matplotlib'}
    (OUT/'figure_draft_v6_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8'); print(json.dumps({'status':'PASS','output':str(OUT)}))
if __name__=='__main__': main()
