import json, hashlib, re
from html.parser import HTMLParser
from pathlib import Path
import pandas as pd
R=Path('/share/user_data/dthbca/public/experiment/BigBrainLayer/results/homolomap_layer_branch_compare_20260812_060115'); O=R/'output'; H=O/'layer_pipeline_report.html'
class P(HTMLParser):
 def __init__(self): super().__init__(); self.tags=[]; self.img=[]; self.href=[]
 def handle_starttag(self,t,a):
  self.tags.append(t); d=dict(a)
  if t=='img': self.img.append(d.get('src'))
  if t=='a': self.href.append(d.get('href'))
p=P(); text=H.read_text(encoding='utf-8'); p.feed(text)
s=pd.read_csv(R/'branch_summary.csv'); b=s.iloc[0]
critical=[str(int(b.n_fdr_sig)),f'{b.mean_abs_r:.4f}',f'{b.whole_match_p:.7f}','106、118、194','1,000','normalized feature → relabel']
assets_ok=all((O/x).exists() and (O/x).stat().st_size>0 for x in p.img)
links_ok=all((O/x).resolve().exists() for x in p.href if x and not x.startswith('#'))
manifest=json.loads((O/'output_manifest.json').read_text())
hash_ok=all(hashlib.sha256((O/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
qa={'status':'PASS','html_bytes':H.stat().st_size,'doctype':text.lower().startswith('<!doctype html>'),'has_html_head_body':all(x in p.tags for x in ['html','head','body']),'image_count':len(p.img),'images':p.img,'assets_ok':assets_ok,'file_link_count':len(p.href),'links_ok':links_ok,'offline_no_http':not bool(re.search(r'https?://|//cdn',text,re.I)),'critical_values':{x:(x in text) for x in critical},'critical_values_ok':all(x in text for x in critical),'manifest_hash_ok':hash_ok,'manifest':manifest}
qa['status']='PASS' if all([qa['doctype'],qa['has_html_head_body'],assets_ok,links_ok,qa['offline_no_http'],qa['critical_values_ok'],hash_ok,len(p.img)==4]) else 'FAIL'
(O/'qa_report.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(qa,indent=2,ensure_ascii=False))
