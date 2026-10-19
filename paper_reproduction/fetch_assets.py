"""Download only requested paper assets from a pinned public dataset revision."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, urllib.request
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('group',choices=['data','o1','pga','sampling','o4'])
    p.add_argument('--workspace',type=Path,default=ROOT/'runs/workspace')
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--list',action='store_true',help='List the download plan and byte count without downloading.')
    a=p.parse_args();manifest=json.loads((ROOT/'provenance/public_assets.json').read_text())
    selected=[]
    for item in manifest['assets']:
        n=item['path'];dest=None
        if a.group=='data' and n.startswith('MDOF/'):dest=a.data_root/n
        if a.group=='o1' and n.startswith('experiments/O1_baseline/'):
            dest=a.workspace/'6-code/O1-baseline-comparison/results'/n.split('/artifact/')[1]
        if a.group in ['pga','sampling'] and f'/{a.group}_ood/' in n and '/BiLSTM/' not in n:
            dest=a.workspace/'6-code/O3-generalization/results'/n.split('/artifact/')[1]
        if a.group=='o4' and n.startswith('experiments/O4_multi_structure/'):
            dest=a.workspace/'6-code/O4-multi-structure/results/runs'/n.split('/runs/')[1]
        if dest is not None:selected.append((item,dest.resolve()))
    print(json.dumps({'revision':manifest['revision'],'bytes':sum(i.get('size') or 0 for i,_ in selected),'files':[str(d) for _,d in selected]},indent=2))
    if a.list:return
    for item,dest in selected:
        expected=item['sha256']
        if not expected:raise RuntimeError('Missing content SHA256 for '+item['path'])
        if dest.exists():
            if sha(dest)==expected:continue
            raise RuntimeError('Existing file checksum mismatch; inspect it before retrying: '+str(dest))
        dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.part')
        with urllib.request.urlopen(item['url'],timeout=120) as response,tmp.open('wb') as out:shutil.copyfileobj(response,out,8*1024*1024)
        if sha(tmp)!=expected:raise RuntimeError('Download checksum mismatch: '+str(tmp))
        os.replace(tmp,dest)
    if a.group=='data':
        src=a.data_root/'MDOF/knet-250/Data/fno/Blg_F6_18m_IM7_st0.h5'
        dst=a.data_root/'f6_only'/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            try:os.link(src,dst)
            except OSError:shutil.copy2(src,dst)
        if sha(dst)!=sha(src):raise RuntimeError('F6-only data differs from the public F6 artifact')
if __name__=='__main__':main()
