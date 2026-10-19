"""Make exact float32 DynamicDataset-equivalent arrays; never mutate raw HDF5."""
from pathlib import Path
import argparse,json,time,hashlib
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--gm',required=True);p.add_argument('--building',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);meta={'files':{},'conversion':'float64 or native HDF5 -> float32 identical to torch.from_numpy(...).float()','created_unix':time.time()}
for key,path,dataset in [('x50',a.gm,'Acc_GMs'),('y50',a.building,'Acc_Floor_Response')]:
 target=out/(key+'.npy')
 with h5py.File(path,'r') as f:
  src=f[dataset]; dst=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=src.shape)
  for i in range(0,len(src),1024):dst[i:i+1024]=src[i:i+1024].astype(np.float32);print(key,i,len(src),flush=True)
  dst.flush();del dst
  st=Path(path).stat();meta['files'][key]={'source':path,'dataset':dataset,'source_shape':list(src.shape),'source_dtype':str(src.dtype),'source_size':st.st_size,'source_mtime':st.st_mtime,'cache':str(target)}
 h=hashlib.sha256()
 with target.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
 meta['files'][key]['cache_sha256']=h.hexdigest()
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
 meta['files'][key]['source_sha256']=h.hexdigest()
 (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
print(json.dumps(meta,indent=2))
