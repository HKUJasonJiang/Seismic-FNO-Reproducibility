"""Read-only R3-C6 database checks; diagnostic flags do not certify stability."""
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import h5py
import numpy as np

FILES = {'F2':'Blg_F2_6m_IM7_st0.h5','F6':'Blg_F6_18m_IM7_st0.h5',
         'F10':'Blg_F10_30m_IM7_st0.h5','FW10':'Blg_F10_30m_IM7_st1.h5',
         'FW14':'Blg_F14_42m_IM7_st1.h5','FW17':'Blg_F17_52m_IM7_st1.h5'}

def scan(path, dataset, out, label):
    start=time.time()
    with h5py.File(path,'r') as f:
        d=f[dataset]; n,t=d.shape
        peak=np.empty(n); tail=np.empty(n,dtype=np.int32)
        nfinite=np.empty(n,dtype=np.int32); zeros=np.empty(n,dtype=np.int32)
        digest=hashlib.sha256()
        for lo in range(0,n,512):
            a=d[lo:lo+512,:]; digest.update(np.ascontiguousarray(a).tobytes())
            finite=np.isfinite(a); nfinite[lo:lo+len(a)]=np.sum(~finite,axis=1)
            peak[lo:lo+len(a)]=np.max(np.where(finite,np.abs(a),0),axis=1)
            nonzero=a!=0
            zeros[lo:lo+len(a)]=np.sum(~nonzero,axis=1)
            tail[lo:lo+len(a)]=np.where(np.any(nonzero,axis=1),np.argmax(nonzero[:,::-1],axis=1),t)
        meta=dict(path=str(path),size_bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                  dataset=dataset,shape=list(d.shape),dtype=str(d.dtype),
                  logical_dataset_sha256=digest.hexdigest(),
                  hash_definition='SHA256 concatenated C-order raw dtype bytes, row order; not container file hash',
                  elapsed_seconds=time.time()-start,attributes={})
        if 'Blg_Attributes' in f:
            meta['attributes']={k:np.asarray(v).tolist() for k,v in f['Blg_Attributes'].items()}
        if 'Blg_Damage_State' in f:
            ds=np.asarray(f['Blg_Damage_State']).reshape(-1)
            vals,counts=np.unique(ds,return_counts=True)
            meta['damage_state_counts']={str(v):int(c) for v,c in zip(vals,counts)}
            meta['damage_state_logical_sha256']=hashlib.sha256(ds.tobytes()).hexdigest()
        else: ds=np.full(n,np.nan)
    np.savez_compressed(out/f'{label}_samples.npz',peak=peak,trailing_zero_points=tail,
                        nonfinite_count=nfinite,zero_count=zeros,damage_state=ds)
    meta.update(n_nonfinite_values=int(nfinite.sum()),n_nonfinite_samples=int((nfinite>0).sum()),
                all_zero_samples=int((zeros==t).sum()),
                nonzero_with_trailing_zeros=int(((tail>0)&(tail<t)).sum()),
                tail_zero_points_quantiles=np.quantile(tail,[0,.25,.5,.75,1]).tolist(),
                peak_quantiles_ms2=np.quantile(peak,[0,.01,.25,.5,.75,.95,.99,1]).tolist(),
                complete_label_count=int((ds==4).sum()),
                complete_label_fraction=float((ds==4).mean()) if np.any(np.isfinite(ds)) else None,
                diagnostic_gt_100_ms2_count=int((peak>100).sum()),
                diagnostic_gt_1000_ms2_count=int((peak>1000).sum()),
                diagnostic_gt_1e6_ms2_count=int((peak>1e6).sum()))
    flags=np.flatnonzero((nfinite>0)|(zeros==t)|(peak>1000))
    with (out/f'{label}_flagged_samples.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f); w.writerow(['sample_index','gm_index','scale_index','peak_ms2','DS',
                                     'nonfinite_values','zero_points','trailing_zero_points'])
        for i in flags:w.writerow([i,i//57,i%57,peak[i],ds[i],nfinite[i],zeros[i],tail[i]])
    print(label,meta['peak_quantiles_ms2'],meta.get('damage_state_counts'),
          'nonfinite',meta['n_nonfinite_values'],'zero',meta['all_zero_samples'],flush=True)
    return meta,peak,tail,ds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    gm=args.base/'All_GMs/GMs_knet_3474_AF_57.h5'
    report={'environment':dict(python=sys.version,numpy=np.__version__,h5py=h5py.__version__,
                               platform=platform.platform()),'units_basis':'m/s2 per generation README',
            'large_value_thresholds_are_diagnostic_not_failure_criteria':[100,1000,1e6],
            'data':{},'comparison':{}}
    m,pga,gtail,_=scan(gm,'Acc_GMs',args.out,'inputs');report['data']['inputs']=m
    perac=[]
    for ac in range(57):
        q=pga[ac::57]; perac.append(dict(scale_index=ac,min=float(q.min()),median=float(np.median(q)),max=float(q.max())))
    report['input_pga_by_scale']=perac
    for name,file in FILES.items():
        path=args.base/'knet-250/Data/fno'/file
        if not path.exists():path=args.base/'knet-250/Data'/file
        m,pfa,tail,ds=scan(path,'Acc_Floor_Response',args.out,name);report['data'][name]=m
        ratio=np.divide(pfa,pga,out=np.full_like(pfa,np.nan),where=pga!=0)
        report['comparison'][name]=dict(peak_amplification_quantiles=np.quantile(ratio,[0,.5,.95,.99,1]).tolist(),
                                        output_tails_and_nonzero_input=int(((tail>0)&(gtail==0)).sum()),
                                        output_zero_when_input_nonzero=int((pfa==0).sum()))
        (args.out/'database_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    fields=['structure','n_samples','pga_min','pga_max','response_peak_max','complete_count','complete_pct',
            'nonfinite_values','all_zero_responses','responses_with_tail_zeros','gt100_ms2','gt1000_ms2']
    with (args.out/'database_health_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name in FILES:
            m=report['data'][name]
            w.writerow(dict(structure=name,n_samples=m['shape'][0],pga_min=pga.min(),pga_max=pga.max(),
                            response_peak_max=m['peak_quantiles_ms2'][-1],complete_count=m['complete_label_count'],
                            complete_pct=100*m['complete_label_fraction'],nonfinite_values=m['n_nonfinite_values'],
                            all_zero_responses=m['all_zero_samples'],responses_with_tail_zeros=m['nonzero_with_trailing_zeros'],
                            gt100_ms2=m['diagnostic_gt_100_ms2_count'],gt1000_ms2=m['diagnostic_gt_1000_ms2_count']))

if __name__=='__main__':main()
