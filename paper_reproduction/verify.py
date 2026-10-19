"""Audit frozen splits, complete training evidence, selection and reported metrics."""
from pathlib import Path
import csv, hashlib, json, math, sys
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=1e-10),(a,b)
def main():
    checks=[]
    manifest=read(ROOT/'provenance/source_manifest.json')
    for x in manifest['files']:
        assert hashlib.sha256((ROOT/x['destination']).read_bytes()).hexdigest()==x['packaged_sha256'],x['destination']
    checks.append({'check':'packaged source/evidence hashes','files':len(manifest['files'])})
    data=read(ROOT/'provenance/training_data_manifest.json')['files']
    assets=read(ROOT/'provenance/public_assets.json')['assets']
    for key,filename in [('x50','GMs_knet_3474_AF_57.h5'),('y50','Blg_F6_18m_IM7_st0.h5')]:
        assert next(x['sha256'] for x in assets if x['path'].endswith('/'+filename))==data[key]['source_sha256']
    checks.append({'check':'pinned public GM/F6 content hashes match E5 training sources','passed':True})
    expected=rows(ROOT/'recorded/six_architecture_tuning.csv'); count=0; results=[]
    for exp in ['E5','E5_additional_20261017']:
        base=ROOT/'6-code/O7-revision'/exp
        cfg=read(base/'configs/protocol.json');split=np.load(base/'configs/frozen_splits.npz')
        for key,h in cfg['split_hashes'].items():
            assert hashlib.sha256(np.asarray(split[key],dtype='<i8').tobytes()).hexdigest()==h,key
        assert [len(split[k]) for k in ['full_train','full_val','test','screen_train','screen_val']]==[136800,34200,27018,45600,11400]
        gm=[set(split[k].tolist()) for k in ['train_gms','val_gms','test_gms']]
        assert not gm[0]&gm[1] and not gm[0]&gm[2] and not gm[1]&gm[2]
        for k,g in zip(['full_train','full_val','test'],gm):assert set((split[k]//57).tolist())==g
        candidates=read(base/'configs/search_space.json')
        records=ROOT/'recorded/6-code/O7-revision'/exp/'results'
        for family in candidates:
            screen=[]
            for stage,epochs in [('screen',20),('full',50)]:
                paths=sorted((records/family/stage).glob('*/summary.json'))
                assert len(paths)==(9 if stage=='screen' else 1),(family,stage)
                for path in paths:
                    summary=read(path);config=read(path.parent/'config.json');log=rows(path.parent/'training_log.csv')
                    assert summary['status']=='complete' and summary['test_used_for_selection'] is False
                    assert [int(r['epoch']) for r in log]==list(range(1,epochs+1))
                    assert config['candidate']==next(c for c in candidates[family] if c['trial_id']==path.parent.name)
                    assert config['protocol_sha256']==hashlib.sha256((base/'configs/protocol.json').read_bytes()).hexdigest()
                    assert config['batch']==64 and config['seed']==42 and config['precision']=='fp32' and config['tf32'] is False
                    assert all(math.isfinite(float(r['val_mse'])) and math.isfinite(float(r['train_mse'])) for r in log)
                    close(summary['best_val_mse'],min(float(r['val_mse']) for r in log))
                    if stage=='screen':
                        assert not any(k.startswith('test_') and k!='test_used_for_selection' for k in summary),path
                        screen.append((float(summary['best_val_mse']),path.parent.name));count+=1
                    else:
                        assert path.parent.name==min(screen)[1],(family,min(screen),path.parent.name)
                        target=next(r for r in expected if r['model']==family)
                        for key in ['mse','r2','pfa_r2']:close(summary['test_same_grid'][key],target[key])
                        assert summary['params']['real_params']==int(target['real_params'])
                        results.append({'family':family,'selected_trial':path.parent.name,'epochs':epochs,'r2':float(target['r2'])})
    checks.append({'check':'frozen grouped splits and no GM leakage','passed':True})
    checks.append({'check':'complete validation-only candidate selection','screen_trials':count,'final_trials':len(results),'results':results})
    output={'status':'pass','checks':checks,'scope':'Saved evidence integrity and arithmetic; not a rerun of GPU training or checkpoint inference.'}
    dest=ROOT/'runs/verification.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
