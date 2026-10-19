"""Reproduce the nine archived candidate identities and select on new validation results."""
from pathlib import Path
import argparse,csv,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
FAMILIES=['FNO','Transformer','BiLSTM','ResCNN1D','UNet1D','MLP']
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',required=True,choices=FAMILIES)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--workspace',type=Path,default=ROOT/'runs/workspace')
    p.add_argument('--stage',required=True,choices=['dryrun','screen','select','full'])
    a=p.parse_args();additional=a.family in FAMILIES[3:]
    exp='E5_additional_20261017' if additional else 'E5'
    archived=ROOT/'recorded/6-code/O7-revision'/exp/'results'/a.family/'screen'
    trials=sorted(s.name for s in archived.iterdir() if s.is_dir())
    assert len(trials)==9
    base=a.workspace.resolve()/'6-code/O7-revision'/exp/'results'/a.family
    def invoke(trial,stage):
        subprocess.run([sys.executable,str(ROOT/'run.py'),'--workspace',str(a.workspace.resolve()),'e5-additional' if additional else 'e5','--family',a.family,'--trial',trial,'--stage',stage,'--cache',str(a.cache.resolve())],check=True)
    if a.stage in ['dryrun','screen']:
        for trial in trials:invoke(trial,a.stage)
        return
    values=[]
    for trial in trials:
        folder=base/'screen'/trial
        d=json.loads((folder/'summary.json').read_text())
        log=list(csv.DictReader((folder/'training_log.csv').open()))
        assert d['status']=='complete' and d['test_used_for_selection'] is False
        assert [int(r['epoch']) for r in log]==list(range(1,21))
        values.append((float(d['best_val_mse']),trial))
    mse,trial=min(values);print(json.dumps({'family':a.family,'selected_trial':trial,'best_val_mse':mse,'selection':'validation only; ties by trial ID'}))
    if a.stage=='full':invoke(trial,'full')
if __name__=='__main__':main()
