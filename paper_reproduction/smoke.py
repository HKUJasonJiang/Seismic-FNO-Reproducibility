"""CPU architecture/import checks, independent of GPU training and saved metrics."""
from pathlib import Path
import csv,json,sys,importlib.util
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent),str(ROOT.parent/'module'),str(ROOT/'6-code/O1-baseline-comparison/scripts'),str(ROOT/'6-code/shared/utils')]
from baseline_models import build_model,count_parameters
from train_fno_large_bs64 import build_fno_large
from metrics import RegressionMeter,PFAMeter
from splits import make_fixed_474_split

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    expected=list(csv.DictReader((ROOT/'recorded/6-code/O1-baseline-comparison/results/report/o1_report_summary.csv').open()))
    results=[]
    for row in expected:
        name=row['model'];model=(build_fno_large() if name=='FNO-Large' else build_model(name)).eval()
        params=count_parameters(model)
        assert params['numel']==int(row['numel']) and params['real_params']==int(row['real_params']),name
        with torch.inference_mode():y=model(torch.randn(1,1,3000))
        assert y.shape==(1,1,3000) and torch.isfinite(y).all(),name
        results.append({'model':name,'shape':list(y.shape),'numel':params['numel'],'real_params':params['real_params']})
        del model,y
    # Closed-form check, not reusing the implementation's running sums.
    target=torch.tensor([[[1.,2.,4.]],[[2.,3.,5.]]]);pred=target+1
    m=RegressionMeter();m.update(pred,target);metrics=m.compute()
    assert abs(metrics['mse']-1)<1e-10 and abs(metrics['r2']-(1-6/float(((target-target.mean())**2).sum())))<1e-6
    pfa=PFAMeter();pfa.update(pred,target);assert abs(pfa.compute()['pfa_rmse_g']-1/9.81)<1e-6
    split=make_fixed_474_split(seed=42,use_scales_count=57)
    frozen=np.load(ROOT/'6-code/O7-revision/E5/configs/frozen_splits.npz')
    assert np.array_equal(split.test_indices,frozen['test'])
    assert np.array_equal(split.train_indices,frozen['full_train'])
    assert np.array_equal(split.val_indices,frozen['full_val'])
    selected=[]
    extra=ROOT/'6-code/O7-revision/E5_additional_20261017/sources/baseline_models.py'
    spec=importlib.util.spec_from_file_location('additional_models',extra);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    for row in csv.DictReader((ROOT/'recorded/six_architecture_tuning.csv').open()):
        name=row['model'];exp='E5' if name in ['FNO','Transformer','BiLSTM'] else 'E5_additional_20261017'
        space=json.loads((ROOT/'6-code/O7-revision'/exp/'configs/search_space.json').read_text())
        cfg=next(c for c in space[name] if c['trial_id']==row['trial']);cfg={k:v for k,v in cfg.items() if k not in ['trial_id','lr','original']}
        if name=='FNO':
            from neuralop.models import FNO
            cfg['n_modes']=(cfg['n_modes'],);model=FNO(**cfg,in_channels=1,out_channels=1,projection_channel_ratio=2,domain_padding=0.1)
        else:model=(build_model if exp=='E5' else mod.build_model)(name,**cfg)
        model.eval();params=count_parameters(model);assert params['real_params']==int(row['real_params']),name
        with torch.inference_mode():y=model(torch.randn(1,1,3000))
        assert y.shape==(1,1,3000) and torch.isfinite(y).all(),name
        selected.append({'model':name,'trial':row['trial'],'real_params':params['real_params'],'finite_forward':True});del model,y
    output={'status':'pass','torch':torch.__version__,'device':'cpu','models':results,'selected_models':selected,'checks':['original and selected parameter counts','3000-sample finite forwards','closed-form regression/PFA metrics','regenerated grouped split']}
    dest=ROOT/'runs/smoke.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
