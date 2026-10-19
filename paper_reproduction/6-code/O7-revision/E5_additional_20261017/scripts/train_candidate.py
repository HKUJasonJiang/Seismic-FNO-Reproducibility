"""Isolated FP32 E5 candidate with resumable epochs and test-free screening."""
from pathlib import Path
import argparse,json,sys,time,csv,hashlib,platform,random,os,copy
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

p=argparse.ArgumentParser()
p.add_argument('--root',required=True);p.add_argument('--family',required=True);p.add_argument('--trial',required=True)
p.add_argument('--stage',choices=['dryrun','screen','full'],required=True)
p.add_argument('--cache',required=True);p.add_argument('--max-batches',type=int,default=0)
p.add_argument('--dryrun-tag',default='')
a=p.parse_args();assert a.max_batches == 0, 'Official runs must use every batch';root=Path(a.root);base=root/'6-code/O7-revision/E5_additional_20261017';out=base/'results'/a.family/a.stage/a.trial
if a.dryrun_tag:
 assert a.stage=='dryrun' and a.dryrun_tag.isalnum()
 out=out/a.dryrun_tag
out.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(base/'sources')]
from baseline_models import build_model,count_parameters
from metrics import RegressionMeter,PFAMeter
protocol=json.loads((base/'configs/protocol.json').read_text());candidates=json.loads((base/'configs/search_space.json').read_text())
candidate=next(v for v in candidates[a.family] if v['trial_id']==a.trial)
cfg={k:v for k,v in candidate.items() if k not in ('trial_id','original','lr')}
hz=20 if a.stage=='transfer' else 50;seq_len=60*hz
def build(length):
 if a.family=='FNO':
  from neuralop.models import FNO
  settings=dict(cfg);settings['n_modes']=(settings['n_modes'],)
  return FNO(**settings,in_channels=1,out_channels=1,projection_channel_ratio=2,domain_padding=0.1)
 if a.family=='Transformer':return build_model('Transformer',**cfg,seq_len=length)
 return build_model(a.family,**cfg)

random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(8)
model=build(seq_len).cuda();opt=torch.optim.AdamW(model.parameters(),lr=candidate['lr'],weight_decay=1e-4)
epochs=20 if a.stage=='screen' else 50
step=8 if a.stage=='screen' else 20
scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=step,gamma=0.5)
env=dict(host=platform.node(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
 gpu=torch.cuda.get_device_name(),device_total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
 seed=42,tf32=False,precision='fp32',stage=a.stage,candidate=candidate,architecture=cfg,seq_len=seq_len,
 epochs=epochs,batch=64,grad_clip=1.0,scheduler_step=step,params=count_parameters(model),
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 protocol_sha256=hashlib.sha256((base/'configs/protocol.json').read_bytes()).hexdigest(),
 cache_manifest=json.loads((Path(a.cache)/'manifest.json').read_text()) if (Path(a.cache)/'manifest.json').exists() else None)
env.update(family=a.family, trial_id=a.trial, max_batches=a.max_batches,
 source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/'sources').glob('*.py')},
 search_space_sha256=hashlib.sha256((base/'configs/search_space.json').read_bytes()).hexdigest(),
 split_file_sha256=hashlib.sha256((base/'configs/frozen_splits.npz').read_bytes()).hexdigest())
if (out/'config.json').exists():
 previous=json.loads((out/'config.json').read_text())
 for key in ['candidate','protocol_sha256','script_sha256','source_sha256','split_file_sha256']:
  if previous.get(key)!=env[key]:raise RuntimeError('Resume provenance mismatch: '+key)
(out/'config.json').write_text(json.dumps(env,indent=2),encoding='utf8')

def checkpoint(path,payload):
 tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)

if a.stage=='dryrun':
 torch.cuda.reset_peak_memory_stats();times=[]
 try:
  for i in range(3):
   start=time.perf_counter();x=torch.randn(64,1,seq_len,device='cuda');y=torch.randn_like(x)
   opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(x),y)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();torch.cuda.synchronize()
   times.append(time.perf_counter()-start)
  result=dict(status='ok',batch_times_s=times,steady_batch_s=float(np.median(times[1:])),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,params=env['params'])
 except torch.cuda.OutOfMemoryError as e:result=dict(status='oom',message=str(e))
 (out/'dryrun.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);sys.exit(0 if result['status']=='ok' else 3)

splits=np.load(base/'configs/frozen_splits.npz');cache=Path(a.cache)
x=np.load(cache/f'x{hz}.npy',mmap_mode='r');y=np.load(cache/f'y{hz}.npy',mmap_mode='r')
prefix='screen' if a.stage in ('screen','scarce') else 'full'
def loader(indices,shuffle):
 # Identical ordered source values to DynamicDataset, without repeatedly rereading attributes.
 xx=torch.from_numpy(np.array(x[indices],dtype=np.float32)).unsqueeze(1)
 yy=torch.from_numpy(np.array(y[indices],dtype=np.float32)).unsqueeze(1)
 return DataLoader(TensorDataset(xx,yy),batch_size=64,shuffle=shuffle,num_workers=0,pin_memory=True)
train=loader(splits[prefix+'_train'],True);val=loader(splits[prefix+'_val'],False)
def epoch(loader,optimize):
 model.train(optimize);reg=RegressionMeter();pfa=PFAMeter();t=time.perf_counter()
 for i,(xx,yy) in enumerate(loader):
  if a.max_batches and i>=a.max_batches:break
  xx=xx.cuda(non_blocking=True);yy=yy.cuda(non_blocking=True)
  if optimize:opt.zero_grad(set_to_none=True)
  with torch.set_grad_enabled(optimize):
   pred=model(xx);loss=torch.nn.functional.mse_loss(pred,yy)
   if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss batch {i}')
   if optimize:
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
  reg.update(pred,yy);pfa.update(pred,yy)
 torch.cuda.synchronize();return dict(**reg.compute(),**pfa.compute(),seconds=time.perf_counter()-t, evaluated_samples=pfa.n, evaluated_scalars=reg.n)

last=out/'checkpoints/last.pt';best=out/'checkpoints/best.pt';last.parent.mkdir(exist_ok=True)
best_mse=float('inf');start_epoch=1;elapsed=0.;rows=[]
if last.exists():
 state=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
 start_epoch=state['epoch']+1;best_mse=state['best_val_mse'];elapsed=state['elapsed_s']
 torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
 print('Resuming at epoch',start_epoch,flush=True)
logpath=out/'training_log.csv'
if start_epoch>1 and logpath.exists():
 old_rows=list(csv.DictReader(logpath.open()))
 keep=[r for r in old_rows if int(r['epoch'])<start_epoch]
 if [int(r['epoch']) for r in keep]!=list(range(1,start_epoch)):raise RuntimeError('Incomplete checkpoint/log history')
 if len(keep)!=len(old_rows):
  with logpath.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(old_rows[0]));w.writeheader();w.writerows(keep)
fields=['epoch','lr','train_mse','train_r2','val_mse','val_r2','val_pfa_r2','train_s','val_s','epoch_s','peak_allocated_gib','peak_reserved_gib']
with logpath.open('a' if start_epoch>1 else 'w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields)
 if start_epoch==1:writer.writeheader();f.flush()
 for e in range(start_epoch,epochs+1):
  t=time.perf_counter();torch.cuda.reset_peak_memory_stats();lr=opt.param_groups[0]['lr']
  tm=epoch(train,True);vm=epoch(val,False);scheduler.step();secs=time.perf_counter()-t;elapsed+=secs
  record=dict(epoch=e,lr=lr,train_mse=tm['mse'],train_r2=tm['r2'],val_mse=vm['mse'],val_r2=vm['r2'],val_pfa_r2=vm['pfa_r2'],train_s=tm['seconds'],val_s=vm['seconds'],epoch_s=secs,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
  writer.writerow(record);f.flush();print(json.dumps(record),flush=True)
  if vm['mse']<best_mse:
   best_mse=vm['mse'];checkpoint(best,dict(model_state_dict=model.state_dict(),epoch=e,metrics=vm,config=env,candidate=candidate))
  checkpoint(last,dict(model_state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=e,best_val_mse=best_mse,elapsed_s=elapsed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),config=env))
summary=dict(status='complete',family=a.family,stage=a.stage,trial_id=a.trial,best_val_mse=best_mse,train_time_s=elapsed,best_checkpoint=str(best),params=env['params'],test_used_for_selection=False)
# Screening never constructs a test loader.
if a.stage!='screen':
 state=torch.load(best,map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict']);model.eval();test=loader(splits['test'],False)
 summary['test_same_grid']=epoch(test,False)
 if a.stage=='transfer':
  del test;del x,y
  x=np.load(cache/'x50.npy',mmap_mode='r');y=np.load(cache/'y50.npy',mmap_mode='r');test=loader(splits['test'],False)
  if a.family=='Transformer':
   model50=build(3000).cuda();ss=dict(state['model_state_dict']);ss['pos_embed']=torch.nn.functional.interpolate(ss['pos_embed'].permute(0,2,1),size=model50.n_patches,mode='linear',align_corners=True).permute(0,2,1);model50.load_state_dict(ss);model=model50
  summary['test_direct50']=epoch(test,False)
  # Resampling wrapper uses exactly SciPy polyphase filters both directions.
  from scipy.signal import resample_poly
  if a.family=='Transformer':model=build(1200).cuda();model.load_state_dict(state['model_state_dict'])
  model.eval();reg=RegressionMeter();pfa=PFAMeter()
  for xx,yy in test:
   xx20=torch.from_numpy(resample_poly(xx.numpy(),2,5,axis=-1).astype(np.float32)).cuda()
   with torch.inference_mode():pp=model(xx20).cpu().numpy()
   pp50=torch.from_numpy(resample_poly(pp,5,2,axis=-1)[...,:3000].astype(np.float32));reg.update(pp50,yy);pfa.update(pp50,yy)
  summary['test_wrapped50']=dict(**reg.compute(),**pfa.compute())
summary_tmp=out/'summary.json.tmp';summary_tmp.write_text(json.dumps(summary,indent=2),encoding='utf8');os.replace(summary_tmp,out/'summary.json');print(json.dumps(summary),flush=True)
