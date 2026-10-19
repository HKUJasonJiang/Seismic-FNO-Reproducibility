"""R3-C5: isolated FP32 model-forward deployment measurements; no training."""
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
O1 = ROOT / '6-code/O1-baseline-comparison/scripts'
UP = ROOT / '6-code/shared/upstream/Seismic-FNO-clean'
sys.path[:0] = [str(O1), str(UP), str(UP/'module'), str(ROOT)]
from baseline_models import build_model, count_parameters

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def worker(args):
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.model == 'FNO-Large':
        from neuralop.models import FNO
        model = FNO(n_modes=(512,), in_channels=1, out_channels=1,
                    hidden_channels=64, projection_channel_ratio=2,
                    n_layers=8, domain_padding=0.1)
    else:
        model = build_model(args.model)
    weights = 'random_initialization_seed42_architecture_cost_only'
    checkpoint_hash = None
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = ck.get('model_state_dict', ck.get('model_state', ck.get('state_dict', ck)))
        model.load_state_dict(state, strict=True)
        weights, checkpoint_hash = str(args.checkpoint), sha(args.checkpoint)
    stats = count_parameters(model)
    model.to('cuda').eval()
    x = torch.randn(args.batch, 1, 3000, device='cuda', dtype=torch.float32)
    props = torch.cuda.get_device_properties(0)
    row = dict(model=args.model, batch=args.batch, input_shape=list(x.shape),
               dtype='float32', seed=42, warmups=20, repeats=100,
               gpu=props.name, torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), python=sys.version,
               platform=platform.platform(), tf32=False, cudnn_benchmark=False,
               weights=weights, checkpoint_sha256=checkpoint_hash,
               source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                   [O1/'baseline_models.py', O1/'train_fno_large_bs64.py', HERE]},
               **stats)
    torch.cuda.synchronize()
    row['resident_allocated_gib'] = torch.cuda.memory_allocated()/1024**3
    try:
        with torch.inference_mode():
            for _ in range(20):
                y = model(x)
                torch.cuda.synchronize()
                del y
            torch.cuda.reset_peak_memory_stats()
            elapsed = []
            for _ in range(100):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                y = model(x)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter_ns()-start)/1e6)
                assert tuple(y.shape) == (args.batch, 1, 3000)
                del y
        row.update(status='ok', latency_median_ms=float(np.median(elapsed)),
                   latency_p95_ms=float(np.percentile(elapsed,95)),
                   throughput_samples_s=args.batch*1000/float(np.median(elapsed)),
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                   latency_samples_ms=elapsed)
    except torch.OutOfMemoryError as exc:
        row.update(status='OOM', error=str(exc),
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2), encoding='utf8')
    print(args.model, args.batch, row['status'], row.get('latency_median_ms'), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--batch', type=int)
    ap.add_argument('--checkpoint', type=Path); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.model:
        worker(args); return
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ['FNO-Large','BiLSTM','Transformer','ResCNN1D','UNet1D','MLP']:
        for batch in [1,8,64]:
            cmd = [sys.executable, str(HERE), '--model',name,'--batch',str(batch),
                   '--out',str(args.out/f'{name}_b{batch}.json')]
            subprocess.run(cmd, check=True)
    rows = [json.loads(p.read_text()) for p in sorted(args.out.glob('*_b*.json'))]
    fields = ['model','batch','status','real_params','resident_allocated_gib',
              'peak_allocated_gib','latency_median_ms','latency_p95_ms','throughput_samples_s','weights']
    with (args.out/'inference_summary.csv').open('w',newline='',encoding='utf8') as f:
        w = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

if __name__ == '__main__': main()
