"""E4: reproduce original checkpoints; train a controlled rotary counterpart.

No upstream model is modified. Physical-time RoPE is an evaluation of the
same trained weights using time in units of the training patch duration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[4]
for p in [ROOT, ROOT / 'module', ROOT / '6-code/shared/upstream/Seismic-FNO-clean',
          ROOT / '6-code/shared/upstream/Seismic-FNO-clean/module',
          ROOT / '6-code/O1-baseline-comparison/scripts', ROOT / '6-code/shared/utils']:
    if p.exists():
        sys.path.insert(0, str(p))
from baseline_models import PatchTransformer, count_parameters
from splits import make_fixed_474_split, save_split
from train_baselines import set_seed, run_epoch, save_checkpoint
from metrics import RegressionMeter, PFAMeter


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class Pairs(Dataset):
    def __init__(self, gm, response, ids, hz=50, cache_x=None, cache_y=None):
        self.paths = gm, response
        self.ids = ids
        self.hz = hz
        self.files = None
        self.cache = ((np.load(cache_x, mmap_mode='r'), np.load(cache_y, mmap_mode='r'))
                      if cache_x and cache_y else None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        if self.cache:
            x, y = self.cache[0][idx], self.cache[1][idx]
        else:
            if self.files is None:
                self.files = h5py.File(self.paths[0], 'r'), h5py.File(self.paths[1], 'r')
            x = np.asarray(self.files[0]['Acc_GMs'][idx], dtype=np.float32)
            y = np.asarray(self.files[1]['Acc_Floor_Response'][idx], dtype=np.float32)
        if self.hz == 20:
            x = resample_poly(x, 2, 5)
            y = resample_poly(y, 2, 5)
        return torch.from_numpy(np.array(x, dtype=np.float32))[:, None], {}, \
            torch.from_numpy(np.array(y, dtype=np.float32))[:, None], torch.tensor(0)


def rotate(t, positions):
    # Interleaved pairs; q and k receive the same rotation, preserving norms.
    d = t.shape[-1]
    inv = 10000.0 ** (-torch.arange(0, d, 2, device=t.device, dtype=t.dtype) / d)
    angle = positions[:, None] * inv[None, :]
    cos, sin = angle.cos()[None, None], angle.sin()[None, None]
    even, odd = t[..., ::2], t[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class RotaryTransformer(PatchTransformer):
    def __init__(self, seq_len=1200, hz=20, physical=False, **kwargs):
        super().__init__(seq_len=seq_len, **kwargs)
        # Initialisation consumes the same random draws as the original model.
        del self.pos_embed
        self.hz, self.physical = hz, physical
        self.use_rope = True
        self.check_position_table = None

    def forward(self, x):
        batch, channels, length = x.shape
        pad = (-length) % self.patch_size
        x = F.pad(x, (0, pad)) if pad else x
        patches = x.shape[-1] // self.patch_size
        x = x.reshape(batch, channels, patches, self.patch_size).permute(0, 2, 1, 3)
        x = self.input_norm(self.patch_embed(x.flatten(-2)))
        if self.check_position_table is not None:
            x = x + self.check_position_table[:, :patches]
        positions = torch.arange(patches, device=x.device, dtype=x.dtype)
        if self.physical:
            positions = (positions + 0.5) * 20.0 / self.hz - 0.5
        for layer in self.encoder.layers:
            z = layer.norm1(x)
            q, k, v = F.linear(z, layer.self_attn.in_proj_weight,
                              layer.self_attn.in_proj_bias).chunk(3, -1)
            heads = layer.self_attn.num_heads
            def split(t):
                return t.reshape(batch, patches, heads, -1).transpose(1, 2)
            q, k, v = split(q), split(k), split(v)
            if self.use_rope:
                q, k = rotate(q, positions), rotate(k, positions)
            z = F.scaled_dot_product_attention(q, k, v,
                    dropout_p=layer.self_attn.dropout if self.training else 0.0)
            z = z.transpose(1, 2).reshape(batch, patches, -1)
            z = layer.self_attn.out_proj(z)
            x = x + layer.dropout1(z)
            z = layer.norm2(x)
            z = layer.linear2(layer.dropout(layer.activation(layer.linear1(z))))
            x = x + layer.dropout2(z)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        x = self.output_proj(self.output_norm(x))
        return x.reshape(batch, patches, self.out_channels, self.patch_size).permute(
            0, 2, 1, 3).reshape(batch, self.out_channels, -1)[..., :length]


def loader(args, ids, hz=50, shuffle=False):
    ds = Pairs(args.gm, args.response, ids, hz, args.cache_x, args.cache_y)
    return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=True, persistent_workers=args.workers > 0)


def evaluate(model, batches, hz=50, wrapped=False):
    reg, pfa = RegressionMeter(), PFAMeter()
    err, ref = np.zeros(3), np.zeros(3)
    n = 0
    model.eval()
    with torch.inference_mode():
        for x, _, y, _ in batches:
            x, y = x.permute(0, 2, 1).cuda(), y.permute(0, 2, 1).cuda()
            if wrapped:
                low = resample_poly(x.cpu().numpy(), 2, 5, axis=-1)
                pred = model(torch.from_numpy(low).cuda())
                pred = torch.from_numpy(resample_poly(pred.cpu().numpy(), 5, 2, axis=-1)).cuda()
            else:
                pred = model(x)
            if not torch.isfinite(pred).all():
                raise RuntimeError('Nonfinite prediction')
            reg.update(pred, y)
            pfa.update(pred, y)
            # Integer grid / nominal duration avoids float32 frequency rounding
            # accidentally putting the Nyquist bin above hz/2 and dropping it.
            ff = torch.arange(y.shape[-1]//2+1, device='cuda', dtype=torch.float64) * hz / y.shape[-1]
            de = torch.abs(torch.fft.rfft(pred - y)) ** 2
            rf = torch.abs(torch.fft.rfft(y)) ** 2
            weights = torch.full_like(ff, 2.0)
            weights[0] = 1.0
            if y.shape[-1] % 2 == 0:
                weights[-1] = 1.0
            for b, (low, high) in enumerate([(0, 256/72), (256/72, 10), (10, hz/2)]):
                mask = ((ff >= low) if b == 0 else (ff > low)) & (ff <= high)
                err[b] += float((de[..., mask] * weights[mask]).double().sum().item())
                ref[b] += float((rf[..., mask] * weights[mask]).double().sum().item())
            n += y.shape[0]
    out = {**reg.compute(), **pfa.compute(), 'n_samples': n}
    for b in range(3):
        out[f'band{b}_error_energy_ratio'] = err[b] / ref[b] if ref[b] else None
        out[f'band{b}_reference_energy'] = ref[b] / (int(60 * hz) * hz)
        out[f'band{b}_reference_energy_share'] = ref[b] / sum(ref) if sum(ref) else None
    return out


def load_original(name, hz, args, eval_len=3000):
    ckpt = Path(args.original_root) / name / f'S-{hz}Hz/checkpoints/{name}_S-{hz}Hz_best.pt'
    from train_eval_sampling_ood import build_sampling_model, interpolate_pos_embed
    model = build_sampling_model(name, eval_len)
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = dict(saved['model_state_dict'])
    if name == 'Transformer':
        state['pos_embed'] = interpolate_pos_embed(state['pos_embed'], model.n_patches)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval(), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['check', 'original', 'train', 'rotary'], required=True)
    ap.add_argument('--gm', required=True)
    ap.add_argument('--response', required=True)
    ap.add_argument('--cache-x')
    ap.add_argument('--cache-y')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--output', required=True)
    ap.add_argument('--original-root', default=str(ROOT / '6-code/O3-generalization/results/sampling_ood/runs'))
    ap.add_argument('--epochs', type=int, default=50)
    args = ap.parse_args()
    torch.set_num_threads(8)
    set_seed(42)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / '6-code/O3-generalization/scripts'))
    split = make_fixed_474_split(seed=42)
    save_split(split, out / 'split')
    if args.mode == 'check':
        model = RotaryTransformer().cuda()
        x = torch.randn(64, 1, 1200, device='cuda')
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        t = time.time()
        for _ in range(3):
            opt.zero_grad()
            model(x).square().mean().backward()
            opt.step()
        torch.cuda.synchronize()
        model.eval()
        with torch.inference_mode():
            y = model(x)
        report = {'forward_shape': list(y.shape), 'finite': bool(y.isfinite().all()),
                  'seconds_per_step': (time.time()-t)/3,
                  'peak_gib': torch.cuda.max_memory_allocated()/1024**3,
                  'parameters': count_parameters(model),
                  'torch': torch.__version__, 'cuda': torch.version.cuda}
        # Verify the custom residual/attention path against the original model,
        # and the norm and relative-position properties of the rotation itself.
        original = PatchTransformer(seq_len=1200).cuda().eval()
        state = dict(original.state_dict())
        positions_table = state.pop('pos_embed')
        model.load_state_dict(state, strict=True)
        model.use_rope = False
        model.check_position_table = positions_table
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode():
            a, b = original(x[:2]), model(x[:2])
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
            q, k = torch.randn(2, 8, 80, 32, device='cuda', dtype=torch.float64), \
                   torch.randn(2, 8, 80, 32, device='cuda', dtype=torch.float64)
            p = torch.arange(80, device='cuda', dtype=torch.float64)
            qr, kr = rotate(q, p), rotate(k, p)
            torch.testing.assert_close(qr.square().sum(-1), q.square().sum(-1))
            dot1 = qr @ kr.transpose(-2, -1)
            dot2 = rotate(q, p+17) @ rotate(k, p+17).transpose(-2, -1)
            torch.testing.assert_close(dot1, dot2, atol=1e-10, rtol=1e-10)
        torch.backends.mha.set_fastpath_enabled(True)
        report['manual_attention_max_diff'] = float((a-b).abs().max())
        report['rotary_norm_and_relative_position_checks'] = 'passed'
        (out/'preflight.json').write_text(json.dumps(report, indent=2))
        print(report, flush=True)
        return
    if args.mode == 'original':
        rows = []
        test = loader(args, split.test_indices)
        for name in ['FNO-Large', 'Transformer']:
            for hz in [20, 50]:
                model, ckpt = load_original(name, hz, args)
                for wrapped in ([False, True] if hz == 20 else [False]):
                    print('EVALUATE', name, hz, wrapped, flush=True)
                    if wrapped and name == 'Transformer':
                        model, ckpt = load_original(name, hz, args, 1200)
                    rows.append({'model': name, 'train_hz': hz,
                                 'eval_hz': 50, 'inference': 'wrapped' if wrapped else 'direct',
                                 'checkpoint_sha256': sha(ckpt), **evaluate(model, test, wrapped=wrapped)})
                    write_csv(out/'original_resolution_results.csv', rows)
                del model
                torch.cuda.empty_cache()
        return
    if args.mode == 'train':
        train = loader(args, split.train_indices, 20, True)
        val = loader(args, split.val_indices, 20)
        model = RotaryTransformer().cuda()
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(optim, 20, 0.5)
        config = {'model': 'Transformer-RoPE', 'epochs': args.epochs, 'train_hz': 20,
                  'patch_size': 15, 'd_model': 256, 'n_heads': 8, 'n_layers': 10,
                  'seed': 42, 'batch_size': 64, 'lr': 3e-4, 'weight_decay': 1e-4,
                  'scheduler_step': 20, 'scheduler_gamma': 0.5, 'grad_clip': 1.,
                  'rope_base': 10000, 'positions': 'token index during 20Hz training',
                  'precision': 'FP32', 'selected_by': '20Hz validation MSE',
                  'parameters': count_parameters(model), 'script_sha256': sha(__file__)}
        (out/'rotary_training_config.json').write_text(json.dumps(config, indent=2))
        best, rows = float('inf'), []
        for epoch in range(1, args.epochs+1):
            t = time.time()
            tr = run_epoch(model, train, nn.MSELoss(), optim, 'cuda', '', False, 0, 1.)
            vl = run_epoch(model, val, nn.MSELoss(), None, 'cuda', '', False)
            sched.step()
            if not np.isfinite(vl['mse']):
                raise RuntimeError('Nonfinite validation MSE')
            if vl['mse'] < best:
                best = vl['mse']
                save_checkpoint(out/'rotary_best.pt', model, optim, epoch, vl, config)
            rows.append({'epoch': epoch, 'train_mse': tr['mse'], 'val_mse': vl['mse'],
                         'val_r2': vl['r2'], 'seconds': time.time()-t,
                         'peak_gib': torch.cuda.max_memory_allocated()/1024**3})
            write_csv(out/'rotary_training_log.csv', rows)
            print(rows[-1], flush=True)
        (out/'rotary_training_complete.json').write_text(json.dumps({'best_val': best, 'epochs': args.epochs}))
        return
    ckpt = torch.load(out/'rotary_best.pt', map_location='cpu', weights_only=False)
    model = RotaryTransformer().cuda()
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    rows = []
    model.hz, model.physical = 20, False
    rows.append({'model': 'Transformer-RoPE', 'train_hz': 20, 'eval_hz': 20,
                 'inference': 'matched', **evaluate(model, loader(args, split.test_indices, 20), 20)})
    for physical in [False, True]:
        model.hz, model.physical = 50, physical
        rows.append({'model': 'Transformer-RoPE', 'train_hz': 20, 'eval_hz': 50,
                     'inference': 'direct_time' if physical else 'direct_index',
                     **evaluate(model, loader(args, split.test_indices))})
    model.hz, model.physical = 20, False
    rows.append({'model': 'Transformer-RoPE', 'train_hz': 20, 'eval_hz': 50,
                 'inference': 'wrapped', **evaluate(model, loader(args, split.test_indices), wrapped=True)})
    del model
    torch.cuda.empty_cache()
    for name in ['Transformer', 'FNO-Large']:
        original, path = load_original(name, 20, args, 1200)
        rows.append({'model': name, 'train_hz': 20, 'eval_hz': 20,
                     'inference': 'matched', **evaluate(original, loader(args, split.test_indices, 20), 20)})
        del original
        torch.cuda.empty_cache()
    write_csv(out/'rotary_resolution_results.csv', rows)


if __name__ == '__main__':
    main()
