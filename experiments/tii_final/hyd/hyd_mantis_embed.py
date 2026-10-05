"""Frozen MantisV2 per-sensor embeddings of the hydraulic cycles (GPU), as embed() of gpu_run.py.
Usage: V3_MANTIS=/path/to/mantis_weights python hyd_mantis_embed.py cuda:0
Input  /path/to/vioexplain/data/hydraulic/mantis_in<TAG>.npz  S (N x M x W), standardized per sensor by hyd_mantis.py prep
Output /path/to/vioexplain/results/hyd_v1/mantis_emb<TAG>.npz E (N x M x D), float16
"""
import os, sys, time
import numpy as np
import torch
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:0'
ROOT = '/path/to/vioexplain/'
sys.path.insert(0, ROOT + 'envs')
from smoke_v3 import load_mantis
t0 = time.time(); model = load_mantis(os.environ['V3_MANTIS'], DEV)
os.makedirs(ROOT + 'results/hyd_v1', exist_ok=True)
for tag in ('', '_splitB', '_splitC'):
    S = np.load(ROOT + 'data/hydraulic/mantis_in%s.npz' % tag)['S']; N, M, W = S.shape; flat = S.reshape(-1, W); out = []
    with torch.inference_mode():
        for s in range(0, len(flat), 4096):
            t = torch.as_tensor(flat[s:s + 4096, None, :], device=DEV)
            t = torch.nn.functional.interpolate(t, size=512, mode='linear', align_corners=False)
            out.append(model(t).float().cpu().numpy().astype(np.float16))
    E = np.concatenate(out).reshape(N, M, -1)
    np.savez_compressed(ROOT + 'results/hyd_v1/mantis_emb%s.npz' % tag, E=E)
    print('embedded', tag or 'main', S.shape, '->', E.shape, 'finite', bool(np.isfinite(E.astype(np.float32)).all()), '[%.0fs]' % (time.time() - t0), flush=True)
print('DONE', flush=True)
