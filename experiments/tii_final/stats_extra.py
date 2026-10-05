"""Statistics quoted in Example 1 and the dataset table, computed with the final windows and violation definition."""
from common import *
out = {}
g = ok_tr[1]; V1 = tviol_raw(RAW['tr'][1][g]); V0 = tviol_raw(RAW['tr'][0][g]); caused = V1 & ~V0
out['f1_caused_mean'] = float(caused.sum(1).mean()); out['f1_all_mean'] = float(V1.sum(1).mean())
out['f1_sensors_mean'] = float(V1.reshape(len(V1), M, 6).any(2).sum(1).mean())
Vn = tviol_raw(RAW['tr'][0]); out['normal_mean'] = float(Vn.sum(1).mean()); out['normal_share'] = float(Vn.any(1).mean())
Vn_e = tviol_raw(RAW['ev'][0]); out['normal_share_eval'] = float(Vn_e.any(1).mean())
sub = []
for c in (3, 9, 15):
    e = (tviol_raw(RAW['ev'][c]) & ~Vn_e).any(1); sub.append(e[~FR['ev'][c]])
out['sub_share'] = float(np.concatenate(sub).mean())
out['n_single_eval'] = int(len(EV_Z)); out['n_pair'] = int(sum(len(c[1]) for c in COMP)); out['n_triple'] = int(sum(len(c[1]) for c in TRIP))
out['n_quad'] = int(sum(len(c[1]) for c in QUAD)); out['n_tcal'] = int(len(TC_Z)); out['relations'] = NR
log('stats', json.dumps(out)); json.dump(out, open(OUT + 'stats_extra.json', 'w'), indent=1)
