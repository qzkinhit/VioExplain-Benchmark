"""Fold h<H>_<METHOD>_part2.json (helper runs of other n values) into h<H>_<METHOD>.json; the main file wins on overlap.
The part2 file is renamed to *.folded so the merge reads every n from the main file."""
import json, glob, os, sys
U = sys.argv[1]
for f in sorted(glob.glob(U + '/h*_part2.json')):
    base = f[:-len('_part2.json')] + '.json'
    if not os.path.exists(base): print('no main file for', os.path.basename(f), '(left as is)'); continue
    p = json.load(open(f)); b = json.load(open(base))
    for k, v in p['curve'].items(): b['curve'].setdefault(k, v)
    b['helper_runs'] = sorted(set(b.get('helper_runs', [])) | {os.path.basename(f)})
    json.dump(b, open(base, 'w'), indent=1); os.rename(f, f + '.folded')
    print(os.path.basename(base), sorted(b['curve'].keys()))
