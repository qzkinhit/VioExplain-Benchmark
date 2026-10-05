"""Build scripts/simproc/common.py from the TEP reference common.py by adding the V3_DATA switch (run locally)."""
import sys
REF = sys.argv[1] if len(sys.argv) > 1 else '/path/to/work'
s = open(REF).read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a, s.count(a)); s = s.replace(a, b)


rep('''Outputs (OUT/MODE/): metrics.json, records_<method>.pkl (per-window predicted sets for every evaluation group).
"""''', '''Outputs (OUT/MODE/): metrics.json, records_<method>.pkl (per-window predicted sets for every evaluation group).

Simulated processes (V3_DATA=cstr or qtank, scripts/simproc/gen_data.py): the same protocol on the data of
data/simproc/<process>/. Fit runs 0-299 and calibration runs 300-399 of training_arrays (onset 64, 7 windows per run),
test runs split by the same hash (onset 160, 12 windows), concurrent sets from test_pairs and test_triples (onset 160),
every fault effective, composed triple sets scaled to the number of fault pairs (dataset.json, tri_sets). TEP
(V3_DATA unset) is unchanged.
"""''')
rep('''EFF = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
t0 = time.time()''', '''EFF = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
DATA = os.environ.get('V3_DATA', 'tep'); ONSET_TR, ONSET_TE, NTRI, FROZ = 20, 160, 300, 30
if DATA != 'tep':
    DD = ('/home/user' if HIT else '/home/user') + '/vioexplain-v3/data/simproc/%s/' % DATA
    DS = json.load(open(DD + 'dataset.json'))
    TRAIN = DD + 'training_arrays/'; TEST = DD + 'testing_arrays/'; SIM = DD
    W = DS['W']; C = DS['C']; M = DS['M']; K = C - 1; EFF = [int(e) for e in DS['EFF']]
    ONSET_TR, ONSET_TE, NTRI = DS['onset_train'], DS['onset_test'], DS['tri_sets']; FROZ = max(1, int(np.ceil(30 * M / 52)))
t0 = time.time()''')
rep("return (Xw.std(1) < 1e-9).sum(1) >= 30", "return (Xw.std(1) < 1e-9).sum(1) >= FROZ")
rep('''FIT = runs_of('fit'); CALR = runs_of('calibration') or runs_of('cal')
TESTRUNS = list(range(500)); TCAL''', '''if DATA == 'tep':
    FIT = runs_of('fit'); CALR = runs_of('calibration') or runs_of('cal'); TESTRUNS = list(range(500))
else:
    FIT = list(range(DS['n_fit'])); CALR = list(range(DS['n_fit'], DS['n_fit'] + DS['n_cal'])); TESTRUNS = list(range(DS['n_test']))
TCAL''')
for sp, run, on in (('tr', 'FIT', 'TR'), ('cal', 'CALR', 'TR'), ('tcal', 'TCAL', 'TE'), ('ev', 'TEV', 'TE')):
    arr = 'tr_arr' if on == 'TR' else 'te_arr'; o = '20' if on == 'TR' else '160'
    rep("'%s': {c: wins(%s[c], %s, %s)" % (sp, arr, run, o), "'%s': {c: wins(%s[c], %s, ONSET_%s)" % (sp, arr, run, on))
rep("target_, got_, tries_ = int(round(300 * PCT / 100)), 0, 0", "target_, got_, tries_ = int(round(NTRI * PCT / 100)), 0, 0")
rep("log('runs fit', len(FIT),", "log('data', DATA, 'runs fit', len(FIT),")
open('common.py', 'w').write(s)
print('written common.py from', REF)
