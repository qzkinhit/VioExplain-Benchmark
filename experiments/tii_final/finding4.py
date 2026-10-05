"""Finding 4 from the final code: on simulated two-fault windows (TEP-C pairs, effective set of size two), how often the
first selection of a scorer trained on single faults is a true fault, and how often its residual evidence (residual
posterior with the conformal threshold, the rule used without compositions) adds a true second fault or a false one."""
import pickle, sys
from common import *
rec = pickle.load(open(sys.argv[1], 'rb'))
n = first_ok = add_true = add_false = add_any = 0
for (ev, S_all), (ev2, Zw, tru, V, causes) in zip(rec['pair'], COMP):
    assert tuple(ev) == tuple(ev2)
    for S, T in zip(S_all, tru):
        if len(T) != 2: continue
        n += 1
        if not S or S[0] not in T: continue
        first_ok += 1
        if len(S) > 1:
            add_any += 1; add_true += int(S[1] in T); add_false += int(S[1] not in T)
out = dict(windows=n, first_true=first_ok / n, second_true_given_first=add_true / max(first_ok, 1),
           second_false_given_first=add_false / max(first_ok, 1))
log('finding4', json.dumps(out)); json.dump(out, open(OUT + 'finding4.json', 'w'), indent=1)
