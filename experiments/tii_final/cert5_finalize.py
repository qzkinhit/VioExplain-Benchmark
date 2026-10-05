"""Write results/final_v4/certificate_final.json from the runs of final_cert5.py.

Usage: python cert5_finalize.py PRIMARY_TAG [OTHER_TAG ...]
Flat top-level keys (fractions) come from the primary run; every run is kept under 'runs'.
"""
import sys, json
R = '/path/to/vioexplain/results/final_v4/'
tags = sys.argv[1:]
runs = {t: json.load(open(R + 'cert_work/%s/f100t/certificate_run.json' % t)) for t in tags}
p = runs[tags[0]]; gp = p['groups']['pair']; gt = p['groups']['triple']
tp = gp['interaction_terciles']['terciles']; tt = gt['interaction_terciles']['terciles']
out = dict(
    pair_cert_share=gp['certified'],
    pair_cert_recovered_share=gp['recovered_given_certified'],
    pair_recovered_cert_share=gp['certified_given_recovered'],
    pair_cert_share_low_interaction=tp[0]['certified'],
    pair_cert_share_mid_interaction=tp[1]['certified'],
    pair_cert_share_high_interaction=tp[2]['certified'],
    triple_cert_share=gt['certified'],
    triple_cert_recovered_share=gt['recovered_given_certified'],
    triple_recovered_cert_share=gt['certified_given_recovered'],
    triple_cert_share_low_interaction=tt[0]['certified'],
    triple_cert_share_mid_interaction=tt[1]['certified'],
    triple_cert_share_high_interaction=tt[2]['certified'],
    pair_recovered_share=gp['recovered_exact'], triple_recovered_share=gt['recovered_exact'],
    pair_n_windows=gp['n_windows'], triple_n_windows=gt['n_windows'],
    pair_n_certified_not_recovered=gp['n_certified_not_recovered'],
    triple_n_certified_not_recovered=gt['n_certified_not_recovered'],
    pair_not_recovered_first_fail=gp['not_recovered_first_fail'],
    triple_not_recovered_first_fail=gt['not_recovered_first_fail'],
    pair_not_recovered_first_fail_by_step=gp['not_recovered_first_fail_by_step'],
    triple_not_recovered_first_fail_by_step=gt['not_recovered_first_fail_by_step'],
    pair_eff_traced=gp['implementation_check']['eff_traced'], pair_eff_stored=gp['implementation_check']['eff_stored_metrics'],
    triple_eff_traced=gt['implementation_check']['eff_traced'], triple_eff_stored=gt['implementation_check']['eff_stored_metrics'],
    pair_sets_identical_to_stored_share=gp['implementation_check']['stored_identical_set_share'],
    triple_sets_identical_to_stored_share=gt['implementation_check']['stored_identical_set_share'],
    pair_cert_share_omega_over_all_events=gp['certified_omega_over_all_events'],
    triple_cert_share_omega_over_all_events=gt['certified_omega_over_all_events'],
    pair_certified_recovered_in_stored_records=gp['implementation_check']['certified_windows_recovered_in_stored_records'],
    triple_certified_recovered_in_stored_records=gt['implementation_check']['certified_windows_recovered_in_stored_records'],
    primary_run=tags[0],
    notes=[
        'Retrains are not bit-identical to the stored run (results/final_v4/f100t/log_B.txt, records_VioExplain-nostep-B.pkl). '
        'Run nj8 (LightGBM 8 threads, BLAS 3) reproduces the stored scorer exactly (label macro F1 0.918962931718175), TAU0 and '
        'TAU0F to 1e-10 and the decision training set (105643 rows, 4139 positives), but TAU_C is 0.0018629 against 0.0015719. '
        'Run nj3 (3 threads, the setting of the stored run) differs already in the scorer. Two further runs that load the nj8 '
        'models and refit only the addition decision with 3 threads and forced col-wise or row-wise histograms (cert_work/hybcol, '
        'hybrow) give the nj8 thresholds and traces exactly, so the decision fit itself is deterministic and the TAU_C gap comes '
        'from the BR detectors or scorer of the stored run, whose models were not saved.',
        'Inside every run the traced sets equal explain_t() of final_fuse5.py on all checked windows (5426 pair, 3741 triple).',
        'omega_S is the maximum over unselected events, as the proof of Theorem 3 uses. The supplement defines omega_j as a maximum '
        'over all events E in the event set; with selected events included (their scores are masked by the algorithm and never '
        'seen in the decision training) the certificate holds on the shares in *_cert_share_omega_over_all_events.',
        'Run smoke (6 trees per model) only tested the code and is not reported.'],
    description=(
        'Recovery certificate of Theorem 3 on TEP-C pairs and triples with the final pipeline (final_fuse5.py tag B: '
        'V3_MODE=f100t V3_ABL=full V3_FUNC=1 V3_NOTWIN=0 V3_NOSTEP=1 V3_TFEAT=0, ResNet posterior resnet_probs_v2.npz). '
        'Script final_cert5.py executes final_fuse5.py verbatim up to its thresholds and traces explain_t. '
        'H is the effective set of eff_truth, windows with empty H are excluded. Conditions: p_0 < TAU0F on the mixed '
        'posterior; eta_0 = best true minus best false posterior > 0; at every visited prefix S strictly inside H, '
        'eta_S = min(best true z_ref minus best false z_ref, best true z_ref minus TAU_C) > 2 omega_S with z_ref on '
        'psi_S = Fn(paired run in which only H minus S act) and omega_S the largest |z_act - z_ref| over unselected '
        'events; at S = H, eta_H = TAU_C minus best false z_ref on the paired fault-free description > omega_H. '
        'A window is recovered when the output set equals H. first_fail categories: first_choice (eta_0 <= 0 at step 0 '
        'or reference selection lead <= 0 at a prefix), stop (p_0 >= TAU0F, best true z_ref <= TAU_C at a prefix, '
        'eta_H <= 0), perturbation (reference margin positive but not above the omega bound). Interaction terciles use '
        'I = ||phi(X_E) - sum_e phi(X_e) + (m-1) phi(X_0)|| / sqrt(sum_e ||phi(X_e) - phi(X_0)||^2) with phi = Fn clipped '
        'to [-50, 50], terciles over the windows with non-empty H; interaction_terciles_unclipped repeats it without '
        'clipping. certified_omega_over_all_events takes omega over all 20 events (selected ones included). '
        'certified_rule_of_earlier_analysis uses the two inequalities of results/final_v1/certificate.json. '
        'Per-window records: cert_work/<run>/f100t/certificate_windows.pkl.'),
    runs=runs)
json.dump(out, open(R + 'certificate_final.json', 'w'), indent=1)
for k, v in out.items():
    if k not in ('runs', 'description'): print(k, v)
