"""Run official-data tracks with an explicit protocol and audit trail."""
from pathlib import Path
import sys
import runpy

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"statistical": "run_cpu.py", "tranad": "run_tranad.py", "treeshap": "run_shap.py", "tep-classifiers": "run_tep_classifiers.py", "sarad": "run_sarad.py", "event-matching": "run_event_matching.py"}
USAGE = "Usage: python -m run_vioexplain.formal_runner {statistical|tranad|treeshap|tep-classifiers|sarad|event-matching} --help"

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(USAGE)
        print("statistical: CPU detection and optional BARO attribution")
        print("tranad: official network under the recorded five-epoch GPU protocol")
        print("treeshap: official TreeSHAP of Isolation Forest")
        print("tep-classifiers: conditional TEP fault classification with SVM, LDA, and Random Forest")
        print("sarad: official SARAD network under a separately recorded GPU protocol")
        print("event-matching: locked TEP event representations with interval, raw temporal, or frozen Chronos-2 costs")
        raise SystemExit(0)
    if sys.argv[1] not in TASKS:
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    task = sys.argv.pop(1)
    runpy.run_path(str(ROOT / "benchmark/evaluation" / TASKS[task]), run_name="__main__")
