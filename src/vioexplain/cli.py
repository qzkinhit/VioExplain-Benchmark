"""Small reproducible smoke entry point. This produces no benchmark score."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from . import Representation, Violation, explain


def smoke() -> dict:
    knowledge = [Representation({"d1", "d2"}, 1.0, 2), Representation({"d1", "d2", "d4"}, 2.0, 1)]
    violations = [Violation(d, "domain", 1, 1.0, (1.0, 1.0), 3) for d in ("d1", "d2", "d3")]
    result = explain(violations, knowledge)
    if result.selected != (0,) or result.unmatched_dims != ("d3",) or result.covered != 2:
        raise RuntimeError("synthetic partial-explanation smoke failed")
    return {"schema_version": 1, "purpose": "synthetic_smoke_only", "result": result.to_dict()}


def prototype_smoke() -> dict:
    from .setcover import demo_data
    from .setcover.set_covering import MyCover
    rules = demo_data.demo_rules()
    rows = []
    for i, candidates in enumerate(demo_data.demo_explanation_sets(), 1):
        cost, selected = MyCover(candidates, demo_data.demo_constraints(), demo_data.demo_constraints(), rules)
        rows.append({"fixture": i, "prototype_cost": cost, "selected_events": [r.name for r in selected]})
    return {"schema_version": 1, "purpose": "synthetic_prototype_smoke_only", "method": "AEC-Prototype", "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "prototype-smoke"])
    parser.add_argument("--out", type=Path, help="optional JSON path; an existing file is not overwritten")
    args = parser.parse_args()
    result = smoke() if args.command == "smoke" else prototype_smoke()
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
