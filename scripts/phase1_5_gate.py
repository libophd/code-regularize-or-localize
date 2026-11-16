from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _find_results(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.rglob("probe_result.json")))
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"not found: {p}")
    if not found:
        raise FileNotFoundError(f"no probe_result.json under {paths}")
    seen: set[Path] = set()
    uniq = []
    for f in found:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    return uniq


def _fmt_extra(r: dict[str, Any]) -> str:
    ex = r.get("extra") or {}
    if "nll_delta" in ex:
        return (f"Δnll={ex['nll_delta']:+.4f} (≤{ex.get('nll_delta_threshold', 0.15)}) "
                f"aniso_ok={ex.get('anisotropy_ok')} nll_ok={ex.get('nll_ok')}")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1.5 exit gate (PASS if any probe clears).")
    ap.add_argument("paths", nargs="+", help="probe_result.json files, or a dir to search.")
    args = ap.parse_args()

    try:
        files = _find_results(args.paths)
    except FileNotFoundError as e:
        print(f"[phase1.5-gate] ERROR: {e}", file=sys.stderr)
        return 2

    print("=== Phase 1.5 exit gate ===")
    hdr = f"{'probe':<28} {'gate metric':<34} {'value':>9} {'thresh':>7} {'verdict':>7}"
    print(hdr)
    print("-" * len(hdr))
    any_pass = False
    rows = []
    for f in files:
        r = json.loads(f.read_text())
        passed = bool(r.get("passed"))
        any_pass = any_pass or passed
        rows.append((r, passed))
        extra = _fmt_extra(r)
        print(f"{r.get('probe', '?'):<28} {r.get('gate_metric', '?'):<34} "
              f"{r.get('gate_value', float('nan')):>9.4f} {r.get('gate_threshold', float('nan')):>7.3f} "
              f"{'PASS' if passed else 'FAIL':>7}"
              + (f"   {extra}" if extra else ""))

    print("-" * len(hdr))
    verdict = "PASS" if any_pass else "FAIL"
    cleared = [r.get("probe") for r, p in rows if p]
    if any_pass:
        print(f"[phase1.5-gate] VERDICT: PASS — cleared by {', '.join(cleared)}. "
              f"Systems-track premise holds; Phase 3.5 is on the table.")
    else:
        print(f"[phase1.5-gate] VERDICT: FAIL — no probe cleared across {len(files)} result(s). "
              f"Per plan §226, drop Phase 3.5 before spending 3.5a/b compute.")
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
