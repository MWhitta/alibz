#!/usr/bin/env python
"""Rebuild the V_pure timing-study ledgers from the CORRECTED (PIXEL_OFFSET -18)
data ledger produced by the 2026-09-24 in-place re-derivation.

The corrected data ledger (``--src``) holds every re-derived run with the
recomputed Pantheum scores.  This tool selects the vanadium timing sessions and
emits the two per-run ledgers that ``cal/scripts/v_timing_study.py`` consumes,
keeping only the fields that script reads:

    run/run_id, delay, period, position, site(=location), created_at,
    score, shots_acq/shots_stored(=shots), line_count, session, lap

lap = position // 12 (each raster lap == one delay row), EXCEPT the single
delay-0/period-5 extra point (session opt-35a72aa5) which is tagged lap="extra"
so v_timing_study.py keeps it out of the 5x4 grid — matching the 2026-09-23
hand-built ledger.

Usage:
    .venv/bin/python cal/scripts/build_v_ledgers.py \
        --src <corrected>/ledger.json \
        --run2 cal/data/v-pure-run2-ledger.json \
        --run1 cal/data/v-pure-run1-ledger.json
"""
import argparse
import json
import os

# session id prefixes (the corrected ledger stores full opt-<uuid> ids)
RUN2_GRID = "opt-c86274e9"      # 5x4 delay x period grid (59 batches)
RUN2_EXTRA = "opt-35a72aa5"     # delay-0 / period-5 extra point (1 batch)
RUN1 = "opt-9d384296"           # first study (27 batches)


def to_record(r, lap):
    """Project one corrected-ledger row onto the fields v_timing_study.py uses."""
    shots = r.get("shots")
    return dict(
        run=r["run"],
        run_id=r["run"],
        session=r.get("session"),
        delay=r["delay"],
        period=r["period"],
        position=r.get("position"),
        site=r.get("location"),          # v_timing_study reads meta["site"] first
        location=r.get("location"),
        created_at=r["created_at"],
        score=r.get("score"),
        line_count=r.get("line_count"),
        shots_acq=shots,
        shots_stored=shots,              # corrected ledger stores one shots count
        lap=lap,
    )


def build(src_rows, prefixes, extra_prefix=None):
    out = []
    for r in src_rows:
        sess = r.get("session") or ""
        if extra_prefix and sess.startswith(extra_prefix):
            out.append(to_record(r, "extra"))
        elif any(sess.startswith(p) for p in prefixes):
            pos = r.get("position")
            lap = (pos // 12) if isinstance(pos, int) else None
            out.append(to_record(r, lap))
    out.sort(key=lambda r: r["created_at"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="corrected data ledger.json")
    ap.add_argument("--run2", required=True)
    ap.add_argument("--run1", required=True)
    args = ap.parse_args()

    src = json.load(open(args.src))
    run2 = build(src, [RUN2_GRID], extra_prefix=RUN2_EXTRA)
    run1 = build(src, [RUN1])

    for path, rows, label in [(args.run2, run2, "run2"), (args.run1, run1, "run1")]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(rows, open(path, "w"), indent=1)
        laps = {}
        for r in rows:
            laps[r["lap"]] = laps.get(r["lap"], 0) + 1
        print(f"[{label}] {len(rows)} rows -> {path}  laps={laps}")


if __name__ == "__main__":
    main()
