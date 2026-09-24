#!/usr/bin/env python3
"""Re-derive stored Z300 datasets built with the wrong pixel->wavelength offset
and re-score the optimization sessions that used them (step 2 of the -18 px fix).

Root cause: reports/2026-09-23-native-axis-18px-offset.md. Every native/API and
legacy-ZIP dataset was built with ``pixels_to_wavelength`` at pixel offset 0, so
every line was labelled ~18 detector columns too blue. Step 1 (deploy) installs
the fixed pantheum/alibz/z300_calibration.py (PIXEL_OFFSET = -18) on Moissanite.
This tool then rebuilds each affected dataset's average.csv / shots/*.csv from the
raw stored pixels + calibration already on disk (no analyzer refetch is needed),
under NEW dataset ids, and re-scores every affected batch/session with Pantheum's
own optimization_metrics -- exactly the way scripts/recover-alibz-awaiting-data.py
--refetch already does for a single run.

Run on Moissanite (imports ~/pantheum-I). Default is a DRY RUN: it inventories and
classifies every acquisition, computes the Ar I NIR axis check BEFORE (offset 0)
and AFTER (offset -18) for each run, predicts score changes for a sample of
batches, and writes a JSON report -- and touches nothing. With --apply it holds
the reservation lock, refuses if any acquisition/job/hardware op is active or
unresolved, backs up the SQLite database and config, stops the alibz services,
re-derives each run into a new dataset (old files copied to a timestamped backup
inside the run dir first; old datasets are never deleted), moves the acquisition's
dataset pointer, re-scores the batches and refreshes the session proposal/best,
then restarts the services. It is idempotent: a run already re-derived is skipped.

SINGLE SOURCE OF TRUTH: --apply refuses unless the LIVE deployed
~/pantheum-I/pantheum/alibz/z300_calibration.py hashes to the fixed module
(FIXED_CALIBRATION_SHA256) and uses that live module for the conversion. For a
DRY-RUN preview before deploy, pass --preview-module PATH to a copy of the fixed
module (the .sh wrapper stages the local pantheum-I copy into a bundle dir under
~); the preview is used for reporting only and is never written to live state.

No fire, motion, cancel, settings push, or service restart is ever issued on a
dry run.
"""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FIXED_CALIBRATION_SHA256 = '449dfb4981ecfc993c07489b65ee8241b794dc87894e921be4d0371c16217beb'
PIXEL_OFFSET_BEFORE = 0
PIXEL_OFFSET_AFTER = -18
REASON = 'pixel_offset -18 (reports/2026-09-23-native-axis-18px-offset.md)'
SERVICES = ['pantheum-alibz-worker.service', 'pantheum-alibz.service']
RESAMPLED_POINTS = 23250  # the superseded linear 1/30 nm grid (200-961 nm)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Pure logic (no pantheum / no I/O) -- covered by tests/test_rederive_axis_offset.py
# ---------------------------------------------------------------------------

def classify(facts):
    """Classify one acquisition run from facts gathered off disk + the DB.

    Returns a dict with 'action' in {'rederive','skip','report_only'} and a
    'class' tag matching the report's (a)-(e) taxonomy. Pure: no I/O.
    """
    raw_available = bool(facts.get('has_raw_average') or facts.get('has_raw_shots')
                         or facts.get('has_all_zip'))
    if not raw_available:
        # (d) No raw pixels on disk: cannot re-derive; a refetch is needed while
        # the test is still on the analyzer. Report only, never refetch here.
        return {'action': 'report_only', 'class': 'd_no_raw',
                'refetchable': bool(facts.get('test_id')),
                'note': 'no raw pixels on disk; refetch needed (not done here)'}
    axis = facts.get('test_axis_check')
    if axis and axis.get('offset_px') is not None and abs(axis['offset_px']) <= 2.0:
        # (e) Already built with the fix (axis check recorded and aligned).
        return {'action': 'skip', 'class': 'e_already_correct',
                'note': 'test.json axis_check present and aligned'}
    if facts.get('opal_pixel_offset') == PIXEL_OFFSET_AFTER:
        # (e) FlatBuffers/legacy already carries -18.
        return {'action': 'skip', 'class': 'e_already_correct',
                'note': 'opal manifest pixel_offset == -18'}
    if facts.get('has_opal_manifest') and facts.get('opal_pixel_offset') in (None, 0, 0.0):
        # (b) Opal legacy ZIP/gzip-JSON built at offset 0; native raw JSON is on
        # disk so it re-derives on the same path as a native run.
        source = 'raw_json' if facts.get('has_raw_average') else 'all_zip'
        return {'action': 'rederive', 'class': 'b_opal_legacy', 'source': source}
    # (a) API/native dataset built at offset 0 (no axis_check in test.json).
    return {'action': 'rederive', 'class': 'a_api_native', 'source': 'raw_json'}


def active_work_reasons(active_acquisitions, active_jobs, unresolved_hardware):
    """Reasons to refuse --apply. Pure: takes the three counts, returns a list."""
    reasons = []
    if active_acquisitions:
        reasons.append(f'{active_acquisitions} acquisition(s) queued/running/cancelling')
    if active_jobs:
        reasons.append(f'{active_jobs} analysis job(s) queued/running')
    if unresolved_hardware:
        reasons.append(f'{unresolved_hardware} hardware operation(s) pending/uncertain')
    return reasons


def already_rederived(dataset_meta, test_axis_check):
    """Idempotence: True if this run has already been re-derived by this tool.

    Two independent signals, either sufficient: the current dataset's metadata
    carries our ``rederived_from`` link, or test.json already records an aligned
    Ar I axis check (|offset| <= 2 px). Pure: no I/O.
    """
    if isinstance(dataset_meta, dict):
        acq = dataset_meta.get('acquisition') or {}
        if dataset_meta.get('rederived_from') or acq.get('rederived_from'):
            return True
    if test_axis_check and test_axis_check.get('offset_px') is not None \
            and abs(test_axis_check['offset_px']) <= 2.0:
        return True
    return False


# ---------------------------------------------------------------------------
# Conversion module loading (single source of truth)
# ---------------------------------------------------------------------------

def load_calibration_module(path):
    """Import a z300_calibration.py by file path (for preview) and return it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_rederive_calibration', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_xy(mod, shot, offset):
    """(wavelength, intensity), strictly increasing, at the given pixel offset."""
    wl, inten = mod.pixels_to_wavelength(shot['wlCalibrations'], shot['knots'],
                                         shot['pixels'], pixel_offset=offset)
    xs, ys = [], []
    for x, y in zip(wl, inten):
        if xs and x <= xs[-1]:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def axis_check_at(mod, shot, offset):
    try:
        xs, ys = native_xy(mod, shot, offset)
        return mod.argon_axis_offset_px(xs, ys)
    except Exception as exc:  # noqa: BLE001 - a QC probe must never abort inventory
        return {'offset_px': None, 'peak_corr': 0.0, 'runner_up_corr': 0.0,
                'ambiguous': True, 'error': f'{type(exc).__name__}: {exc}'}


def _short_axis(axis):
    if not axis:
        return None
    return {'offset_px': axis.get('offset_px'), 'peak_corr': round(axis.get('peak_corr', 0.0), 3),
            'ambiguous': axis.get('ambiguous'), 'error': axis.get('error')}


# ---------------------------------------------------------------------------
# Disk facts
# ---------------------------------------------------------------------------

def gather_facts(run_dir, acq_row):
    """Facts for classify(), read off the run dir + the acquisition row."""
    run_dir = Path(run_dir)
    raw = run_dir / 'raw'
    test_json = run_dir / 'test.json'
    opal = run_dir / 'opal-manifest.json'
    test_axis_check = None
    if test_json.is_file():
        try:
            test_axis_check = json.loads(test_json.read_text()).get('axis_check')
        except (ValueError, OSError):
            test_axis_check = None
    opal_pixel_offset = None
    has_opal = opal.is_file()
    if has_opal:
        try:
            man = json.loads(opal.read_text())
            opal_pixel_offset = man.get('pixel_offset')
            if opal_pixel_offset is None:
                opal_pixel_offset = (man.get('provenance') or {}).get('pixel_offset')
        except (ValueError, OSError):
            opal_pixel_offset = None
    return {
        'state': acq_row['state'],
        'test_id': acq_row['test_id'],
        'has_dir': run_dir.is_dir(),
        'has_raw_average': (raw / 'shot--1.json').is_file(),
        'has_raw_shots': (raw / 'shot-0.json').is_file(),
        'has_all_zip': (raw / 'all.zip').is_file(),
        'has_test_json': test_json.is_file(),
        'test_axis_check': test_axis_check,
        'has_opal_manifest': has_opal,
        'opal_pixel_offset': opal_pixel_offset,
    }


def load_raw_shots(run_dir):
    """Average shot (shot--1.json) and shot-0.. from raw/, as parsed JSON."""
    raw = Path(run_dir) / 'raw'
    avg_path = raw / 'shot--1.json'
    average = json.loads(avg_path.read_text()) if avg_path.is_file() else None
    shots = []
    idx = 0
    while (raw / f'shot-{idx}.json').is_file():
        shots.append(json.loads((raw / f'shot-{idx}.json').read_text()))
        idx += 1
    return average, shots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true', help='execute (default is a dry run)')
    parser.add_argument('--run', action='append', default=[],
                        help='limit to this run id (repeatable); default is every live run')
    parser.add_argument('--preview-module', type=Path, default=None,
                        help='dry-run only: a copy of the FIXED z300_calibration.py to use for '
                             'the BEFORE/AFTER preview when the live module is not yet deployed')
    parser.add_argument('--sample', action='append', default=[],
                        help='run id to include in the predicted-score preview (repeatable)')
    parser.add_argument('--report', type=Path, default=None,
                        help='where to write the JSON report (default: a staging dir under ~)')
    parser.add_argument('--root', type=Path, default=Path.home() / 'pantheum-I',
                        help='pantheum-I checkout to import (default ~/pantheum-I)')
    args = parser.parse_args(argv)

    root = args.root
    sys.path.insert(0, str(root))
    from pantheum.alibz.service import Service  # noqa: E402
    from pantheum.alibz.acquire import _now  # noqa: E402
    from pantheum.alibz.optimization import _canonical  # noqa: E402

    live_cal = root / 'pantheum' / 'alibz' / 'z300_calibration.py'
    live_hash = digest(live_cal) if live_cal.is_file() else None
    fix_deployed = live_hash == FIXED_CALIBRATION_SHA256

    # Pick the conversion module (single source of truth).
    conv_mod = None
    conv_source = None
    if fix_deployed:
        conv_mod = load_calibration_module(live_cal)
        conv_source = f'live {live_cal} ({live_hash[:12]})'
    elif args.preview_module is not None:
        prev_hash = digest(args.preview_module)
        if prev_hash != FIXED_CALIBRATION_SHA256:
            print(json.dumps({'error': 'preview module hash mismatch',
                              'expected': FIXED_CALIBRATION_SHA256, 'got': prev_hash}))
            return 2
        conv_mod = load_calibration_module(args.preview_module)
        conv_source = f'preview {args.preview_module} ({prev_hash[:12]}) [NOT deployed]'

    if args.apply and not fix_deployed:
        print(json.dumps({'error': 'refusing --apply: live z300_calibration.py is not the fixed '
                                   'module; deploy step 1 first',
                          'live_hash': live_hash, 'expected': FIXED_CALIBRATION_SHA256}))
        return 3

    config_path = Path.home() / '.config/pantheum/alibz.json'
    config = json.loads(config_path.read_text())
    service = Service(config)
    acq = service.acquisition
    state_root = service.root
    db_path = state_root / 'alibz.sqlite'

    # ---- inventory ----
    with service.db() as db:
        acq_rows = [dict(r) for r in db.execute(
            "SELECT * FROM acquisitions WHERE run_mode='live' AND test_id IS NOT NULL "
            "ORDER BY created_at")]
        # runs with no test_id are also listed (class d) so nothing is silent.
        acq_rows += [dict(r) for r in db.execute(
            "SELECT * FROM acquisitions WHERE run_mode='live' AND test_id IS NULL "
            "ORDER BY created_at")]
        batch_rows = [dict(r) for r in db.execute('SELECT * FROM optimization_batches')]
        session_rows = {r['id']: dict(r) for r in db.execute('SELECT * FROM optimization_sessions')}
    if args.run:
        wanted = set(args.run)
        acq_rows = [r for r in acq_rows if r['id'] in wanted]
    batches_by_run = {}
    for b in batch_rows:
        batches_by_run.setdefault(b['run_id'], []).append(b)

    table = []
    counts = {}
    for row in acq_rows:
        run_id = row['id']
        run_dir = acq._run_dir(run_id)
        facts = gather_facts(run_dir, row)
        verdict = classify(facts)
        counts[verdict['class']] = counts.get(verdict['class'], 0) + 1
        before = after = None
        if conv_mod is not None and facts['has_raw_average']:
            try:
                average = json.loads((run_dir / 'raw' / 'shot--1.json').read_text())
                before = axis_check_at(conv_mod, average, PIXEL_OFFSET_BEFORE)
                after = axis_check_at(conv_mod, average, PIXEL_OFFSET_AFTER)
            except (ValueError, OSError, KeyError):
                before = after = None
        sess_ids = sorted({b['session_id'] for b in batches_by_run.get(run_id, [])})
        elements = sorted({session_rows[s]['element'] for s in sess_ids if s in session_rows})
        table.append({
            'run_id': run_id, 'created_at': row['created_at'], 'state': row['state'],
            'test_id': row['test_id'], 'sessions': sess_ids, 'elements': elements,
            'class': verdict['class'], 'action': verdict['action'],
            'note': verdict.get('note'), 'source': verdict.get('source'),
            'raw_sources': [k for k in ('raw_json_average', 'raw_json_shots', 'all_zip')
                            if {'raw_json_average': facts['has_raw_average'],
                                'raw_json_shots': facts['has_raw_shots'],
                                'all_zip': facts['has_all_zip']}[k]],
            'axis_before': _short_axis(before), 'axis_after': _short_axis(after),
        })

    rederive_runs = [t['run_id'] for t in table if t['action'] == 'rederive']
    affected_batches = [b for r in rederive_runs for b in batches_by_run.get(r, [])]
    affected_sessions = sorted({b['session_id'] for b in affected_batches})

    # ---- predicted score preview (dry-run reporting; a sample of batches) ----
    preview = []
    sample = set(args.sample)
    if not args.apply and conv_mod is not None:
        # Default sample: V_pure_run2 (by run id if given) + one batch per Fe session.
        default_sample = set()
        seen_elt = {}
        for t in table:
            if t['action'] != 'rederive':
                continue
            for elt in t['elements']:
                if elt not in seen_elt:
                    seen_elt[elt] = t['run_id']
                    default_sample.add(t['run_id'])
        sample |= default_sample
        for run_id in sorted(sample):
            entry = predict_scores(service, acq, conv_mod, run_id, batches_by_run.get(run_id, []),
                                   session_rows)
            if entry:
                preview.append(entry)

    report = {
        'tool': 'rederive-alibz-axis-offset', 'apply': args.apply,
        'generated_at': _now(), 'host': os.uname().nodename,
        'live_calibration_sha256': live_hash, 'fix_deployed': fix_deployed,
        'conversion_module': conv_source,
        'counts_by_class': counts,
        'rederive_runs': rederive_runs,
        'affected_batches': [b['id'] for b in affected_batches],
        'affected_sessions': affected_sessions,
        'closed_sessions_affected': sorted(
            {s for s in affected_sessions if session_rows.get(s, {}).get('state') == 'closed'}),
        'table': table,
        'score_preview': preview,
    }

    if not args.apply:
        report_path = args.report or (Path.home() /
                                      ('pantheum-rederive-dryrun-' + time.strftime('%Y%m%dT%H%M%S') + '.json'))
        Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
        report['report_path'] = str(report_path)
        print(json.dumps({k: report[k] for k in (
            'apply', 'fix_deployed', 'conversion_module', 'counts_by_class',
            'rederive_runs', 'affected_sessions', 'closed_sessions_affected', 'report_path')},
            indent=2))
        print(f'\ndry run only; wrote {report_path}; rerun with --apply to re-derive and re-score')
        return 0

    # ---- APPLY ----
    changes = apply_rederivation(service, acq, config_path, config, db_path, state_root,
                                 table, batches_by_run, session_rows, _canonical, _now)
    report['changes'] = changes
    report_path = args.report or (Path.home() /
                                  ('pantheum-rederive-apply-' + time.strftime('%Y%m%dT%H%M%S') + '.json'))
    Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
    report['report_path'] = str(report_path)
    print(json.dumps({'apply': True, 'report_path': str(report_path),
                      'rederived': len(changes.get('runs', [])),
                      'backup': changes.get('backup')}, indent=2))
    return 0


def predict_scores(service, acq, conv_mod, run_id, batches, session_rows):
    """Predict BEFORE/AFTER batch scores for a run without touching live state.

    BEFORE = analyze the run dir as it stands (offset-0 CSVs). AFTER = re-derive
    the shot CSVs at -18 into a temp dir and analyze that. Reporting only.
    """
    completed = [b for b in batches if b['state'] == 'completed']
    if not completed:
        return None
    run_dir = acq._run_dir(run_id)
    average, shots = load_raw_shots(run_dir)
    if not shots:
        return None
    # Mirror the count that was originally scored (analyze_batch wants 6-10
    # contiguous shots), so BEFORE and AFTER compare like with like.
    n_scored = len(list((run_dir / 'shots').glob('shot-*.csv')))
    if n_scored:
        shots = shots[:n_scored]
    try:
        from pantheum.alibz.z300 import spectrum_to_csv
    except Exception:  # noqa: BLE001
        spectrum_to_csv = None
    out = {'run_id': run_id, 'batches': []}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'shots').mkdir()
        for i, shot in enumerate(shots):
            xs, ys = native_xy(conv_mod, shot, PIXEL_OFFSET_AFTER)
            if spectrum_to_csv is not None:
                csv = spectrum_to_csv(xs, ys)
            else:
                csv = 'wavelength_nm,intensity\n' + ''.join(
                    f'{x:.6f},{y:.6f}\n' for x, y in zip(xs, ys))
            dest = tmp / 'shots' / f'shot-{i}.csv'
            dest.write_bytes(csv if isinstance(csv, bytes) else csv.encode())
        for b in completed:
            session = session_rows.get(b['session_id'], {})
            elt = session.get('element', 'Fe')
            # Composition is what Pantheum's optimizer actually scores with.
            composition = service.optimization._composition_fields(session)[0]
            before = after = None
            try:
                before = service.optimization._analyze(run_dir, composition)
            except Exception as exc:  # noqa: BLE001
                before = {'error': f'{type(exc).__name__}: {exc}'}
            try:
                after = service.optimization._analyze(tmp, composition)
            except Exception as exc:  # noqa: BLE001
                after = {'error': f'{type(exc).__name__}: {exc}'}
            out['batches'].append({
                'batch_id': b['id'], 'session_id': b['session_id'], 'element': elt,
                'delay': b['delay'], 'period': b['period'],
                'score_before': (before or {}).get('score'),
                'score_after': (after or {}).get('score'),
                'eligible_before': (before or {}).get('eligible'),
                'eligible_after': (after or {}).get('eligible'),
                'grid_before': (before or {}).get('grid'), 'grid_after': (after or {}).get('grid'),
                'before_error': (before or {}).get('error'), 'after_error': (after or {}).get('error'),
            })
    return out


def apply_rederivation(service, acq, config_path, config, db_path, state_root,
                       table, batches_by_run, session_rows, _canonical, _now):
    """Execute the re-derivation + re-score. Refuse if anything is active."""
    live = sqlite3.connect(str(db_path))
    active_acq = live.execute(
        "SELECT COUNT(*) FROM acquisitions WHERE state IN ('queued','running','cancelling')").fetchone()[0]
    active_jobs = live.execute(
        "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
    hw = live.execute(
        "SELECT COUNT(*) FROM hardware_operations WHERE state IN ('pending','uncertain')").fetchone()[0]
    live.close()
    reasons = active_work_reasons(active_acq, active_jobs, hw)
    if reasons:
        raise RuntimeError('Refusing --apply: ' + '; '.join(reasons))

    lock_path = state_root / 'reservation.lock'
    import fcntl
    lock = lock_path.open('a+b')
    deadline = time.monotonic() + 60
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise RuntimeError('reservation lock busy for 60 s; a dispatch or deploy is in progress')
            time.sleep(0.5)

    backup = Path.home() / ('pantheum-rederive-backup-' + time.strftime('%Y%m%dT%H%M%S'))
    backup.mkdir()
    shutil.copy2(config_path, backup / 'alibz.json')
    with sqlite3.connect(str(db_path)) as src, sqlite3.connect(str(backup / 'alibz.sqlite')) as dst:
        src.backup(dst)

    subprocess.run(['systemctl', '--user', 'stop', *SERVICES], check=True)
    ts = time.strftime('%Y%m%dT%H%M%S')
    result = {'backup': str(backup), 'runs': [], 'skipped': []}
    try:
        for t in table:
            if t['action'] != 'rederive':
                continue
            run_id = t['run_id']
            entry = rederive_one_run(service, acq, run_id, batches_by_run.get(run_id, []),
                                     session_rows, backup, ts, _canonical, _now)
            if entry.get('skipped'):
                result['skipped'].append(entry)
            else:
                result['runs'].append(entry)
            print(json.dumps(entry), flush=True)
    finally:
        subprocess.run(['systemctl', '--user', 'start', *reversed(SERVICES)], check=True)
    return result


def rederive_one_run(service, acq, run_id, batches, session_rows, backup, ts, _canonical, _now):
    """Re-derive one run into a new dataset and re-score its batches. Per-run
    transaction for the DB writes; file backup for the on-disk rewrite."""
    with service.db() as db:
        row = db.execute('SELECT * FROM acquisitions WHERE id=?', (run_id,)).fetchone()
        old_ids = json.loads(row['dataset_ids'] or '[]')
        old_meta = None
        test_axis = None
        if old_ids:
            ds = db.execute('SELECT metadata FROM datasets WHERE id=?', (old_ids[-1],)).fetchone()
            if ds and ds['metadata']:
                old_meta = json.loads(ds['metadata'])
    run_dir = acq._run_dir(run_id)
    test_json = run_dir / 'test.json'
    if test_json.is_file():
        try:
            test_axis = json.loads(test_json.read_text()).get('axis_check')
        except (ValueError, OSError):
            test_axis = None
    if already_rederived(old_meta, test_axis):
        return {'run': run_id, 'skipped': True, 'reason': 'already re-derived (idempotent)'}

    params = json.loads(row['params'])
    average, shots = load_raw_shots(run_dir)
    if average is None or not shots:
        return {'run': run_id, 'skipped': True, 'reason': 'no raw pixels; cannot re-derive'}

    # Back up the old on-disk files (never delete) and record hashes.
    run_backup = run_dir / f'rederive-backup-{ts}'
    run_backup.mkdir(exist_ok=True)
    old_hashes, new_hashes = {}, {}
    for rel in ['average.csv', 'test.json', 'manifest.json'] + \
            [f'shots/{p.name}' for p in sorted((run_dir / 'shots').glob('shot-*.csv'))]:
        src = run_dir / rel
        if src.is_file():
            dest = run_backup / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            old_hashes[rel] = digest(src)

    requested = params.get('numShotsPerLocation', len(shots)) * params.get('numlocations', 1)
    # _finish_dataset rewrites average.csv/shots/*.csv in place with the live
    # (fixed) module, writes provenance incl. axis_check, and creates a NEW
    # dataset id via import_bytes (which also enqueues the auto analysis engines).
    dataset_id, count, axis_warning = acq._finish_dataset(
        run_id, row['run_mode'], params, row['test_id'], average, shots, run_dir,
        requested=requested)
    for rel in list(old_hashes):
        if (run_dir / rel).is_file():
            new_hashes[rel] = digest(run_dir / rel)

    with service.db() as db:
        db.execute('BEGIN IMMEDIATE')
        # Link the new dataset back to the originals.
        ds = db.execute('SELECT metadata FROM datasets WHERE id=?', (dataset_id,)).fetchone()
        meta = json.loads(ds['metadata']) if ds and ds['metadata'] else {}
        new_axis = ((json.loads((run_dir / 'test.json').read_text()) or {}).get('axis_check')
                    if (run_dir / 'test.json').is_file() else None)
        meta['rederived_from'] = old_ids
        meta['rederive_reason'] = REASON
        meta['axis_check'] = new_axis
        db.execute('UPDATE datasets SET metadata=? WHERE id=?', (json.dumps(meta), dataset_id))
        # Move the acquisition's dataset pointer (old dataset kept).
        prior_detail = row['detail'] or ''
        db.execute('UPDATE acquisitions SET dataset_ids=?, detail=? WHERE id=?',
                   (json.dumps([dataset_id]),
                    (prior_detail + f' Re-derived at pixel_offset -18 on {_now()} '
                     f'(rederive-alibz-axis-offset); previous datasets {old_ids}.').strip(),
                    run_id))
        # Re-score each batch on this run.
        rescored = []
        touched_sessions = set()
        for b in batches:
            if b['state'] != 'completed':
                continue
            session = session_rows.get(b['session_id'])
            if session is None:
                continue
            composition = service.optimization._composition_fields(session)[0]
            metrics = service.optimization._analyze(run_dir, composition)
            old_metrics = json.loads(b['metrics']) if b['metrics'] else None
            prov = {'previous_metrics': old_metrics, 'rederived_from': old_ids, 'reason': REASON}
            new_detail = (b['detail'] or '') + \
                ' | Re-scored at pixel_offset -18; previous metrics kept in provenance: ' + \
                json.dumps(prov)
            db.execute('UPDATE optimization_batches SET dataset_ids=?, metrics=?, detail=? WHERE id=?',
                       (_canonical([dataset_id]), _canonical(metrics), new_detail, b['id']))
            rescored.append({'batch_id': b['id'], 'session_id': b['session_id'],
                             'score_old': (old_metrics or {}).get('score'),
                             'score_new': metrics.get('score'),
                             'eligible_new': metrics.get('eligible')})
            touched_sessions.add(b['session_id'])
        # Refresh proposal/best on each touched session.
        session_updates = []
        for sid in sorted(touched_sessions):
            session = db.execute('SELECT * FROM optimization_sessions WHERE id=?', (sid,)).fetchone()
            proposal, best = service.optimization._next_proposal(db, session)
            if session['state'] == 'closed':
                # Closed sessions never had `best` recomputed. Refresh `best`
                # (and proposal for the record) but DO NOT reopen: leave state
                # 'closed' so a finished study is not silently revived.
                db.execute('UPDATE optimization_sessions SET best=? WHERE id=?',
                           (_canonical(best) if best else None, sid))
                session_updates.append({'session': sid, 'state': 'closed(best-only)',
                                        'best': best})
            elif session['state'] in ('ready', 'complete'):
                db.execute('UPDATE optimization_sessions SET state=?,proposal=?,best=? WHERE id=?',
                           ('ready' if proposal else 'complete',
                            _canonical(proposal) if proposal else None,
                            _canonical(best) if best else None, sid))
                session_updates.append({'session': sid,
                                        'state': 'ready' if proposal else 'complete',
                                        'best': best, 'proposal': proposal})
            else:
                session_updates.append({'session': sid, 'state': session['state'],
                                        'note': 'left as-is (not ready/complete/closed)'})

    return {'run': run_id, 'new_dataset': dataset_id, 'previous_datasets': old_ids,
            'shots': count, 'axis_warning': axis_warning,
            'old_file_hashes': old_hashes, 'new_file_hashes': new_hashes,
            'run_backup': str(run_backup), 'rescored': rescored,
            'sessions': session_updates}


if __name__ == '__main__':
    sys.exit(main())
