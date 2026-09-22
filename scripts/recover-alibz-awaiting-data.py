#!/usr/bin/env python3
"""Recover spectra for alibz runs left in ``awaiting_data`` by deferred retrieval.

Run on Moissanite (imports ~/pantheum-I). Default is a dry run: it fetches the
average and every shot of each awaiting run from the analyzer's /data/shotspectrum
endpoint (read-only GETs), converts them with the deployed code, and reports.
With --apply it stops the alibz services, backs up the SQLite database, stores
the spectra exactly as a live data_api run would (Acquisition._finish_dataset),
marks each run ``succeeded``, re-queues any ``awaiting_data`` optimization batch
and reconciles its session so the study scores the batch and proposes the next
condition, optionally switches acquire.retrieval to data_api, then restarts.
No fire, motion, cancel, or settings push is ever issued.
"""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / 'pantheum-I'
sys.path.insert(0, str(ROOT))
from pantheum.alibz.acquire import Acquisition, _now  # noqa: E402
from pantheum.alibz.optimization import _canonical  # noqa: E402
from pantheum.alibz.service import Service  # noqa: E402
from pantheum.alibz.z300 import Z300Client  # noqa: E402

SERVICES = ['pantheum-alibz-worker.service', 'pantheum-alibz.service']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', default=[], help='run id to recover (repeatable)')
    parser.add_argument('--all-awaiting', action='store_true', help='recover every awaiting_data run')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--enable-data-api', action='store_true',
                        help='with --apply: set acquire.retrieval=data_api in the private config')
    parser.add_argument('--refetch', action='store_true',
                        help='also re-fetch runs already succeeded (given with --run) and re-store them '
                             'on the native grid, re-scoring any completed optimization batch')
    args = parser.parse_args()
    config_path = Path.home() / '.config/pantheum/alibz.json'
    config = json.loads(config_path.read_text())
    service = Service(config)
    acq = service.acquisition
    cfg = acq._cfg()
    states = ('awaiting_data', 'succeeded', 'failed') if args.refetch else ('awaiting_data',)
    with service.db() as db:
        rows = [dict(r) for r in db.execute(
            f"SELECT * FROM acquisitions WHERE state IN ({','.join('?' * len(states))}) "
            "AND test_id IS NOT NULL AND run_mode='live' ORDER BY created_at", states)]
    if not args.all_awaiting:
        rows = [r for r in rows if r['id'] in set(args.run)]
    else:
        rows = [r for r in rows if r['state'] == 'awaiting_data' or r['id'] in set(args.run)]
    if not args.run and not args.all_awaiting:
        rows = []  # config-only invocation (--enable-data-api) touches no run
    if not rows and not args.enable_data_api:
        print(json.dumps({'error': 'no matching runs; use --run or --all-awaiting'})); return 1
    client = Z300Client(cfg['analyzer_url'], timeout=min(90.0, float(cfg['timeout_seconds'])))
    fetched = []
    for row in rows:
        params = json.loads(row['params'])
        total = params['numShotsPerLocation'] * params['numlocations']
        try:
            average = client.shot_spectrum(row['test_id'], -1)
            shots = [client.shot_spectrum(row['test_id'], n) for n in range(total)]
        except Z300Error as exc:
            # The instrument stored fewer shots than requested (a 404 on shot n):
            # report it and leave the run alone; it cannot become a full batch.
            stored = 0
            for n in range(total):
                try:
                    client.shot_spectrum(row['test_id'], n); stored += 1
                except Z300Error:
                    break
            print(json.dumps({'run': row['id'], 'test_id': row['test_id'], 'expected_shots': total,
                              'stored_shots': stored, 'skipped': True, 'error': str(exc)[:160]}), flush=True)
            continue
        lines = [len(Acquisition._sorted_unique_csv(s).splitlines()) for s in [average] + shots]
        if min(lines) < 1000:
            raise RuntimeError(f'{row["id"]}: a converted spectrum has only {min(lines)} lines')
        fetched.append((row, params, average, shots))
        print(json.dumps({'run': row['id'], 'test_id': row['test_id'], 'shots': total,
                          'csv_lines': lines[0], 'fetched': True}), flush=True)
    with service.db() as db:
        batches = {r['run_id']: dict(r) for r in db.execute(
            "SELECT * FROM optimization_batches WHERE state IN ('awaiting_data','completed','failed')")}
        batches = {k: v for k, v in batches.items()
                   if v['state'] == 'awaiting_data' or args.refetch}
    print(json.dumps({'apply': args.apply, 'runs': [r['id'] for r, *_ in fetched],
                      'batches_to_rescore': [b['id'] for b in batches.values() if b['run_id'] in {r['id'] for r, *_ in fetched}],
                      'enable_data_api': args.enable_data_api, 'retrieval_now': cfg['retrieval']}), flush=True)
    if not args.apply:
        return 0
    state = service.root
    db_path = state / 'alibz.sqlite'
    live = sqlite3.connect(str(db_path))
    active = live.execute("SELECT COUNT(*) FROM acquisitions WHERE state IN ('queued','running','cancelling')").fetchone()[0]
    pending = live.execute("SELECT COUNT(*) FROM hardware_operations WHERE state IN ('pending')").fetchone()[0]
    live.close()
    if active or pending:
        raise RuntimeError('Active acquisition or pending hardware operation; refusing')
    backup = Path.home() / ('pantheum-recover-backup-' + time.strftime('%Y%m%dT%H%M%S'))
    backup.mkdir()
    shutil.copy2(config_path, backup / 'alibz.json')
    with sqlite3.connect(str(db_path)) as src, sqlite3.connect(str(backup / 'alibz.sqlite')) as dst:
        src.backup(dst)
    subprocess.run(['systemctl', '--user', 'stop', *SERVICES], check=True)
    results = []
    try:
        for row, params, average, shots in fetched:
            run_id = row['id']
            work_dir = acq._run_dir(run_id)
            (work_dir / 'raw').mkdir(parents=True, exist_ok=True)
            (work_dir / 'shots').mkdir(parents=True, exist_ok=True)
            dataset_id, count = acq._finish_dataset(run_id, row['run_mode'], params, row['test_id'],
                                                    average, shots, work_dir)
            acq._set_state(run_id, 'succeeded', finished_at=_now(), test_id=row['test_id'],
                           dataset_ids=json.dumps([dataset_id]), shots=count,
                           detail=(f'Fired earlier under acquire.retrieval=deferred as analyzer test '
                                   f'{row["test_id"]}; {count} shots recovered from /data/shotspectrum '
                                   f'at {_now()} by recover-alibz-awaiting-data.'))
            entry = {'run': run_id, 'dataset_id': dataset_id, 'shots': count,
                     'previous_dataset_ids': json.loads(row['dataset_ids'] or '[]')}
            batch = batches.get(run_id)
            if batch and batch['state'] == 'completed':
                # Re-score a batch that was completed on the old resampled grid so
                # the session compares like with like; then refresh its proposal.
                session_id = batch['session_id']
                with service.db() as db:
                    session = db.execute('SELECT * FROM optimization_sessions WHERE id=?', (session_id,)).fetchone()
                    metrics = service.optimization._analyze(acq._run_dir(run_id), session['element'])
                    ok = service.optimization._valid_metrics(metrics) and metrics['shot_count'] == 10 \
                        and metrics['eligible'] and metrics['score'] is not None
                    db.execute('UPDATE optimization_batches SET dataset_ids=?,metrics=?,detail=? WHERE id=?',
                               (_canonical([dataset_id]), _canonical(metrics),
                                batch['detail'] + ' Re-scored on the native grid.', batch['id']))
                    if session['state'] in ('ready', 'complete'):
                        proposal, best = service.optimization._next_proposal(db, session)
                        db.execute('UPDATE optimization_sessions SET state=?,proposal=?,best=? WHERE id=?',
                                   ('ready' if proposal else 'complete',
                                    _canonical(proposal) if proposal else None,
                                    _canonical(best) if best else None, session_id))
                entry.update(session=session_id, rescored=ok, score=metrics.get('score'),
                             grid=metrics.get('grid'), proposal=proposal if session['state'] in ('ready', 'complete') else None)
            elif batch and batch['state'] == 'failed':
                # A run the retrieval gave up on: the spectra exist after all, so
                # re-queue the batch and let the optimizer score it.
                with service.db() as db:
                    db.execute("UPDATE optimization_batches SET state='queued', detail=? WHERE id=?",
                               ('Spectra recovered from the analyzer after retrieval gave up; scoring.', batch['id']))
                    db.execute("UPDATE optimization_sessions SET state='acquiring' WHERE id=?", (batch['session_id'],))
                view = service.optimization._reconcile(batch['session_id'])
                scored = next((b for b in view.get('batches', []) if b['id'] == batch['id']), {})
                entry.update(session=batch['session_id'], session_state=view.get('state'),
                             proposal=view.get('proposal'), batch_state=scored.get('state'),
                             score=(scored.get('metrics') or {}).get('score'))
            elif batch:
                with service.db() as db:
                    db.execute("UPDATE optimization_batches SET state='queued', detail=? WHERE id=?",
                               ('Spectra recovered from the analyzer; rescoring.', batch['id']))
                    db.execute("UPDATE optimization_sessions SET state='acquiring' WHERE id=?",
                               (batch['session_id'],))
                view = service.optimization._reconcile(batch['session_id'])
                scored = next((b for b in view.get('batches', []) if b['id'] == batch['id']), {})
                entry.update(session=batch['session_id'], session_state=view.get('state'),
                             proposal=view.get('proposal'), batch_state=scored.get('state'),
                             score=(scored.get('metrics') or {}).get('score'),
                             batch_detail=scored.get('detail'))
            results.append(entry)
            print(json.dumps(entry), flush=True)
        if args.enable_data_api:
            config['acquire']['retrieval'] = 'data_api'
            tmp = config_path.with_name(config_path.name + '.recover-tmp')
            tmp.write_text(json.dumps(config, indent=2) + '\n')
            os.replace(tmp, config_path)
            print(json.dumps({'config': str(config_path), 'retrieval': 'data_api',
                              'config_sha256': digest(config_path)}), flush=True)
    finally:
        subprocess.run(['systemctl', '--user', 'start', *reversed(SERVICES)], check=True)
    print(json.dumps({'backup': str(backup), 'recovered': results}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
