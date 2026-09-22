#!/usr/bin/env python3
"""Mark a stuck awaiting_data run failed, drop a period from its session grid,
and reconcile the session so it proposes its next condition.

Run on Moissanite AFTER the continue-after-failure optimizer is deployed (the
script refuses otherwise). Default is a dry run. It never fires, moves, or
cancels hardware; the analyzer test id stays recorded on the run.
"""
import argparse
import http.client
import json
import os
import socket
import sqlite3
import sys
import time
from pathlib import Path

STATE = Path.home() / '.local/state/pantheum/alibz'
SOCK = Path.home() / '.local/state/pantheum/alibz-gateway/alibz.sock'
MARKER = '_fail_and_continue'


class UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(30)
        self.sock.connect(str(SOCK))


def get(path):
    conn = UnixHTTP('localhost')
    conn.request('GET', path, headers={'X-Portal-User': 'mwhittaker@lbl.gov'})
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    if response.status != 200:
        raise RuntimeError((path, response.status, body))
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--drop-period', type=int, default=None)
    parser.add_argument('--reason', required=True, help='why the run is failed (goes into run detail)')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    source = Path.home() / 'pantheum-I/pantheum/alibz/optimization.py'
    if MARKER not in source.read_text():
        print(json.dumps({'error': 'deployed optimization.py lacks the continue-after-failure rule; deploy first'}))
        return 1
    db = sqlite3.connect(str(STATE / 'alibz.sqlite'))
    db.row_factory = sqlite3.Row
    run = db.execute('SELECT id,state,test_id,retrieval_claim,retrieval_attempts,detail FROM acquisitions WHERE id=?',
                     (args.run,)).fetchone()
    session = db.execute('SELECT id,state,delays,periods,proposal FROM optimization_sessions WHERE id=?',
                         (args.session,)).fetchone()
    if run is None or session is None:
        print(json.dumps({'error': 'run or session not found'})); return 1
    periods = json.loads(session['periods'])
    plan = {'apply': args.apply, 'run': dict(run), 'session': {k: session[k] for k in session.keys()},
            'new_periods': [p for p in periods if p != args.drop_period] if args.drop_period else periods}
    print(json.dumps(plan, default=str), flush=True)
    already_terminal = run['state'] in ('failed', 'cancelled', 'uncertain')
    if run['state'] != 'awaiting_data' and not (already_terminal and args.drop_period is not None):
        print(json.dumps({'error': f"run is {run['state']}, not awaiting_data"})); return 1
    if already_terminal:
        print(json.dumps({'note': f"run is already {run['state']}; only the grid change and reconcile will run"}), flush=True)
    if not args.apply:
        db.close(); return 0
    if not already_terminal:
        detail = (f"{args.reason} Analyzer test {run['test_id']} stays on the instrument; "
                  f"marked failed by the operator after {run['retrieval_attempts']} retrieval attempts.")
        changed = 0
        for _ in range(20):  # the worker holds a claim only for the ~2 s of an attempt
            with db:
                changed = db.execute("UPDATE acquisitions SET state='failed',finished_at=?,retrieval_claim=NULL,"
                                     "retrieval_error=?,detail=? WHERE id=? AND state='awaiting_data' AND retrieval_claim IS NULL",
                                     (time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()), args.reason, detail, args.run)).rowcount
            if changed:
                break
            time.sleep(1)
        if not changed:
            print(json.dumps({'error': 'could not claim the run; retrieval kept it busy'})); return 1
    if args.drop_period is not None and args.drop_period in periods:
        with db:
            db.execute('UPDATE optimization_sessions SET periods=? WHERE id=? AND periods=?',
                       (json.dumps([p for p in periods if p != args.drop_period], separators=(',', ':')),
                        args.session, session['periods']))
    db.close()
    view = get(f'/api/optimization/{args.session}')['session']
    batches = {b['id']: b['state'] for b in view['batches']}
    print(json.dumps({'run_state': run['state'] if already_terminal else 'failed', 'session_state': view['state'], 'proposal': view.get('proposal'),
                      'best': view.get('best'), 'periods': view.get('periods'), 'batch_states': batches,
                      'detail': view.get('detail')}, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
