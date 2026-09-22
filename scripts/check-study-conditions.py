#!/usr/bin/env python3
"""Pre-start check for a stepped acquisition (delay/period) study.

The optimizer refuses any live batch whose delay/period/pulsePeriod condition has
no fully stored live run in the acquisition ledger ("Unverified condition(s)
..."). This lists, for a grid, which conditions are verified, which are not, and
the exact Acquire-panel runs that would verify them, plus the analyzer flags
that block a fire. Read-only: it queries Pantheum's SQLite on Moissanite over
SSH and GETs /instrument/id through the proxy. Exit 0 = the study can run to
completion as far as these gates go; 1 = something would refuse.

  scripts/check-study-conditions.py                      default grid 5,10,20 x 10,25,50 @ pulsePeriod 100, 10 shots
  scripts/check-study-conditions.py --delays 5,10,20,90 --periods 10,25,50,90
  scripts/check-study-conditions.py --session opt-b1c216f4   check an existing session's grid
  scripts/check-study-conditions.py --json
"""
import argparse, json, subprocess, sys

REMOTE = 'moissanite'
PROXY = 'http://192.168.50.112:19000'
RASTER_SITES = 12   # [134..206] x [76..124] step 24 = 4 x 3 sites; batches wrap to a new lap after that

QUERY = r'''
import os, sqlite3, json, sys
db = sqlite3.connect(os.path.expanduser('~/.local/state/pantheum/alibz/alibz.sqlite')); db.row_factory = sqlite3.Row
out = {'runs': [dict(r) for r in db.execute(
    "SELECT id,state,created_at,shots, json_extract(params,'$.intergrationDelay') d, json_extract(params,'$.intergrationPeriod') p, "
    "json_extract(params,'$.pulsePeriod') pp, json_extract(params,'$.numShotsPerLocation')*json_extract(params,'$.numlocations') req "
    "FROM acquisitions WHERE run_mode='live' ORDER BY created_at")],
    'active': db.execute("SELECT COUNT(*) FROM acquisitions WHERE state IN ('queued','running','cancelling','awaiting_data')").fetchone()[0],
    'sessions': {r['id']: dict(r) for r in db.execute("SELECT id,state,delays,periods,params,study_type,plan FROM optimization_sessions")}}
try:
    cfg = json.load(open(os.path.expanduser('~/.config/pantheum/alibz.json')))
    out['require_verified'] = (cfg.get('optimization') or {}).get('require_verified_conditions', True)
    out['min_shots'] = (cfg.get('acquire') or {}).get('min_shots', 6)
except Exception as e:
    out['config_error'] = str(e)
print(json.dumps(out))
'''

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--delays', default='5,10,20'); ap.add_argument('--periods', default='10,25,50')
    ap.add_argument('--pulse-period', type=int, default=100); ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--session', help='session id (prefix ok): use its grid instead')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    data = json.loads(subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', REMOTE, 'python3', '-'],
                                     input=QUERY, capture_output=True, text=True, check=True).stdout)
    delays = [int(x) for x in a.delays.split(',')]; periods = [int(x) for x in a.periods.split(',')]
    pp, shots = a.pulse_period, a.shots
    if a.session:
        match = [s for s in data['sessions'] if s.startswith(a.session)]
        if len(match) != 1: sys.exit(f'session {a.session}: {len(match)} matches')
        s = data['sessions'][match[0]]; params = json.loads(s['params'])
        delays, periods, pp = json.loads(s['delays']), json.loads(s['periods']), params['pulsePeriod']
        shots = 10 if (s['study_type'] or 'delay_period') == 'delay_period' else params['numShotsPerLocation']
    grid = [(d, p) for d in delays for p in periods]
    best = {}   # condition -> best evidence
    for r in data['runs']:
        key = (r['d'], r['p'], r['pp'])
        if r['state'] == 'succeeded' and r['shots'] and r['shots'] > 0:
            e = best.setdefault(key, {'full': 0, 'partial': 0, 'runs': 0})
            e['runs'] += 1
            if r['shots'] == r['req']: e['full'] = max(e['full'], r['shots'])
            else: e['partial'] = max(e['partial'], r['shots'])
    rows = []
    for d, p in grid:
        e = best.get((d, p, pp), {'full': 0, 'partial': 0, 'runs': 0})
        verified = e['full'] >= shots
        rows.append({'delay': d, 'period': p, 'pulsePeriod': pp, 'verified': verified, 'full_shots': e['full'],
                     'best_partial_shots': e['partial'], 'succeeded_runs': e['runs']})
    unverified = [r for r in rows if not r['verified']]
    try:
        ident = json.loads(subprocess.run(['ssh', '-o', 'BatchMode=yes', REMOTE, 'curl', '-s', '-m', '5', PROXY + '/instrument/id'],
                                          capture_output=True, text=True, check=True).stdout)
    except Exception as exc:
        ident = {'error': str(exc)}
    blockers = []
    if data.get('require_verified', True) and unverified:
        blockers.append(f"{len(unverified)} unverified condition(s): " + ', '.join(f"{r['delay']}/{r['period']}" for r in unverified))
    if data.get('active'): blockers.append(f"{data['active']} active/awaiting acquisition(s)")
    if ident.get('triggerLocked'): blockers.append('trigger locked (padlock on the handheld)')
    if 'error' in ident: blockers.append('analyzer API unreachable: ' + ident['error'])
    notes = []
    if ident.get('wlCalibrationNeededCode') == 0: notes.append('analyzer wants a wavelength calibration (advisory)')
    if len(grid) > RASTER_SITES: notes.append(f'{len(grid)} conditions > {RASTER_SITES} raster sites: later batches re-use sites on a second lap')
    notes.append('laser arming has no API: arm on the handheld (PIN) and keep Geochem Pro on START')
    if pp != 100: notes.append(f'pulsePeriod {pp} is not the verified 10 Hz encoding (100); 10 (100 Hz) is refused by the instrument')
    result = {'grid': rows, 'unverified': [(r['delay'], r['period']) for r in unverified], 'blockers': blockers, 'notes': notes,
              'require_verified_conditions': data.get('require_verified', True), 'min_shots': data.get('min_shots'),
              'analyzer': {k: ident.get(k) for k in ('id', 'triggerLocked', 'wlCalibrationNeededCode', 'argonPSILevel')}}
    if a.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"grid {delays} x {periods} @ pulsePeriod {pp}, {shots} shots per batch; require_verified_conditions={result['require_verified_conditions']}")
        print(f"{'delay/period':>13}  {'verified':>8}  {'full-run shots':>14}  {'best partial':>12}  {'runs':>4}")
        for r in rows:
            print(f"{r['delay']:>6}/{r['period']:<6}  {'yes' if r['verified'] else 'NO':>8}  {r['full_shots'] or '-':>14}  {r['best_partial_shots'] or '-':>12}  {r['succeeded_runs']:>4}")
        if unverified:
            print('\nTo verify, run one live Acquire-panel batch (10 shots, 1 location, pulsePeriod %d, fresh spot) at:' % pp)
            for r in unverified: print(f"  delay {r['delay']} / period {r['period']}")
        print('\nanalyzer:', result['analyzer'])
        for b in blockers: print('BLOCKER:', b)
        for n in notes: print('note:', n)
        print('\nRESULT:', 'study can start' if not blockers else 'study would be refused')
    sys.exit(1 if blockers else 0)

if __name__ == '__main__':
    main()
