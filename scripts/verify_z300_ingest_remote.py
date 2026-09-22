#!/usr/bin/env python3
"""On Moissanite, verify a real native archive in a separate temporary queue.

Production state is read-only. The temporary queue never dispatches hardware;
this exercises the configured SSH/Opal transfer, atomic completion and preview.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = Path.home() / 'pantheum-I'
    state = Path.home() / '.local/state/pantheum/alibz'
    with sqlite3.connect((state / 'alibz.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT * FROM acquisitions WHERE id=?', (args.run_id,)).fetchone()
        if row is None or not row['test_id'] or row['state'] not in ('succeeded', 'awaiting_data'):
            raise RuntimeError('Validation requires an existing acknowledged completed run')
        run = dict(row)
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    sys.path.insert(0, str(root))
    import pantheum.alibz
    pantheum.alibz.__path__.insert(0, str(args.bundle / 'pantheum/alibz'))
    from pantheum.alibz.service import Service
    from pantheum.alibz.retrieval import OpalRetrieval
    from pantheum.alibz.z300 import Z300Client
    def forbidden(*args, **kwargs):
        raise AssertionError('Physical/API acquisition is forbidden in retrieval verification')
    Z300Client.fire_test = Z300Client.cancel = Z300Client.shot_spectrum = Z300Client.test_page = forbidden
    with tempfile.TemporaryDirectory(prefix='opal-ingest-verify-', dir=state.parent) as directory:
        acquire = dict(manifest['acquire_patch'], enabled=False, enabled_actions=[])
        service = Service({'data_dir': directory, 'auto_engines': ['preview'], 'acquire': acquire})
        with service.db() as db:
            db.execute('INSERT INTO acquisitions(id,state,mode,run_mode,params,request_sha256,test_id,'
                       'created_at,dataset_ids,shots,hardware_resolved,retrieval) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (run['id'], 'awaiting_data', run['mode'], 'live', run['params'], run['request_sha256'],
                        run['test_id'], run['created_at'], '[]', 0, 1, 'opal_database'))
        retrieval = OpalRetrieval(service)
        retrieval.recover()
        assert retrieval.run_one()
        completed = service.acquisition.get_run(run['id'])
        if completed['state'] != 'succeeded':
            raise RuntimeError(completed.get('detail', 'Retrieval did not complete'))
        with service.db() as db:
            assert db.execute('SELECT count(*) FROM datasets').fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0] == 1
        assert not retrieval.run_one(), 'Completed retrieval must be idempotent'
        assert service.run_one(), 'Native preview job did not execute'
        with service.db() as db:
            job_status = db.execute('SELECT status FROM jobs').fetchone()[0]
        assert job_status == 'completed', 'Downstream preview did not complete: ' + job_status
        spectrum = json.loads((service.root / 'spectra' / (completed['dataset_ids'][0] + '.json')).read_text())
        axis = spectrum['wavelength']
        pitch = [b-a for a,b in zip(axis,axis[1:])]
        assert len({round(p, 8) for p in pitch}) > 10, 'Native grid unexpectedly uniform'
        result = {'verified': True, 'isolated_validation_queue': True, 'production_state_modified': False,
                  'physical_commands': 0, 'run_id': run['id'], 'test_id': run['test_id'],
                  'shots': completed['shots'], 'native_points': len(axis),
                  'minimum_pitch_nm': min(pitch), 'maximum_pitch_nm': max(pitch),
                  'acquisition_state': completed['state'], 'downstream_preview': job_status,
                  'idempotent': True}
        print(json.dumps(result))


if __name__ == '__main__':
    main()
