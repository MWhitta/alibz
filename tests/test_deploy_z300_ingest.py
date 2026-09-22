import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import deploy_z300_ingest as deploy


class DeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.bundle, self.state = (base / x for x in ('source', 'bundle', 'state'))
        for p in (self.root, self.bundle, self.state):
            p.mkdir()
        self.config = base / 'config.json'
        self.config.write_text(json.dumps({'acquire': {'retrieval': 'deferred', 'enabled_actions': ['z300.firetest']}, 'other': 23}))
        self.original_config = self.config.read_bytes()
        (self.root / 'old.py').write_text('before')
        (self.bundle / 'old.py').write_text('after')
        (self.bundle / 'new.py').write_text('new')
        self.manifest = {'files': {n: deploy.digest(self.bundle / n) for n in ('old.py', 'new.py')},
                         'before': {'old.py': deploy.digest(self.root / 'old.py'), 'new.py': None},
                         'config_sha256': deploy.digest(self.config),
                         'acquire_patch': {'retrieval': 'opal_database', 'opal_retrieval': {'command': ['configured-helper']}}}
        (self.bundle / 'manifest.json').write_text(json.dumps(self.manifest))
        with sqlite3.connect(self.state / 'alibz.sqlite') as db:
            db.executescript('CREATE TABLE acquisitions(state TEXT); CREATE TABLE jobs(status TEXT); CREATE TABLE hardware_operations(state TEXT);')

    def call(self, apply=False, run=None):
        return deploy.install(self.bundle, self.root, self.config, self.state, apply, run or Mock())

    def test_dry_run_has_no_mutations_or_service_calls(self):
        run = Mock()
        self.assertFalse(self.call(run=run)['apply'])
        run.assert_not_called()
        self.assertEqual((self.root / 'old.py').read_text(), 'before')
        self.assertFalse((self.root / 'new.py').exists())
        self.assertEqual(self.config.read_bytes(), self.original_config)

    def test_rejects_concurrent_source_and_config_changes(self):
        for target in (self.root / 'old.py', self.config):
            original = target.read_bytes()
            target.write_bytes(original + b' ')
            with self.assertRaises(ValueError):
                self.call(True)
            target.write_bytes(original)

    def test_active_queue_refuses_before_stop(self):
        run = Mock()
        with sqlite3.connect(self.state / 'alibz.sqlite') as db:
            db.execute("INSERT INTO acquisitions VALUES ('running')")
        with self.assertRaises(RuntimeError):
            self.call(True, run)
        run.assert_not_called()

    def test_apply_preserves_other_configuration_and_backups(self):
        run = Mock()
        result = self.call(True, run)
        self.assertEqual(run.call_count, 2)
        cfg = json.loads(self.config.read_text())
        self.assertEqual(cfg['other'], 23)
        self.assertEqual(cfg['acquire']['enabled_actions'], ['z300.firetest'])
        self.assertEqual(cfg['acquire']['retrieval'], 'opal_database')
        self.assertEqual((Path(result['backup']) / 'alibz.json').read_bytes(), self.original_config)
        self.assertEqual(deploy.digest(self.root / 'new.py'), self.manifest['files']['new.py'])

    def test_failed_write_restores_source_and_config(self):
        real_copy = deploy.atomic_copy
        def fail_once(source, target):
            if source == self.bundle / 'new.py':
                raise OSError('simulated write failure')
            real_copy(source, target)
        run = Mock()
        with patch.object(deploy, 'atomic_copy', side_effect=fail_once):
            with self.assertRaises(OSError):
                self.call(True, run)
        self.assertEqual(run.call_count, 2)
        self.assertEqual((self.root / 'old.py').read_text(), 'before')
        self.assertFalse((self.root / 'new.py').exists())
        self.assertEqual(self.config.read_bytes(), self.original_config)


if __name__ == '__main__':
    unittest.main()
