import unittest
from scripts.validate_z300_opal import command


class ValidationPlanTests(unittest.TestCase):
    def test_default_is_nonwriting_dry_run(self):
        plan = command('run-example', 'test-example', 10)
        self.assertIn('--dry-run', plan)
        self.assertNotIn('Get-Content', plan)

    def test_apply_keeps_payload_on_opal(self):
        plan = command('run-example', 'test-example', 10, True)
        self.assertIn('--output', plan)
        self.assertNotIn('--dry-run', plan)
        self.assertIn('native_points=', plan)

    def test_untrusted_ids_rejected_before_remote_execution(self):
        for value in ("run'; echo bad", '../run', 'run;bad', '$env:TOKEN', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                command(value, 'test-example', 10)

    def test_unsupported_count_rejected(self):
        for value in (True, 0, -1, 1001, '10'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                command('run-example', 'test-example', value)
