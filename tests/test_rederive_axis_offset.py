"""Pure-logic tests for scripts/rederive-alibz-axis-offset.py.

The tool imports pantheum only inside main(); its classification, refusal, and
idempotence helpers and native_xy dedup are pure and covered here with synthetic
fixtures. The module filename is hyphenated (deploy-* convention), so it is loaded
by path rather than imported as scripts.<name>.
"""
import importlib.util
import unittest
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'rederive-alibz-axis-offset.py'
_spec = importlib.util.spec_from_file_location('rederive_axis_offset', _MOD_PATH)
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


def facts(**over):
    base = dict(state='succeeded', test_id='t-1', has_dir=True,
                has_raw_average=True, has_raw_shots=True, has_all_zip=False,
                has_test_json=True, test_axis_check=None,
                has_opal_manifest=False, opal_pixel_offset=None)
    base.update(over)
    return base


class ClassifyTests(unittest.TestCase):
    def test_a_api_native_offset0_rederives(self):
        v = rd.classify(facts())
        self.assertEqual(v['action'], 'rederive')
        self.assertEqual(v['class'], 'a_api_native')

    def test_b_opal_legacy_offset0_rederives(self):
        v = rd.classify(facts(has_opal_manifest=True, opal_pixel_offset=0))
        self.assertEqual(v['action'], 'rederive')
        self.assertEqual(v['class'], 'b_opal_legacy')

    def test_b_opal_legacy_offset_none_rederives(self):
        v = rd.classify(facts(has_opal_manifest=True, opal_pixel_offset=None))
        self.assertEqual((v['action'], v['class']), ('rederive', 'b_opal_legacy'))

    def test_e_opal_flatbuffers_already_18_is_skipped(self):
        v = rd.classify(facts(has_opal_manifest=True, opal_pixel_offset=-18))
        self.assertEqual(v['action'], 'skip')
        self.assertEqual(v['class'], 'e_already_correct')

    def test_e_axis_check_present_and_aligned_is_skipped(self):
        v = rd.classify(facts(test_axis_check={'offset_px': 0.25}))
        self.assertEqual((v['action'], v['class']), ('skip', 'e_already_correct'))

    def test_axis_check_present_but_offset_large_still_rederives(self):
        v = rd.classify(facts(test_axis_check={'offset_px': -18.0}))
        self.assertEqual(v['action'], 'rederive')

    def test_axis_check_none_offset_does_not_skip(self):
        v = rd.classify(facts(test_axis_check={'offset_px': None}))
        self.assertEqual(v['action'], 'rederive')

    def test_d_no_raw_pixels_report_only_refetchable(self):
        v = rd.classify(facts(has_raw_average=False, has_raw_shots=False, has_all_zip=False))
        self.assertEqual(v['action'], 'report_only')
        self.assertEqual(v['class'], 'd_no_raw')
        self.assertTrue(v['refetchable'])

    def test_d_no_raw_pixels_no_test_id_not_refetchable(self):
        v = rd.classify(facts(has_raw_average=False, has_raw_shots=False,
                              has_all_zip=False, test_id=None))
        self.assertEqual(v['action'], 'report_only')
        self.assertFalse(v['refetchable'])

    def test_all_zip_only_still_rederivable(self):
        v = rd.classify(facts(has_raw_average=False, has_raw_shots=False, has_all_zip=True,
                              has_opal_manifest=True, opal_pixel_offset=0))
        self.assertEqual(v['action'], 'rederive')
        self.assertEqual(v['source'], 'all_zip')


class RefusalTests(unittest.TestCase):
    def test_no_active_work_allows(self):
        self.assertEqual(rd.active_work_reasons(0, 0, 0), [])

    def test_active_acquisition_refuses(self):
        self.assertEqual(len(rd.active_work_reasons(1, 0, 0)), 1)

    def test_all_three_refuse(self):
        r = rd.active_work_reasons(2, 3, 1)
        self.assertEqual(len(r), 3)
        self.assertTrue(any('acquisition' in x for x in r))
        self.assertTrue(any('job' in x for x in r))
        self.assertTrue(any('hardware' in x for x in r))


class IdempotenceTests(unittest.TestCase):
    def test_fresh_run_is_not_already_rederived(self):
        self.assertFalse(rd.already_rederived({'acquisition': {}}, None))
        self.assertFalse(rd.already_rederived(None, {'offset_px': -18.0}))

    def test_metadata_link_marks_done(self):
        self.assertTrue(rd.already_rederived({'rederived_from': ['old']}, None))

    def test_metadata_link_nested_in_acquisition_marks_done(self):
        self.assertTrue(rd.already_rederived({'acquisition': {'rederived_from': ['old']}}, None))

    def test_aligned_axis_check_marks_done(self):
        self.assertTrue(rd.already_rederived(None, {'offset_px': 0.25}))

    def test_large_axis_check_is_not_done(self):
        self.assertFalse(rd.already_rederived({}, {'offset_px': -18.0}))

    def test_second_pass_is_a_noop(self):
        """A run re-derived on pass 1 (metadata link written) is skipped on pass 2."""
        after_pass1_meta = {'rederived_from': ['old-1'], 'axis_check': {'offset_px': 0.25}}
        after_pass1_axis = {'offset_px': 0.25}
        self.assertTrue(rd.already_rederived(after_pass1_meta, after_pass1_axis))


class _StubModule:
    """Minimal stand-in for z300_calibration used by native_xy."""
    def __init__(self, wl, inten):
        self._wl, self._inten = wl, inten

    def pixels_to_wavelength(self, calibration, knots, pixels, pixel_offset):
        # offset shifts the axis by 1 nm/px so the two offsets differ measurably.
        return [w + pixel_offset for w in self._wl], list(self._inten)


class NativeXYTests(unittest.TestCase):
    def test_strictly_increasing_dedup(self):
        mod = _StubModule([1.0, 1.0, 2.0, 2.0, 3.0], [10, 11, 20, 21, 30])
        xs, ys = rd.native_xy(mod, {'wlCalibrations': None, 'knots': None, 'pixels': None}, 0)
        self.assertEqual(xs, [1.0, 2.0, 3.0])
        self.assertEqual(ys, [10, 20, 30])

    def test_offset_shifts_axis(self):
        mod = _StubModule([620.0, 700.0, 800.0], [1, 2, 3])
        xs0, _ = rd.native_xy(mod, {'wlCalibrations': 0, 'knots': 0, 'pixels': 0}, 0)
        xs18, _ = rd.native_xy(mod, {'wlCalibrations': 0, 'knots': 0, 'pixels': 0}, -18)
        self.assertEqual(xs18, [x - 18 for x in xs0])


if __name__ == '__main__':
    unittest.main()
