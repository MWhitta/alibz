import unittest
from unittest.mock import patch

from scripts import z300_opal_display as display


def state(on):
    return {"screen_on": on, "wakefulness": "Awake" if on else "Asleep",
            "observed_at": "2026-09-22T18:00:00+00:00"}


class DisplayTests(unittest.TestCase):
    def test_legacy_adb_blank_lines_are_accepted(self):
        raw = "\n\n".join(["4.2.2", "Sciaps LIBZ 100", display.FINGERPRINT,
                             "  mScreenOn=true", "  mWakefulness=Awake"])
        with patch.object(display, "adb", return_value=raw):
            result = display.read_state()
        self.assertTrue(result["screen_on"])
        self.assertEqual(result["wakefulness"], "Awake")

    def test_exact_firmware_is_required_before_mutation(self):
        with patch.object(display, "adb", return_value="4.2.2\nAnother model\nunknown") as adb:
            with self.assertRaises(display.DisplayError):
                display.control("sleep")
        self.assertEqual(adb.call_count, 1)

    def test_signed_long_encoding_and_sleep_reason(self):
        args = display.binder_args("sleep", "4294967.295")
        self.assertEqual(args, ["shell", "service", "call", "power", "7",
                                "i32", "-1", "i32", "0", "i32", "0"])
        self.assertEqual(display.binder_args("wake", "4294967.296")[-4:],
                         ["i32", "0", "i32", "1"])

    def test_dry_run_does_not_dispatch(self):
        with patch.object(display, "read_state", return_value=state(True)), patch.object(display, "adb") as adb:
            self.assertEqual(display.control("sleep", dry_run=True), state(True))
        adb.assert_not_called()

    def test_already_requested_state_is_noop(self):
        for action, on in (("wake", True), ("sleep", False)):
            with self.subTest(action=action), patch.object(display, "read_state", return_value=state(on)), patch.object(display, "adb") as adb:
                self.assertEqual(display.control(action), state(on))
                adb.assert_not_called()

    def test_wake_and_sleep_verify_transition(self):
        for action, on in (("wake", True), ("sleep", False)):
            with self.subTest(action=action), patch.object(display, "read_state", side_effect=[state(not on), state(on)]), patch.object(display, "adb", side_effect=["123.50 0.00", "Result: Parcel(00000000    '....')"]) as adb, patch.object(display.time, "sleep"):
                self.assertEqual(display.control(action), state(on))
                self.assertEqual(adb.call_count, 2)
                self.assertNotIn("keyevent", repr(adb.call_args_list))

    def test_ambiguous_command_is_not_replayed(self):
        with patch.object(display, "read_state", return_value=state(True)), patch.object(display, "adb", side_effect=["123.50 0.00", "Result: Parcel(00000000    '....')"]) as adb, patch.object(display.time, "sleep"):
            with self.assertRaisesRegex(display.DisplayError, "not confirmed"):
                display.control("sleep")
            self.assertEqual(adb.call_count, 2)

    def test_binder_failure_does_not_claim_success(self):
        with patch.object(display, "read_state", return_value=state(False)), patch.object(display, "adb", side_effect=["123.50 0.00", "service: unknown option"]):
            with self.assertRaisesRegex(display.DisplayError, "rejected"):
                display.control("wake")


if __name__ == "__main__":
    unittest.main()
