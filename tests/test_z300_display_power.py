import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import z300_display_power as power


class Z300DisplayPowerTests(unittest.TestCase):
    def test_powershell_encoding_is_utf16le(self):
        script = "$x='Z300'\n"
        encoded = power._encoded_powershell(script)
        self.assertEqual(base64.b64decode(encoded).decode("utf-16le"), script)

    def test_read_state_parses_android_42_power_dump(self):
        responses = iter(
            [
                "device",
                "[ro.build.version.release]: [4.2.2]\n"
                "[ro.product.model]: [Sciaps LIBZ 100]\n"
                "[ro.build.fingerprint]: [generic/full_libz100/test]",
                "600000",
                "3",
                "  mWakefulness=Awake\n  mIsPowered=false\n  mPlugType=0\n"
                "  mBatteryLevel=100\nScreen off timeout: 600000 ms\n  mScreenOn=true\n",
                "mtp,adb",
            ]
        )
        with mock.patch.object(power, "_adb", side_effect=lambda *_: next(responses)):
            state = power.read_state()
        self.assertTrue(state.screen_on)
        self.assertFalse(state.powered)
        self.assertEqual(state.screen_off_timeout_ms, 600000)
        self.assertEqual(state.effective_screen_off_timeout_ms, 600000)
        self.assertEqual(state.android_release, "4.2.2")

    def test_dry_run_does_not_write_or_mutate(self):
        state = power.PowerState(
            serial=power.SERIAL,
            adb_state="device",
            android_release="4.2.2",
            model="Sciaps LIBZ 100",
            fingerprint="generic/full_libz100/test",
            screen_off_timeout_ms=600000,
            effective_screen_off_timeout_ms=600000,
            stay_on_while_plugged_in=3,
            wakefulness="Awake",
            screen_on=True,
            powered=False,
            plug_type=0,
            battery_level=100,
            usb_config="mtp,adb",
        )
        with tempfile.TemporaryDirectory() as directory:
            backup = Path(directory) / "backup.json"
            with mock.patch.object(power, "read_state", return_value=state), mock.patch.object(
                power, "_put_timeout"
            ) as put:
                result = power.apply_timeout(backup, dry_run=True)
            self.assertFalse(backup.exists())
            put.assert_not_called()
            self.assertEqual(result["target_timeout_ms"], power.MAX_TIMEOUT_MS)

    def test_wake_is_noop_if_screen_is_already_on(self):
        state = power.PowerState(
            serial=power.SERIAL,
            adb_state="device",
            android_release="4.2.2",
            model="Sciaps LIBZ 100",
            fingerprint="generic/full_libz100/test",
            screen_off_timeout_ms=power.MAX_TIMEOUT_MS,
            effective_screen_off_timeout_ms=power.MAX_TIMEOUT_MS,
            stay_on_while_plugged_in=3,
            wakefulness="Awake",
            screen_on=True,
            powered=False,
            plug_type=0,
            battery_level=95,
            usb_config="mtp,adb",
        )
        with mock.patch.object(power, "read_state", return_value=state), mock.patch.object(
            power,
            "_adb",
            side_effect=["123.50 99.00", "Result: Parcel(00000000    '....')"],
        ) as adb, mock.patch.object(power.time, "sleep"):
            result = power.wake_display(dry_run=False)
        self.assertEqual(
            adb.call_args_list[-1],
            mock.call(
                [
                    "shell",
                    "service",
                    "call",
                    "power",
                    "6",
                    "i32",
                    "123500",
                    "i32",
                    "0",
                ]
            ),
        )
        self.assertFalse(result["needed"])

    def test_backup_preserves_original_and_refuses_overwrite(self):
        state = power.PowerState(
            serial=power.SERIAL,
            adb_state="device",
            android_release="4.2.2",
            model="Sciaps LIBZ 100",
            fingerprint="generic/full_libz100/test",
            screen_off_timeout_ms=600000,
            effective_screen_off_timeout_ms=600000,
            stay_on_while_plugged_in=3,
            wakefulness="Awake",
            screen_on=True,
            powered=False,
            plug_type=0,
            battery_level=100,
            usb_config="mtp,adb",
        )
        with tempfile.TemporaryDirectory() as directory:
            backup = Path(directory) / "backup.json"
            power._write_backup(backup, state)
            payload = json.loads(backup.read_text(encoding="utf-8"))
            self.assertEqual(payload["original"]["screen_off_timeout_ms"], 600000)
            self.assertFalse(payload["original"]["powered"])
            with self.assertRaises(power.Z300PowerError):
                power._write_backup(backup, state)


if __name__ == "__main__":
    unittest.main()
