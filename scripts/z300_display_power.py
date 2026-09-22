#!/usr/bin/env python3
"""Inspect, extend, wake, or restore the Z300 display timeout over Opal USB ADB.

The analyzer is reached through the existing authenticated SSH connection to Opal
and Opal's existing, isolated ADB server.  No listener or service is installed.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


OPAL = "opal"
ADB = r"C:\Users\Whittaker\AppData\Local\Temp\alibz-api-recovery-20260921\platform-tools\adb.exe"
ADB_PORT = "5038"
SERIAL = "0123456789ABCDEF"
MAX_TIMEOUT_MS = 2_147_483_647
DEFAULT_BACKUP = Path("provenance/z300-display-power-backup-20260922.json")


class Z300PowerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PowerState:
    serial: str
    adb_state: str
    android_release: str
    model: str
    fingerprint: str
    screen_off_timeout_ms: int
    effective_screen_off_timeout_ms: int
    stay_on_while_plugged_in: int
    wakefulness: str
    screen_on: bool
    powered: bool
    plug_type: int
    battery_level: int
    usb_config: str


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _encoded_powershell(script: str) -> str:
    return base64.b64encode(script.encode("utf-16le")).decode("ascii")


def _run_powershell(script: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        OPAL,
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        _encoded_powershell(script),
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise Z300PowerError(
            f"Opal command failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result


def _adb(args: Sequence[str], timeout: int = 30) -> str:
    ps_args = ",".join(_ps_quote(value) for value in args)
    script = (
        "$ErrorActionPreference='Stop'\n"
        f"$adb={_ps_quote(ADB)}\n"
        f"$arguments=@('-P',{_ps_quote(ADB_PORT)},'-s',{_ps_quote(SERIAL)},{ps_args})\n"
        "& $adb @arguments\n"
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
    )
    return _run_powershell(script, timeout=timeout).stdout.replace("\r", "").strip()


def _require_match(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise Z300PowerError(f"Could not parse {label} from dumpsys power")
    return match.group(1)


def _integer_setting(namespace: str, name: str) -> int:
    raw = _adb(["shell", "settings", "get", namespace, name])
    try:
        return int(raw)
    except ValueError as exc:
        raise Z300PowerError(f"Unexpected {namespace}.{name} value: {raw!r}") from exc


def read_state() -> PowerState:
    device_state = _adb(["get-state"])
    if device_state != "device":
        raise Z300PowerError(f"ADB serial {SERIAL} is not authorized: {device_state!r}")

    properties = _adb(["shell", "getprop"])
    props = dict(re.findall(r"^\[([^]]+)\]: \[(.*)\]$", properties, re.MULTILINE))
    expected = {
        "ro.build.version.release": "4.2.2",
        "ro.product.model": "Sciaps LIBZ 100",
    }
    for name, value in expected.items():
        if props.get(name) != value:
            raise Z300PowerError(
                f"Refusing unexpected analyzer identity {name}={props.get(name)!r}"
            )

    configured = _integer_setting("system", "screen_off_timeout")
    stay_on = _integer_setting("global", "stay_on_while_plugged_in")
    power = _adb(["shell", "dumpsys", "power"])
    usb_config = _adb(["shell", "getprop", "sys.usb.config"])
    return PowerState(
        serial=SERIAL,
        adb_state=device_state,
        android_release=props["ro.build.version.release"],
        model=props["ro.product.model"],
        fingerprint=props.get("ro.build.fingerprint", ""),
        screen_off_timeout_ms=configured,
        effective_screen_off_timeout_ms=int(
            _require_match(r"^Screen off timeout: (\d+) ms$", power, "effective timeout")
        ),
        stay_on_while_plugged_in=stay_on,
        wakefulness=_require_match(r"^\s*mWakefulness=(\w+)$", power, "wakefulness"),
        screen_on=_require_match(r"^\s*mScreenOn=(true|false)$", power, "screen state")
        == "true",
        powered=_require_match(r"^\s*mIsPowered=(true|false)$", power, "power source")
        == "true",
        plug_type=int(_require_match(r"^\s*mPlugType=(\d+)$", power, "plug type")),
        battery_level=int(
            _require_match(r"^\s*mBatteryLevel=(\d+)$", power, "battery level")
        ),
        usb_config=usb_config,
    )


def _write_backup(path: Path, state: PowerState) -> None:
    if path.exists():
        raise Z300PowerError(f"Refusing to overwrite existing backup: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": OPAL,
        "setting": "system.screen_off_timeout",
        "original": asdict(state),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _put_timeout(value: int) -> PowerState:
    if not 10_000 <= value <= MAX_TIMEOUT_MS:
        raise Z300PowerError(f"Refusing invalid timeout: {value}")
    _adb(["shell", "settings", "put", "system", "screen_off_timeout", str(value)])
    state = read_state()
    if state.screen_off_timeout_ms != value or state.effective_screen_off_timeout_ms != value:
        raise Z300PowerError(
            "Timeout verification failed: "
            f"stored={state.screen_off_timeout_ms}, effective={state.effective_screen_off_timeout_ms}"
        )
    return state


def apply_timeout(backup: Path, dry_run: bool) -> dict[str, object]:
    before = read_state()
    result: dict[str, object] = {
        "action": "apply",
        "dry_run": dry_run,
        "target_timeout_ms": MAX_TIMEOUT_MS,
        "backup": str(backup),
        "before": asdict(before),
    }
    if dry_run:
        return result
    _write_backup(backup, before)
    result["after"] = asdict(_put_timeout(MAX_TIMEOUT_MS))
    return result


def restore_timeout(backup: Path, dry_run: bool) -> dict[str, object]:
    payload = json.loads(backup.read_text(encoding="utf-8"))
    original = payload.get("original", {})
    if payload.get("setting") != "system.screen_off_timeout" or original.get("serial") != SERIAL:
        raise Z300PowerError("Backup does not describe this Z300 timeout")
    value = int(original["screen_off_timeout_ms"])
    before = read_state()
    result: dict[str, object] = {
        "action": "restore",
        "dry_run": dry_run,
        "restore_timeout_ms": value,
        "backup": str(backup),
        "before": asdict(before),
    }
    if not dry_run:
        result["after"] = asdict(_put_timeout(value))
    return result


def wake_display(dry_run: bool) -> dict[str, object]:
    before = read_state()
    result: dict[str, object] = {
        "action": "wake",
        "dry_run": dry_run,
        "needed": not before.screen_on,
        "before": asdict(before),
    }
    if dry_run:
        return result

    # Android 4.2.2 IPowerManager.aidl declares wakeUp(long) as transaction 6.
    # Unlike KEYCODE_POWER, wakeUp is explicitly a no-op when already awake, so
    # a concurrent operator wake cannot turn the display back off.  Use a fresh
    # uptime for each attempt because PowerManager rejects events older than the
    # most recent sleep transition.
    calls: list[str] = []
    after = before
    for _ in range(2):
        uptime = _adb(["shell", "cat", "/proc/uptime"]).split()[0]
        event_time_ms = int(float(uptime) * 1000)
        low_unsigned = event_time_ms & 0xFFFFFFFF
        high_unsigned = (event_time_ms >> 32) & 0xFFFFFFFF
        low = low_unsigned if low_unsigned < 0x80000000 else low_unsigned - 0x100000000
        high = high_unsigned if high_unsigned < 0x80000000 else high_unsigned - 0x100000000
        # This Android 4.2 service CLI has only i32 and s16 arguments.  AIDL's
        # long is packed into the Parcel as low then high 32-bit words.
        reply = _adb(
            [
                "shell",
                "service",
                "call",
                "power",
                "6",
                "i32",
                str(low),
                "i32",
                str(high),
            ]
        )
        calls.append(reply)
        if not re.fullmatch(r"Result:\s*Parcel\(00000000\s+'\.\.\.\.'\)", reply):
            raise Z300PowerError(f"PowerManager wakeUp binder call failed: {reply}")
        time.sleep(0.5)
        after = read_state()
        if after.screen_on:
            break
    result["binder_calls"] = calls
    result["after"] = asdict(after)
    if not after.screen_on:
        raise Z300PowerError("PowerManager wakeUp completed but the display remains off")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "apply", "wake", "restore"))
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "status":
            result: object = asdict(read_state())
        elif args.action == "apply":
            result = apply_timeout(args.backup, args.dry_run)
        elif args.action == "restore":
            result = restore_timeout(args.backup, args.dry_run)
        else:
            result = wake_display(args.dry_run)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, Z300PowerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
