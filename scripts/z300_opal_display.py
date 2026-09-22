#!/usr/bin/env python3
"""Bounded display-only control for the Z300, executed locally on Opal.

Pantheum owns checkout and acquisition guards. This helper accepts only explicit
status/sleep/wake actions and never toggles the power key or changes the timeout.
"""
import argparse
from datetime import datetime, timezone
import json
import re
import subprocess
import sys
import time

ADB = r"C:\Users\Whittaker\AppData\Local\Temp\alibz-api-recovery-20260921\platform-tools\adb.exe"
SERIAL = "0123456789ABCDEF"
FINGERPRINT = "generic/full_libz100/libz100:4.2.2/JDQ39/217:eng/test-keys"
STATE_COMMAND = (
    "getprop ro.build.version.release; getprop ro.product.model; "
    "getprop ro.build.fingerprint; dumpsys power"
)


class DisplayError(RuntimeError):
    pass


def adb(*args):
    result = subprocess.run(
        [ADB, "-P", "5038", "-s", SERIAL, *args],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=5, check=False, text=True,
    )
    if result.returncode or len(result.stdout) > 65536:
        raise DisplayError("Analyzer USB connection unavailable")
    return result.stdout.replace("\r", "").strip()


def read_state():
    raw = adb("shell", STATE_COMMAND)
    # Legacy ADB emits CR-CR-LF on this firmware; universal newline decoding
    # can therefore leave an empty line between output lines.
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines[:3] != ["4.2.2", "Sciaps LIBZ 100", FINGERPRINT]:
        raise DisplayError("Analyzer firmware does not match the verified display interface")
    screen = re.search(r"^\s*mScreenOn=(true|false)\s*$", raw, re.MULTILINE)
    wake = re.search(r"^\s*mWakefulness=(\w+)\s*$", raw, re.MULTILINE)
    if not screen or not wake:
        raise DisplayError("Analyzer display state unavailable")
    return {
        "screen_on": screen.group(1) == "true",
        "wakefulness": wake.group(1),
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


def binder_args(action, uptime):
    """Android 4.2.2 IPowerManager long is two signed little-endian i32 words."""
    if action not in ("sleep", "wake"):
        raise DisplayError("Unsupported display action")
    # Parse the fixed /proc/uptime decimal without floating-point rounding.
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?", uptime)
    if not match:
        raise DisplayError("Analyzer clock unavailable")
    millis = int(match.group(1)) * 1000 + int(((match.group(2) or "") + "000")[:3])
    if not 0 <= millis < 2**63:
        raise DisplayError("Analyzer clock outside supported range")
    words = [(millis >> shift) & 0xFFFFFFFF for shift in (0, 32)]
    words = [word if word < 0x80000000 else word - 0x100000000 for word in words]
    args = ["shell", "service", "call", "power", "6" if action == "wake" else "7",
            "i32", str(words[0]), "i32", str(words[1])]
    if action == "sleep":
        args += ["i32", "0"]  # Android 4.2.2 GO_TO_SLEEP_REASON_USER
    return args


def control(action, dry_run=False):
    if action not in ("status", "sleep", "wake"):
        raise DisplayError("Unsupported display action")
    before = read_state()
    if action == "status" or dry_run:
        return before
    wanted = action == "wake"
    wanted_wakefulness = "Awake" if wanted else "Asleep"
    if before["screen_on"] == wanted and before["wakefulness"] == wanted_wakefulness:
        return before
    uptime = adb("shell", "cat", "/proc/uptime").split()
    if not uptime:
        raise DisplayError("Analyzer clock unavailable")
    reply = adb(*binder_args(action, uptime[0]))
    if not re.fullmatch(r"Result:\s*Parcel\(00000000\s+'\.\.\.\.'\)", reply):
        raise DisplayError("Analyzer rejected the display command")
    # Observe, never replay: a timeout or operator action may make the result
    # ambiguous. Only a fresh matching state is reported as successful.
    for _ in range(4):
        time.sleep(0.2)
        state = read_state()
        if state["screen_on"] == wanted and state["wakefulness"] == wanted_wakefulness:
            return state
    raise DisplayError("Display command sent, but the requested state was not confirmed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "sleep", "wake"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(control(args.action, args.dry_run), separators=(",", ":")))
        return 0
    except (DisplayError, OSError, subprocess.SubprocessError) as exc:
        # Do not expose local paths, command output, or USB identifiers to HTTP.
        message = str(exc) if isinstance(exc, DisplayError) else "Analyzer display connection failed"
        print(json.dumps({"error": message}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
