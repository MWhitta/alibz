# Z300 recovery after reboot — 2026-09-21

User rebooted the Z300 and asked to retry. At 21:53:20Z, deployed alibz reported
`live_allowed: true`, no blocking reasons, analyzer Z300-0915 reachable with
trigger unlocked, and gantry connected with a fresh Idle observation.

Restored mwhittaker@lbl.gov's expired hardware checkout using the existing
reservation API; lease expires at 22:07:20Z unless renewed. The gantry observer
had stopped for lack of a viewer; GET motion/status?touch=1 resumed read-only
observation. No network/configuration/service change was necessary.

The existing live session opt-78c196a5e31d4f8ea914c1344ded0c9a is ready for
sample Fe_Aesar_99.98%_18823 at delay 10 / period 25, rate 100 Hz, zero shots.
No acquisition, movement, firing, cancellation, or settings push was issued.
The operator can now use the session's batch control and bench confirmation.

Evidence: reports/2026-09-21-after-reboot-readiness.json. Reusable read-only
check: scripts/check-optimization-readiness.py (dry-run and live execution both
passed; live exit code 0). No code deployment or provider switch.
