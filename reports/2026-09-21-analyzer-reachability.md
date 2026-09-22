# Z300 reachability diagnosis — 2026-09-21

## Outcome

The live fault is isolated to the Z300 API listener on TCP `9000`, not to the
Option D bridge, Wi-Fi link, stale IP address, or an offline analyzer. At
2026-09-21 14:33 PDT, xanthiosite's Wi-Fi was connected to `lml_automation` at
`192.168.60.207/24`; the bridge configuration and services were healthy. The
Z300 at `192.168.60.65` answered 2/2 ICMP packets (4–6 ms) and refreshed the
neighbor cache to its expected MAC `40-06-a0-a0-6f-64`, but a direct TCP
connection from xanthiosite to `192.168.60.65:9000` was **actively refused**.

This evidence distinguishes an API/service problem from an address change:
the expected Z300 MAC is currently responding at the expected `.65` address.
The user's contemporaneous statement that the analyzer is powered on, awake,
and connected to `lml_automation` is consistent with the network evidence.

## Evidence

1. The applicable runbook is
   `/Users/mwhittaker/Projects/github/lab-networking/docs/runbooks/lml-automation-wifi.md`,
   Option D (lines 137–185). It defines xanthiosite `192.168.50.112:19000` to
   Z300 `192.168.60.65:9000`, with xanthiosite joined to `lml_automation`.
2. SSH aliases were reviewed before use. `~/.ssh/config:1` includes the generated
   lab config. `lab-networking/ssh/lab.ssh_config:8-14` defines `moissanite`, and
   lines 84–91 define `xanthiosite` as `192.168.50.112` via `moissanite`.
3. From moissanite, a bounded GET to
   `http://192.168.50.112:19000/instrument/id` connected in 0.002 s, then ended
   with `curl: (56) Recv failure: Connection reset by peer` after 2.23 s. In the
   same command, `http://192.168.50.112:53000/` returned the OpenBuilds CONTROL
   HTML, showing the bridge host and Opal/gantry forwarding path were reachable.
4. On xanthiosite, `Wi-Fi` was `Up`, joined to SSID `lml_automation`, with address
   `192.168.60.207/24`, 62% signal, and 44 Mbps link speed.
5. `netsh interface portproxy show v4tov4` showed the expected
   `192.168.50.112:19000 -> 192.168.60.65:9000` rule. A listener existed on
   `192.168.50.112:19000`. Windows IP Helper (`iphlpsvc`) was
   `Running`/`Automatic`, and scheduled task `lab-wifi-watchdog` was `Ready`.
6. A direct xanthiosite GET to
   `http://192.168.60.65:9000/instrument/id` failed with `Unable to connect to
   the remote server`, bypassing portproxy entirely.
7. The bounded identity probe was conclusive: `ping -n 2 -w 1000
   192.168.60.65` received 2/2 replies, TTL 64, 4–6 ms. The refreshed ARP entry
   was `192.168.60.65  40-06-a0-a0-6f-64  dynamic`, matching the Z300 MAC already
   known to xanthiosite.
8. A final direct `TcpClient.ConnectAsync(...).GetResult()` to
   `192.168.60.65:9000` returned: `No connection could be made because the
   target machine actively refused it 192.168.60.65:9000`. This was refusal,
   not timeout.
9. Parent-session evidence at 21:34:52Z showed the gantry freshly `Idle`, while
   the sole acquisition blocker remained analyzer reachability (`Errno 104`).

## Shortest remedy

At the Z300 itself, restart or re-enable its API/server function that listens on
TCP `9000` (or perform the vendor-prescribed analyzer/UI restart if the API has
no separate control), then verify `http://192.168.60.65:9000/instrument/id`.
No bridge, portproxy, Wi-Fi, portal URL, firewall, or repository change is
supported by the evidence. No repair was attempted in this diagnosis.

## Commands and verification limits

Read-only commands used (all SSH calls used existing aliases, `BatchMode=yes`,
and `ConnectTimeout` of 5–8 seconds): `ssh moissanite` plus bounded `curl -m 5`;
`ssh xanthiosite` plus `Get-NetAdapter`, `Get-NetIPAddress`, `netsh wlan show
interfaces`, `netsh interface portproxy show v4tov4`, `Get-NetTCPConnection`,
`Get-Service iphlpsvc`, `Get-ScheduledTask`, `Get-NetNeighbor`, bounded direct
HTTP/TCP checks, and a two-packet `ping` followed by `arp -a` filtering `.65`.
There were no broad scans, physical commands, service/configuration changes,
reboots, cancellations, or provider switches.

Repository tests were run only to satisfy the task verification convention;
this diagnosis changed no product code. The base invocation
`PYTHONPATH=src python3 -m pytest tests/ -q` stopped during collection with 30
errors because system Python lacks `scipy` (0 tests collected/passed). The
pymatgen-enabled invocation
`PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`
reached 233 passed, 1 skipped, and 55 passed subtests before it was interrupted
after 207.35 seconds to preserve the diagnosis's bounded scope; the remaining
suite was not verified. The internal state of the Z300 API process could not
be inspected remotely; active TCP refusal proves only that no accepting
listener was available at the probed time.

## File changes

- `reports/2026-09-21-analyzer-reachability.md:1` — added this read-only
  diagnostic report. No source, configuration, service, or remote state was
  changed.
