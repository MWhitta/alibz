# Z300 legacy API wake/restart audit — 2026-09-21

## Answer

The pinned Profile Builder client exposes **no restart, reboot, service-start, wake,
backlight, network-configuration, ADB, or shell control** for the Z300. Its only recovery
behavior is reconnect/poll from the client side. This proves absence from Profile Builder
2.19.1a; it does not prove that no undocumented instrument-local, Android, factory, or
vendor-support mechanism exists.

No live instrument access, source changes, or tests were performed. No provider switch
occurred.

## Pinned artifact evidence

Inspected read-only:
`/private/tmp/pantheum-acquisition-ProfileBuilder.jar`, SHA-256
`c8164f159eb610b0c7b154df7876010e9699332da5d9cf0704138c59c5814dd7`.

`com.sciaps.common.webserver.LIBZHttpClient` contains only instrument identity/default
settings/save settings/argon/screenshot; test fire/raster/list/spectra/cancel; and CRUD
for standards, regions, models, IR, fingerprint/grade libraries, and drift correction.
The concrete control endpoints include `/data/firetest`, `/laser/raster`, and
`/data/cancel`; none starts or restarts the HTTP server or Android application.
`abortTest()` is the `/data/cancel` operation, not a server restart.

The apparently relevant `ConnectToLibzExecutor.shutDown()` is also not an instrument
hook: its complete bytecode loads the local `ScheduledExecutorService` field and invokes
Java `ScheduledExecutorService.shutdown()`. It only stops Profile Builder's polling
executor and sends no HTTP request.

A whole-JAR constant-pool sweep found no SciAps endpoint or command containing restart,
reboot, wake, shutdown, service, network, Wi-Fi, ADB, shell, or Android `am start` terms.
Matches for generic “service” and “shell” belong to bundled third-party Java libraries,
not `com.sciaps` instrument control.

## Existing observations and practical implication

The September 14 investigation already found no wake/backlight endpoint in the full
Profile Builder path list and no ADB client on Opal
(`instrument-control/reports/2026-09-14-z300-display-and-upload.md:41-46`). When port
9000 stopped listening while the analyzer still answered ping, returning to the home
screen did not restore it; the server autostarted independently of the visible launcher,
and rebooting the analyzer with the cable out restored service
(`2026-09-14-z300-display-and-upload.md:480-504`). The follow-up assigned restarting the
LIBS application locally to the operator (`:573-579`).

The September 17 probe confirms this firmware serves neither the newer `/api/v2` API on
8080 nor `/api/v2` on the legacy port (`2026-09-17-z300-laser-free-probe.md:13-18`), so
newer API capabilities cannot be assumed for this Z300.

Current main-session evidence is consistent: the device answers ping, while TCP 9000
and ADB-over-network TCP 5555 are actively refused. With neither the legacy HTTP server
nor an ADB listener reachable, there is no remotely callable wake/restart path in the
examined client protocol. The evidence-backed recovery remains an operator action on the
analyzer (restart the LIBS app if available) or a device reboot. Do not send repeated
fire/cancel requests as a wake attempt.

Unverified: undocumented firmware endpoints; USB ADB availability/authorization when
physically attached; Android package/service names; factory menus; and any vendor-only
recovery command. Those require primary vendor documentation or controlled bench access.
