# Pantheum alibz gantry connection audit

Date: 2026-09-21. Scope: source review plus read-only status/handshake probes; no
`connectTo`, G-code, homing, unlock, alarm clear, or equipment mutation was sent.
No provider switch occurred.

## Finding

Pantheum and OpenBuildsCONTROL have two distinct connections:

1. `pantheum/alibz/motion.py::_SocketIOTransport.connect()` connects the
   Moissanite service to OpenBuildsCONTROL's Socket.IO server through
   `http://192.168.50.112:53000`.
2. OpenBuildsCONTROL must separately open opal's USB serial port to the BlackBox.
   Its UI emits `connectTo` with
   `{"port":"COM3","baud":115200,"type":"usb"}`. The server constructs a
   `SerialPort`, detects firmware, then publishes the result in
   `status.comms.connectionStatus`, `interfaces.activePort`, `activeBaud`, and
   `machine.firmware`.

Pantheum implements only step 1. It never emits `connectTo`, ignores
`connectionStatus`, `activePort`, the enumerated USB `ports`, and firmware identity,
and treats Socket.IO reachability plus a cached `runStatus == "Idle"` as sufficient
for movement. That explains why an operator currently has to click Connect in
OpenBuildsCONTROL. It also means stale Idle/position values can make the portal look
ready after the serial controller has disconnected.

The live endpoint is OpenBuildsCONTROL 1.0.388 on opal. Read-only probes from
Moissanite established that Engine.IO v4 polling and WebSocket both work with the
installed `python-socketio 5.16.4` / `python-engineio 4.14.0`; the earlier service
`TimeoutError` is therefore not an Engine.IO-version incompatibility. A read-only
`status` event showed the controller currently connected manually as USB `COM3`,
115200 baud, FTDI VID:PID `0403:6001`, serial `A10OF9X9`, grblHAL 1.1f. The
implementation must not commit that live port or serial to the public example config;
they belong in Moissanite's private config after operator review.

## Exact upstream contract

Primary source is the installed OpenBuildsCONTROL tag `v1.0.388`; comparison
against `v1.0.390` (commit `4bc126df96493e03ddfa76dc5390593f2f5b2f2c`)
shows the same contract. Exact v1.0.388 sources and hashes used for verification
are `/private/tmp/openbuilds-control-v1.0.388-index.js`
(`f3749215194ee2dbdd94afdedb4ba4092cabe926949aeefd0de997686ae02d33`) and
`/private/tmp/openbuilds-control-v1.0.388-websocket.js`
(`9d91fa69fe3c7a7c381ea7cee23933097752a4a4254e5589d6fa0a6cc007f9ee`).
`app/js/websocket.js::selectPort()` emits:

```js
socket.emit('connectTo', {port: port, baud: 115200, type: 'usb'})
```

`index.js` registers `socket.on('connectTo', function(data) {...})`, accepts it only
while `status.comms.connectionStatus < 1`, opens `new SerialPort({path:
data.port, baudRate: parseInt(data.baud), ...})`, and publishes status every 100 ms.
Connection status values are documented in source as 0 disconnected, 1 opening, 2
connected, 3 running, 4 paused, 5 alarm, and 6 firmware upgrade. USB inventory comes
from `SerialPort.list()` and includes `path`, `vendorId`, `productId`, and
`serialNumber`.

There is no Socket.IO acknowledgment contract. In v1.0.229, v1.0.388, and v1.0.390,
the server handlers for `connectTo`, `runCommand`, `setZero`, `stop`, and
`clearAlarm` accept only `data`; none accepts or invokes an ack callback. The pinned
AECP frontend passes callbacks, but they are never called; the pinned AECP ROS adapter
correctly uses `client.emit()`. Pantheum's `_SocketIOTransport.emit()` instead uses
`Client.call()` for `runCommand`, `setZero`, and `stop`. A timeout can therefore be
reported after OpenBuildsCONTROL already queued the command. This is an ambiguous
physical-action outcome and must be fixed with the connection change; commands must
never be retried automatically.

Source links:

- OpenBuilds client: https://github.com/OpenBuilds/OpenBuilds-CONTROL/blob/v1.0.388/app/js/websocket.js#L1088-L1116
- OpenBuilds server connect handler: https://github.com/OpenBuilds/OpenBuilds-CONTROL/blob/v1.0.388/index.js#L1104-L1733
- OpenBuilds command handlers: https://github.com/OpenBuilds/OpenBuilds-CONTROL/blob/v1.0.388/index.js#L1766-L2228
- Pinned AECP frontend: https://github.com/Living-Minerals-Lab/AECP_frontend/blob/2624590e116edbf7916aa9dc062d204dfc743047/src/components/GantryBlock.vue
- Pinned AECP adapter: https://github.com/Living-Minerals-Lab/AECP_ROS2/blob/349c5d055363166c06bf812f4a09eef094e1e568/src/utils/utils/gantry_ctrl.py

## Minimal safe implementation

Implement an explicit reserved portal action named `connect`, gated independently by
`gantry.connect`. A GET/status poll must never open the serial port. The POST action may
emit `connectTo` only after resolving a single configured target:

- exact configured `motion.controller.port`, present in the latest upstream port list;
  or
- one and only one port matching configured immutable identity fields, preferably all
  of `vendor_id`, `product_id`, and `serial_number`.

Zero matches, multiple matches, a configured port whose identity does not match, no
fresh upstream status, or an already opening/running/paused/firmware-upgrade controller
must refuse without emitting. Never select the first port, infer from `manufacturer`,
accept a port/identity from the HTTP request, close another active port, or emit home,
unlock, reset, or G-code as part of connection. If the expected controller is already
connected, the action is an idempotent no-op only when active port/baud/identity match.

Track and expose both layers: `socket_connected`, numeric `connection_status`,
bounded `connect_pending`, `active_port`, `active_baud`, sanitized
`available_ports`, and firmware identity.
The existing `connected` field should mean a verified serial controller, not merely a
Socket.IO session. Movement requires status 2 plus Idle and the configured target;
abort/clear-alarm may remain available in appropriate nonzero controller states.
Clear cached position/readiness whenever the Socket.IO session drops so stale data
cannot authorize a later command.

Use fire-and-forget `Client.emit()` for the upstream events because the vendor server
has no ack. A successful local emit means only “submitted to OpenBuildsCONTROL”; it is
not motion completion. Observe status and the upstream `ok` event where correlation is
possible, never resend after timeout/disconnect, and describe uncertain outcomes as
such. The explicit connect action can be verified by later status polling because it
does not itself move, unlock, reset, or home the gantry.

## Files and tests

Backend ownership is `pantheum/alibz/motion.py` plus
`tests/test_alibz_motion.py`. Required tests cover exact `connectTo` payload; disabled
action; missing/stale port inventory; zero/ambiguous identity matches; explicit port
with mismatched identity; already-connected matching and conflicting targets; serial
disconnect clearing readiness; movement refusal when Socket.IO is up but controller
status is 0/1/3/4/6; alarm recovery at status 5; and proof that vendor events use
`emit`, not `call` or retries. The existing real local Socket.IO fake must omit ack
returns so it mirrors OpenBuildsCONTROL.

Follow-on integration belongs in the Pantheum repository:

- `web/alibz/index.html`, `app.js`, and `styles.css`: a reserved-user Connect button,
  separate service/controller indicators, active-port detail, and no automatic action
  from rendering or polling.
- `config/alibz.example.json`: disabled/example controller selector and
  `gantry.connect`; no live serial number.
- `deploy/alibz/enable-gantry.py` and its tests: optional reviewed controller selector,
  preserving private operator values and requiring opt-in to `gantry.connect`.
- `deploy/alibz/README.md` and architecture/decision record: source-backed protocol,
  private config procedure, first-connection bench check, and the lack of vendor ack.

No change is required in `instrument-control`: it remains the source/provenance home,
while the connection lifecycle belongs to the Pantheum adapter that owns portal motion.

## Nearby Z300 rate evidence

This is independent of gantry connection, but the nearby verified record is clear:
`instrument-control/reports/2026-09-17-z300-laser-free-probe.md` establishes that
`pulsePeriod` and `cleaningPulsePeriod` are milliseconds: 20 means 50 Hz and 100 means
10 Hz. Therefore a 100 Hz Test Rate maps to `pulsePeriod: 10`, not `100` and not a raw
Hz field. Integration delay/period units remain unverified.

## Staged implementation and validation

Implemented in `/private/tmp/pantheum-acquisition-20260921`:

- `pantheum/alibz/motion.py`: explicit one-shot `connect`, exact port/identity
  resolution, bounded pending suppression, split Socket.IO/serial status, fresh
  complete-status gating, unknown position as `{}`, stale-state clearing, and
  vendor-compatible fire-and-forget emits without automatic retry.
- `tests/test_alibz_motion.py`: safe-selection, idempotence, ambiguity, wrong-port,
  missing/stale-status, controller-state, disconnect, exact-payload, and no-ack
  loopback coverage.
- `deploy/alibz/enable-gantry.py` and `tests/test_enable_gantry.py`: optional exact
  controller pinning that preserves existing motion tuning and selector fields.
- `config/alibz.example.json` and `deploy/alibz/README.md`: safe placeholder
  selector, separate `gantry.connect` gate, observed deployment identity, and
  commissioning semantics.

Local targeted result: 42 tests passed with the three real Socket.IO tests skipped
because the base interpreter lacks those optional packages. The main session reran
the full motion test module with temporary Socket.IO/aiohttp dependencies: 39 passed,
0 skipped. JSON parsing and Python byte-compilation also passed. No live `connectTo`
or physical command was sent.
