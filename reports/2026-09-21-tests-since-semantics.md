# Z300 test-list query semantics — 2026-09-21

The query uses Unix epoch seconds, not milliseconds. The response contains test
ID strings, and the live firmware also returns an integer `next` cursor.

## Evidence

Pinned Profile Builder 2.19.1a JAR SHA-256:
`c8164f159eb610b0c7b154df7876010e9699332da5d9cf0704138c59c5814dd7`.
The main session inspected the existing local artifact without accessing device
application packages. Disassembly is retained in
`reports/2026-09-21-tests-since-bytecode.txt`.

`com.sciaps.view.acquisitions.TestViewPanel$3.onBackground` calls
`DateUtils.truncate(startDate, 5)`, then `Date.getTime()` at offset 10,
loads long 1000 at offset 13, and performs `ldiv` at offset 16. It passes
that quotient to `getTestsSince` at offset 63. End-date conversion repeats
the division at offsets 26–32. This establishes epoch seconds independently
of the misleading Date constructor used for logging inside the HTTP client.

`LIBZHttpClient.getTestsSince` declares generic signature
`(J)Ljava/util/List<Ljava/lang/String;>;`. It formats the long argument into
`/data/tests?since=%d` unchanged and deserializes the `data` field.

The user-approved live request `since=0` returned HTTP 200, 200 string IDs,
and `next=1613659041`. The response was 7828 bytes and took 1.335 seconds.
No spectrum endpoints were requested and the historical response was not saved.
Android logs show `NumberFormatException: Invalid int: "1790027760000"`
crashing the remote service twice. Therefore validation must reject values
outside 0..2147483647 before issuing any request, including booleans and floats.

## Consequences

The prior audit incorrectly inferred milliseconds from `new Date(long)` used
only for logging, and incorrectly described the return as `List<LIBZTest>`.
Those claims and the corresponding millisecond preflight recommendation are
superseded. The live test-ID response and concrete UI caller resolve both.

Safe polling can start at zero and follow validated server cursors to avoid
assuming the analyzer clock matches the host. It must enforce a finite page
budget and the acquisition deadline, detect non-progressing cursors, and match
the exact acknowledged test ID. A first-page-only query can miss recent tests.
A fire acknowledgement alone does not establish that physical work stopped.

No source was changed by this audit. No shots, movement, cancellation, replay,
APK export, or provider switch occurred.
