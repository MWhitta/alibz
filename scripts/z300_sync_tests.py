#!/usr/bin/env python3
"""Archive every test on the Z300, not only portal-fired runs.

This is a bounded, read-only *producer of producers*: it enumerates the
instrument's current, non-deleted ``type:"test"`` documents over the same
read-only ``sqlite3 -batch -noheader`` SELECT path as
``z300_opal_ingest.revision_query`` (never by name or "latest", never a
write), and hands each not-yet-archived test to ``z300_opal_ingest.ingest``
in-process. The instrument copy is never deleted or altered by this stage.

Each test becomes ``<archive-root>/z300-<test-id>/`` via the existing exact,
idempotent, atomically-published producer, with ``expected_shots`` taken from
the stored ``config`` (``numShotsPerLocation * rasterNumLocations``) and the
producer's ``--min-shots`` rule. Tests already archived under the archive root
or an ``--also-archived`` directory are skipped by reading each bundle's
``manifest.json`` ``test_id``. A test whose bundle is not stored yet is
"pending" and retried while its ``unixTime`` is young; an empty, averaged-only
or permanently-unsafe test is recorded in a skip index so it is not re-read
every pass. One test's failure never stops the pass.

To drain the ~thousands of historical tests on the card while keeping new
tests first, each pass runs two *metadata* listings of up to ``--list-limit``
rows (the archive budget is the separate, smaller ``--limit``): a fresh window
of the newest tests (no cursor), then a backfill window strictly below a
persisted ``backfill_cursor`` (``<state>/cursor.json``). The cursor is
initialised to the minimum sequence of the first fresh window and only advances
to the minimum of a backfill window once *every* row in that window is resolved
(archived or skipped) -- a still-pending test holds it, so it is never skipped.
When a backfill window is empty the walk is ``backfill_complete``; ``--rescan``
resets it. Archive attempts are spent fresh-first, then backfill, up to
``--limit`` total.

A pass-level lock in the state root prevents two passes overlapping; each test
additionally holds the producer's per-run lock. ``--dry-run`` prints the plan
to stderr and touches nothing. The pass summary is JSON on stdout
(``listed``/``listed_fresh``/``listed_backfill``/``new``/``archived``/
``pending``/``skipped``/``errors``/``backfill_cursor``/``backfill_complete``/
``archive_root``); exit 0 when the pass ran, 2 on usage, 3 when the instrument
was unreachable (nothing listed).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import time

try:  # Packaged import, like the producer importing the decoder.
    from . import z300_opal_ingest as producer
except ImportError:  # Direct script execution on Opal.
    import z300_opal_ingest as producer


DEFAULT_ARCHIVE_ROOT = Path(r"C:\LabData\LIBS\z300-tests")
DEFAULT_ALSO_ARCHIVED = Path(r"C:\LabData\LIBS\pantheum-acquisitions")
DEFAULT_STATE_ROOT = Path(r"C:\LabData\.labdesk\libs\z300-tests")
DEFAULT_LIST_LIMIT = 200
DEFAULT_ARCHIVE_LIMIT = 20
DEFAULT_RETRY_WINDOW_HOURS = 24.0
MAX_LIST_LIMIT = 1000
PASS_LOCK_ID = "z300-sync-tests"

# The producer's own test-id rule and pantheum's run-id rule; a test is only
# archived when "z300-<test-id>" satisfies BOTH (validate the id, note the
# pantheum rule). Others are skipped and recorded.
PRODUCER_TEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
PANTHEUM_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{7,47}")

# Android System.currentTimeMillis stores unixTime in milliseconds; a bare
# epoch-seconds value (< ~1e11) is used as-is.
MILLISECOND_THRESHOLD = 1e11


@dataclass(frozen=True)
class TestRow:
    test_id: str
    sequence: int
    unix_time: float | None
    doc: dict = field(repr=False)


def enumeration_query(list_limit: int, before: int | None = None) -> str:
    """Read-only SELECT of current, non-deleted test docs, newest sequence first.

    Same ``sqlite3 -batch -noheader`` transactional style as
    ``producer.revision_query``; never selects by name/time/latest and never
    writes. ``before`` adds ``r.sequence < N`` for the backfill window.
    ``replace(json,' ','')`` makes the ``type`` filter tolerant of any JSON
    spacing; Python re-checks ``type == "test"`` authoritatively.
    """
    if isinstance(list_limit, bool) or not isinstance(list_limit, int) or not 1 <= list_limit <= MAX_LIST_LIMIT:
        raise ValueError("enumeration list-limit out of range")
    where = ("r.current=1 AND r.deleted=0 "
             "AND replace(r.json,' ','') LIKE '%\"type\":\"test\"%'")
    if before is not None:
        if isinstance(before, bool) or not isinstance(before, int) or before < 1:
            raise ValueError("backfill cursor must be a positive integer")
        where += " AND r.sequence < " + str(int(before))
    sql = (
        "BEGIN; SELECT hex(d.docid)||'|'||CAST(r.sequence AS TEXT)||'|'||hex(r.json) "
        "FROM docs AS d JOIN revs AS r ON r.doc_id=d.doc_id "
        "WHERE " + where + " ORDER BY r.sequence DESC LIMIT " + str(int(list_limit)) + "; COMMIT;"
    )
    # Old sqlite3 opens a missing path by creating a DB; guard the mount like
    # the producer does so an unreachable card fails instead of listing empty.
    return ("test -f " + shlex.quote(producer.DB_PATH) + " && sqlite3 -batch -noheader "
            + shlex.quote(producer.DB_PATH) + " " + shlex.quote(sql))


def list_response_limit(list_limit: int) -> int:
    # hex doubles each document; a per-row ceiling times the bounded LIMIT.
    return list_limit * (2 * producer.MAX_DOCUMENT_BYTES + 4096) + 65536


def parse_enumeration(raw: bytes) -> list[TestRow]:
    """Parse the listing defensively, like ``producer.parse_revision``.

    A malformed or oversized row is skipped rather than aborting the pass; the
    SQL pre-filter plus this ``type == "test"`` check keep it to real tests.
    """
    rows: list[TestRow] = []
    for line in raw.strip().splitlines():
        if not line:
            continue
        try:
            docid_hex, sequence_text, json_hex = line.decode("ascii").split("|")
            test_id = bytes.fromhex(docid_hex).decode("utf-8")
            sequence = int(sequence_text)
            document_bytes = bytes.fromhex(json_hex)
            if len(document_bytes) > producer.MAX_DOCUMENT_BYTES:
                continue
            doc = json.loads(document_bytes)
        except (ValueError, UnicodeError):
            continue
        if not isinstance(doc, dict) or doc.get("type") != "test" or sequence < 1:
            continue
        unix_time = doc.get("unixTime")
        if isinstance(unix_time, bool) or not isinstance(unix_time, (int, float)):
            unix_time = None
        rows.append(TestRow(test_id=test_id, sequence=sequence, unix_time=unix_time, doc=doc))
    return rows


def run_id_for(test_id: str) -> tuple[str | None, str | None]:
    """(run_id, None) when "z300-<test-id>" is safe, else (None, reason)."""
    if not PRODUCER_TEST_ID.fullmatch(test_id):
        return None, "unsafe test id (producer rule)"
    run_id = "z300-" + test_id
    if not PANTHEUM_RUN_ID.fullmatch(run_id):
        return None, "run id violates pantheum rule"
    return run_id, None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, float) and value.is_integer() and value >= 1:
        return int(value)
    if isinstance(value, str) and value.isdigit() and int(value) >= 1:
        return int(value)
    return None


def expected_shots_from_doc(doc: dict) -> int | None:
    """numShotsPerLocation * rasterNumLocations from the stored config, or None.

    Reads only documented fields (``stored_configuration``-style); it never
    invents a shot count when the config is absent or malformed.
    """
    config = doc.get("config")
    if not isinstance(config, dict):
        return None
    per_location = _positive_int(config.get("numShotsPerLocation"))
    locations = _positive_int(config.get("rasterNumLocations"))
    if per_location is None or locations is None:
        return None
    product = per_location * locations
    if not 1 <= product <= producer.MAX_SHOTS:
        return None
    return product


def archived_test_ids(directories: list[Path]) -> set[str]:
    """test_ids already archived, read once from each ``<dir>/*/manifest.json``."""
    ids: set[str] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for manifest in directory.glob("*/manifest.json"):
            try:
                data = json.loads(manifest.read_bytes())
            except (OSError, ValueError):
                continue
            test_id = data.get("test_id") if isinstance(data, dict) else None
            if isinstance(test_id, str):
                ids.add(test_id)
    return ids


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(producer.json_bytes(obj))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _epoch_seconds(unix_time: float | None) -> float | None:
    if unix_time is None:
        return None
    value = float(unix_time)
    return value / 1000.0 if value > MILLISECOND_THRESHOLD else value


def _too_old(unix_time: float | None, retry_window_hours: float, now: float) -> bool:
    seconds = _epoch_seconds(unix_time)
    if seconds is None:  # No timestamp: cannot age out, keep retrying.
        return False
    return (now - seconds) > retry_window_hours * 3600.0


def _also_archived(args: argparse.Namespace) -> list[Path]:
    return list(args.also_archived) if args.also_archived else [DEFAULT_ALSO_ARCHIVED]


def build_plan(rows: list[TestRow], args: argparse.Namespace) -> list[dict]:
    archived_ids = archived_test_ids([args.archive_root, *_also_archived(args)])
    skipped_index = load_json(args.state_root / "skipped.json")
    plan: list[dict] = []
    for row in rows:
        test_id = row.test_id
        run_id, reason = run_id_for(test_id)
        expected = expected_shots_from_doc(row.doc)
        if test_id in archived_ids:
            action = "already_archived"
        elif test_id in skipped_index:
            action = "already_skipped"
        elif run_id is None:
            action = "skip:" + str(reason)
        elif expected is None:
            action = "skip:no stored configuration"
        else:
            action = "would_attempt"
        plan.append({"test_id": test_id, "sequence": row.sequence, "run_id": run_id,
                     "expected_shots": expected, "action": action})
    return plan


def _process_window(rows, args, adb, now, archived_ids, pending_index, skipped_index,
                    counters, budget) -> bool:
    """Archive/skip each unresolved row, spending the shared attempt budget.

    Returns whether *every* row is resolved (already archived, archived now, or
    skipped). A still-pending, deferred (budget exhausted) or errored row leaves
    it unresolved so the backfill cursor cannot advance past it.
    """
    all_resolved = True
    for row in rows:
        test_id = row.test_id
        if test_id in archived_ids or test_id in skipped_index:
            continue  # already resolved; never re-read.
        run_id, reason = run_id_for(test_id)
        if run_id is None:  # cheap terminal skip; no attempt spent.
            skipped_index[test_id] = {"reason": reason, "unixTime": row.unix_time}
            counters["new"] += 1
            counters["skipped"] += 1
            continue
        expected = expected_shots_from_doc(row.doc)
        if expected is None:
            skipped_index[test_id] = {"reason": "no stored configuration", "unixTime": row.unix_time}
            counters["new"] += 1
            counters["skipped"] += 1
            continue
        if budget[0] <= 0:  # out of attempts this pass: defer, stay unresolved.
            all_resolved = False
            continue
        budget[0] -= 1
        counters["new"] += 1
        config = producer.Config(
            run_id=run_id, test_id=test_id, expected_shots=expected,
            adb=args.adb, serial=args.serial, adb_port=args.adb_port,
            archive_root=args.archive_root, timeout=args.timeout, min_shots=args.min_shots)
        try:
            producer.ingest(config, adb)
        except producer.PendingData as exc:
            if _too_old(row.unix_time, args.retry_window_hours, now):
                skipped_index[test_id] = {"reason": str(exc), "unixTime": row.unix_time}
                pending_index.pop(test_id, None)
                counters["skipped"] += 1
            else:
                pending_index[test_id] = {"reason": str(exc), "unixTime": row.unix_time}
                counters["pending"] += 1
                all_resolved = False
        except producer.IngestError as exc:
            skipped_index[test_id] = {"reason": str(exc), "unixTime": row.unix_time}
            pending_index.pop(test_id, None)
            counters["skipped"] += 1
        except Exception as exc:  # One test's failure never stops the pass.
            counters["errors"] += 1
            all_resolved = False
            print(f"z300 sync: {test_id}: {exc}", file=sys.stderr)
        else:
            archived_ids.add(test_id)
            counters["archived"] += 1
            pending_index.pop(test_id, None)
    return all_resolved


def execute_pass(fresh_rows, backfill_rows, backfill_listed, backfill_cursor,
                 backfill_complete, args: argparse.Namespace, adb, now: float) -> dict:
    archived_ids = archived_test_ids([args.archive_root, *_also_archived(args)])
    pending_path = args.state_root / "pending.json"
    skipped_path = args.state_root / "skipped.json"
    cursor_path = args.state_root / "cursor.json"
    pending_index = load_json(pending_path)
    skipped_index = load_json(skipped_path)
    counters = {"listed": len(fresh_rows) + len(backfill_rows), "new": 0, "archived": 0,
                "pending": 0, "skipped": 0, "errors": 0,
                "listed_fresh": len(fresh_rows), "listed_backfill": len(backfill_rows)}
    budget = [args.limit]
    # Fresh first, so a new test is always archived before older backlog.
    _process_window(fresh_rows, args, adb, now, archived_ids, pending_index,
                    skipped_index, counters, budget)
    backfill_all_resolved = _process_window(backfill_rows, args, adb, now, archived_ids,
                                            pending_index, skipped_index, counters, budget)
    if backfill_listed and not backfill_complete:
        if not backfill_rows:
            backfill_complete = True  # nothing older remains.
        elif backfill_all_resolved:  # only step past a fully-resolved window.
            backfill_cursor = min(row.sequence for row in backfill_rows)
    write_json(pending_path, pending_index)
    write_json(skipped_path, skipped_index)
    write_json(cursor_path, {"backfill_cursor": backfill_cursor,
                             "backfill_complete": backfill_complete})
    counters["archive_root"] = str(args.archive_root)
    counters["backfill_cursor"] = backfill_cursor
    counters["backfill_complete"] = backfill_complete
    return counters


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--also-archived", type=Path, action="append",
                        help=f"skip tests archived here too (default: {DEFAULT_ALSO_ARCHIVED})")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--list-limit", type=int, default=DEFAULT_LIST_LIMIT,
                        help=f"metadata rows enumerated per window, 1..{MAX_LIST_LIMIT} "
                             f"(default {DEFAULT_LIST_LIMIT})")
    parser.add_argument("--limit", type=int, default=DEFAULT_ARCHIVE_LIMIT,
                        help=f"archive attempts per pass, fresh-first (default {DEFAULT_ARCHIVE_LIMIT})")
    parser.add_argument("--rescan", action="store_true",
                        help="reset the backfill cursor and re-walk from the newest")
    parser.add_argument("--retry-window-hours", type=float, default=DEFAULT_RETRY_WINDOW_HOURS)
    parser.add_argument("--min-shots", type=int, default=None,
                        help="passed through to the producer (default: min(6, expected-shots))")
    parser.add_argument("--adb", default=None,
                        help="ADB executable (producer default when omitted)")
    parser.add_argument("--serial", default="0123456789ABCDEF")
    parser.add_argument("--adb-port", type=int, default=5038)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.adb is None:
        args.adb = r"C:\Users\Whittaker\AppData\Local\Temp\alibz-api-recovery-20260921\platform-tools\adb.exe"
    return args


def _cursor_state(args: argparse.Namespace) -> tuple[int | None, bool]:
    state = {} if args.rescan else load_json(args.state_root / "cursor.json")
    cursor = state.get("backfill_cursor")
    if isinstance(cursor, bool) or not isinstance(cursor, int):
        cursor = None
    complete = (not args.rescan) and bool(state.get("backfill_complete", False))
    return cursor, complete


def main(argv: list[str] | None = None, adb=None, now: float | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.list_limit <= MAX_LIST_LIMIT:
        print(f"z300 sync: --list-limit must be 1..{MAX_LIST_LIMIT}", file=sys.stderr)
        return 2
    if args.limit < 1:
        print("z300 sync: --limit must be >= 1", file=sys.stderr)
        return 2
    if args.retry_window_hours < 0:
        print("z300 sync: --retry-window-hours must be >= 0", file=sys.stderr)
        return 2
    if args.min_shots is not None and args.min_shots < 1:
        print("z300 sync: --min-shots must be >= 1", file=sys.stderr)
        return 2
    now = time.time() if now is None else now
    if adb is None:
        adb = producer.Adb(producer.Config(
            run_id="z300-sync-enumerate", test_id="enumerate", expected_shots=1,
            adb=args.adb, serial=args.serial, adb_port=args.adb_port,
            archive_root=args.archive_root, timeout=args.timeout))
    backfill_cursor, backfill_complete = _cursor_state(args)
    # Fresh window: newest tests, no cursor.
    try:
        fresh_raw = adb.read(enumeration_query(args.list_limit), list_response_limit(args.list_limit))
    except producer.IngestError as exc:
        print(f"z300 sync: instrument unreachable ({exc})", file=sys.stderr)
        print(json.dumps({"listed": 0, "listed_fresh": 0, "listed_backfill": 0, "new": 0,
                          "archived": 0, "pending": 0, "skipped": 0, "errors": 0,
                          "backfill_cursor": backfill_cursor, "backfill_complete": backfill_complete,
                          "archive_root": str(args.archive_root)}, sort_keys=True))
        return 3
    fresh_rows = parse_enumeration(fresh_raw)
    if backfill_cursor is None and fresh_rows:  # one-time init to the fresh minimum.
        backfill_cursor = min(row.sequence for row in fresh_rows)
    # Backfill window: strictly below the cursor, unless the walk is finished.
    backfill_rows: list[TestRow] = []
    backfill_listed = False
    if not backfill_complete and backfill_cursor is not None:
        try:
            backfill_raw = adb.read(enumeration_query(args.list_limit, before=backfill_cursor),
                                    list_response_limit(args.list_limit))
            backfill_rows = parse_enumeration(backfill_raw)
            backfill_listed = True
        except producer.IngestError as exc:  # reachable (fresh worked); skip backfill this pass.
            print(f"z300 sync: backfill listing unavailable ({exc})", file=sys.stderr)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "archive_root": str(args.archive_root),
                          "state_root": str(args.state_root), "list_limit": args.list_limit,
                          "limit": args.limit, "backfill_cursor": backfill_cursor,
                          "backfill_complete": backfill_complete,
                          "listed_fresh": len(fresh_rows), "listed_backfill": len(backfill_rows),
                          "plan_fresh": build_plan(fresh_rows, args),
                          "plan_backfill": build_plan(backfill_rows, args)},
                         sort_keys=True), file=sys.stderr)
        return 0
    try:
        with producer.run_lock(args.state_root, PASS_LOCK_ID):
            summary = execute_pass(fresh_rows, backfill_rows, backfill_listed,
                                   backfill_cursor, backfill_complete, args, adb, now)
    except producer.PendingData:
        print("z300 sync: another pass owns the state-root lock", file=sys.stderr)
        return 0
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
