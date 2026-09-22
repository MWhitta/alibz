"""Enumerate-and-archive tests: synthetic SQLite/FlatBuffers, no instrument."""
import json
import shlex
import sqlite3

import pytest

from scripts import z300_opal_ingest as producer
from scripts import z300_sync_tests as sync

try:  # Reuse the producer test's FlatBuffer/legacy bundle builders.
    from tests.test_z300_opal_ingest import bundle, legacy_bundle
except ImportError:  # pytest prepend mode inserts tests/ onto sys.path.
    from test_z300_opal_ingest import bundle, legacy_bundle


def make_doc(*, shots=2, bundle_name="bundle", legacy=None, only_avg=False,
             num_avg=1, no_config=False, no_bundle=False, unix_time=1_700_000_000):
    """A current, non-deleted CBL test document with a stored config."""
    doc = {"type": "test", "unixTime": unix_time, "displayName": "field test"}
    if not no_config:
        doc["config"] = {"numShotsPerLocation": shots, "rasterNumLocations": 1,
                         "numShotsToAvg": num_avg}
    if only_avg:
        doc["onlyAvgSaved"] = True
    if not no_bundle:
        doc["shotTable"] = {"all": legacy} if legacy else {"all_fb": bundle_name}
    return doc


class SyncMockAdb:
    """Run the producer/enumerator's real SQL against a synthetic CBL1 schema."""

    def __init__(self, entries):
        # entries: list of dicts {test_id, document, blob?, sequence?}
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("""
            CREATE TABLE docs(doc_id INTEGER PRIMARY KEY, docid TEXT);
            CREATE TABLE revs(sequence INTEGER PRIMARY KEY, doc_id INTEGER,
                              revid TEXT, current INTEGER, deleted INTEGER, json BLOB);
        """)
        self.blobs = {}
        for i, entry in enumerate(entries, start=1):
            doc = entry["document"]
            # Compact JSON, exactly as Couchbase Lite 1.x stores it.
            raw = json.dumps(doc, separators=(",", ":")).encode()
            self.db.execute("INSERT INTO docs VALUES(?,?)", (i, entry["test_id"]))
            self.db.execute("INSERT INTO revs VALUES(?,?,?,?,?,?)",
                            (entry.get("sequence", i), i, "1-exact", 1, 0, raw))
            blob = entry.get("blob")
            if blob is not None:
                table = doc.get("shotTable", {})
                if table.get("all_fb"):
                    path = producer.SPECTRA_PATH + "/" + table["all_fb"]
                else:
                    digest = table["all"]
                    path = producer.LEGACY_SPECTRA_PATH + "/" + digest[:2] + "/" + digest
                self.blobs[path] = blob
        self.db.commit()
        self.commands = []
        self._next_doc_id = len(entries) + 1

    def add_test(self, entry):
        """Insert a test mid-run (e.g. a newer acquisition appearing on the card)."""
        i = self._next_doc_id
        self._next_doc_id += 1
        doc = entry["document"]
        raw = json.dumps(doc, separators=(",", ":")).encode()
        self.db.execute("INSERT INTO docs VALUES(?,?)", (i, entry["test_id"]))
        self.db.execute("INSERT INTO revs VALUES(?,?,?,?,?,?)",
                        (entry.get("sequence", i), i, "1-exact", 1, 0, raw))
        blob = entry.get("blob")
        if blob is not None:
            table = doc.get("shotTable", {})
            path = (producer.SPECTRA_PATH + "/" + table["all_fb"] if table.get("all_fb")
                    else producer.LEGACY_SPECTRA_PATH + "/" + table["all"][:2] + "/" + table["all"])
            self.blobs[path] = blob
        self.db.commit()

    def read(self, command, limit):
        self.commands.append(command)
        args = shlex.split(command)
        if args[0] == "test":
            assert args[:4] == ["test", "-f", producer.DB_PATH, "&&"]
            rest = args[4:]
            assert rest[:4] == ["sqlite3", "-batch", "-noheader", producer.DB_PATH]
            statements = rest[4].split(";")
            assert statements[0].strip() == "BEGIN" and statements[2].strip() == "COMMIT"
            self.db.execute("BEGIN")
            rows = self.db.execute(statements[1]).fetchall()
            self.db.commit()
            result = b"".join((row[0] + "\n").encode() for row in rows)
        else:
            assert args[0] == "cat"
            result = self.blobs[args[1]]
        assert len(result) <= limit
        return result

    def cat_paths(self):
        return [shlex.split(c)[1] for c in self.commands if c.startswith("cat ")]


@pytest.fixture
def roots(tmp_path):
    class Roots:
        archive = tmp_path / "z300-tests"
        also = tmp_path / "pantheum-acquisitions"
        state = tmp_path / "state"
    return Roots


def cli_args(roots, *extra):
    return ["--archive-root", str(roots.archive), "--also-archived", str(roots.also),
            "--state-root", str(roots.state), *extra]


def run(roots, adb, *extra, now=None):
    return sync.main(cli_args(roots, *extra), adb=adb, now=now)


def summary_of(capsys):
    out = capsys.readouterr().out.strip()
    return json.loads(out) if out else None


# --- enumeration + parsing ------------------------------------------------

def test_enumeration_lists_only_current_nondeleted_tests_newest_first():
    adb = SyncMockAdb([
        {"test_id": "alpha", "document": make_doc()},
        {"test_id": "beta", "document": make_doc()},
    ])
    # A non-test doc, a deleted rev and an obsolete rev must not appear.
    adb.db.execute("INSERT INTO docs VALUES(90,'not-a-test')")
    adb.db.execute("INSERT INTO revs VALUES(90,90,'1-x',1,0,?)",
                   (json.dumps({"type": "note"}, separators=(",", ":")).encode(),))
    adb.db.execute("INSERT INTO docs VALUES(91,'deleted-test')")
    adb.db.execute("INSERT INTO revs VALUES(91,91,'1-x',1,1,?)",
                   (json.dumps(make_doc(), separators=(",", ":")).encode(),))
    adb.db.commit()
    raw = adb.read(sync.enumeration_query(20), sync.list_response_limit(20))
    rows = sync.parse_enumeration(raw)
    assert [r.test_id for r in rows] == ["beta", "alpha"]  # sequence desc
    assert all(r.doc["type"] == "test" for r in rows)


# --- happy path: archive new, skip already-archived -----------------------

def test_archives_two_new_and_skips_one_already_under_also_archived(roots, capsys):
    adb = SyncMockAdb([
        {"test_id": "aaa-one", "document": make_doc(), "blob": bundle(2)},
        {"test_id": "bbb-two", "document": make_doc(), "blob": bundle(2)},
        {"test_id": "ccc-old", "document": make_doc(), "blob": bundle(2)},
    ])
    # ccc-old already lives under an --also-archived directory.
    prior = roots.also / "z300-ccc-old"
    prior.mkdir(parents=True)
    (prior / "manifest.json").write_text(json.dumps({"test_id": "ccc-old"}))

    assert run(roots, adb) == 0
    result = summary_of(capsys)
    assert (result["listed"], result["new"], result["archived"], result["skipped"],
            result["pending"], result["errors"]) == (3, 2, 2, 0, 0, 0)
    assert (roots.archive / "z300-aaa-one" / "native.zip").is_file()
    assert (roots.archive / "z300-bbb-two" / "native.zip").is_file()
    assert not (roots.archive / "z300-ccc-old").exists()
    assert result["archive_root"] == str(roots.archive)


def test_second_pass_is_a_no_op_with_identical_archives(roots, capsys):
    adb = SyncMockAdb([
        {"test_id": "aaa-one", "document": make_doc(), "blob": bundle(2)},
        {"test_id": "bbb-two", "document": make_doc(), "blob": bundle(2)},
    ])
    assert run(roots, adb) == 0
    first = summary_of(capsys)
    frozen = {p: (roots.archive / p / "native.zip").read_bytes()
              for p in ("z300-aaa-one", "z300-bbb-two")}
    adb.commands.clear()

    assert run(roots, adb) == 0
    second = summary_of(capsys)
    assert (second["new"], second["archived"], second["skipped"], second["pending"]) == (0, 0, 0, 0)
    assert adb.cat_paths() == []  # nothing re-read
    for name, data in frozen.items():
        assert (roots.archive / name / "native.zip").read_bytes() == data
    assert first["archived"] == 2


# --- pending retry window --------------------------------------------------

def test_pending_young_retried_next_pass_and_old_moves_to_skipped(roots, capsys):
    now = 1_700_000_000.0
    young = now - 3600            # 1 h old, inside the 24 h window
    old = now - 48 * 3600        # 48 h old, past the window
    adb = SyncMockAdb([
        {"test_id": "young-pending", "document": make_doc(no_bundle=True, unix_time=young)},
        {"test_id": "old-pending", "document": make_doc(no_bundle=True, unix_time=old)},
    ])
    assert run(roots, adb, now=now) == 0
    first = summary_of(capsys)
    assert (first["pending"], first["skipped"]) == (1, 1)
    pending = json.loads((roots.state / "pending.json").read_text())
    skipped = json.loads((roots.state / "skipped.json").read_text())
    assert "young-pending" in pending and "old-pending" in skipped
    assert "old-pending" not in pending

    # Second pass: young is retried (still pending); old is not re-enumerated for read.
    adb.commands.clear()
    assert run(roots, adb, now=now) == 0
    second = summary_of(capsys)
    assert second["pending"] == 1 and second["new"] == 1  # only young is still "new"
    assert "young-pending" in json.loads((roots.state / "pending.json").read_text())
    # The permanently-skipped old test issued no revision read this pass.
    assert not any("old-pending" in c for c in adb.commands)
    assert any("young-pending" in c for c in adb.commands)


def test_averaged_only_test_is_skipped_with_reason(roots, capsys):
    adb = SyncMockAdb([
        {"test_id": "avg-only", "document": make_doc(only_avg=True), "blob": bundle(2)},
    ])
    assert run(roots, adb) == 0
    result = summary_of(capsys)
    assert (result["new"], result["archived"], result["skipped"]) == (1, 0, 1)
    assert adb.cat_paths() == []  # refused before reading any blob
    skipped = json.loads((roots.state / "skipped.json").read_text())
    assert "individual stored shots" in skipped["avg-only"]["reason"]
    assert not (roots.archive / "z300-avg-only").exists()


def test_no_stored_configuration_is_skipped(roots, capsys):
    adb = SyncMockAdb([
        {"test_id": "no-cfg", "document": make_doc(no_config=True), "blob": bundle(2)},
    ])
    assert run(roots, adb) == 0
    result = summary_of(capsys)
    assert result["skipped"] == 1 and result["archived"] == 0
    skipped = json.loads((roots.state / "skipped.json").read_text())
    assert skipped["no-cfg"]["reason"] == "no stored configuration"


# --- id rules --------------------------------------------------------------

def test_ids_failing_pantheum_rule_are_skipped(roots, capsys):
    adb = SyncMockAdb([
        {"test_id": "under_score", "document": make_doc(), "blob": bundle(2)},   # underscore
        {"test_id": "UpperCase", "document": make_doc(), "blob": bundle(2)},     # uppercase
        {"test_id": "ok", "document": make_doc(), "blob": bundle(2)},            # too short: z300-ok len 7
    ])
    assert run(roots, adb) == 0
    result = summary_of(capsys)
    assert result["archived"] == 0 and result["skipped"] == 3
    assert adb.cat_paths() == []  # never read a bundle for an unsafe run id
    skipped = json.loads((roots.state / "skipped.json").read_text())
    assert set(skipped) == {"under_score", "UpperCase", "ok"}
    for entry in skipped.values():
        assert entry["reason"] == "run id violates pantheum rule"


def test_run_id_helper_enforces_both_rules():
    assert sync.run_id_for("field-run-01") == ("z300-field-run-01", None)
    assert sync.run_id_for("bad id")[0] is None       # producer rule (space)
    assert sync.run_id_for("Upper")[0] is None        # pantheum rule (uppercase)
    assert sync.run_id_for("x" * 60)[0] is None        # pantheum length cap


# --- limit / newest first --------------------------------------------------

def test_limit_respected_newest_first(roots, capsys):
    adb = SyncMockAdb([
        {"test_id": "seq-one", "document": make_doc(), "blob": bundle(2), "sequence": 10},
        {"test_id": "seq-two", "document": make_doc(), "blob": bundle(2), "sequence": 20},
        {"test_id": "seq-three", "document": make_doc(), "blob": bundle(2), "sequence": 30},
    ])
    # list-limit 2 caps each window; --limit 2 caps archive attempts.
    assert run(roots, adb, "--list-limit", "2", "--limit", "2") == 0
    result = summary_of(capsys)
    assert (result["listed_fresh"], result["listed_backfill"], result["archived"]) == (2, 1, 2)
    assert (roots.archive / "z300-seq-three").is_dir()   # newest two archived first
    assert (roots.archive / "z300-seq-two").is_dir()
    assert not (roots.archive / "z300-seq-one").exists()  # backlog deferred to a later pass


# --- dry-run ---------------------------------------------------------------

def test_dry_run_reads_only_the_listing_and_writes_nothing(roots, capsys):
    adb = SyncMockAdb([
        {"test_id": "aaa-one", "document": make_doc(), "blob": bundle(2)},
    ])
    assert run(roots, adb, "--dry-run") == 0
    captured = capsys.readouterr()
    assert captured.out == ""  # no summary on stdout
    plan = json.loads(captured.err)
    assert plan["dry_run"] is True and plan["listed_fresh"] == 1
    assert plan["plan_fresh"][0]["action"] == "would_attempt"
    assert plan["plan_fresh"][0]["run_id"] == "z300-aaa-one"
    # Only metadata listings (fresh + backfill probe); never a blob read.
    assert adb.cat_paths() == [] and all(c.startswith("test ") for c in adb.commands)
    assert not roots.archive.exists() and not roots.state.exists()


# --- unreachable / usage ---------------------------------------------------

def test_unreachable_instrument_exits_3_with_empty_summary(roots, capsys):
    class DeadAdb:
        commands = []
        def read(self, command, limit):
            raise producer.PendingData("instrument read unavailable or timed out")
    assert run(roots, DeadAdb()) == 3
    result = summary_of(capsys)
    assert result["listed"] == 0 and result["archived"] == 0


@pytest.mark.parametrize("extra", [["--limit", "0"], ["--list-limit", "0"],
                                   ["--list-limit", "1001"], ["--retry-window-hours", "-1"],
                                   ["--min-shots", "0"]])
def test_usage_errors_exit_2(roots, extra, capsys):
    adb = SyncMockAdb([{"test_id": "aaa-one", "document": make_doc(), "blob": bundle(2)}])
    assert run(roots, adb, *extra) == 2


# --- backfill cursor: drain the historical backlog, new tests first ---------

def five_tests():
    return [{"test_id": f"run-0{n}", "document": make_doc(), "blob": bundle(2), "sequence": n}
            for n in range(1, 6)]


def test_backlog_drains_two_two_one_then_backfill_complete(roots, capsys):
    adb = SyncMockAdb(five_tests())
    archived, complete = [], []
    for _ in range(4):
        assert run(roots, adb, "--list-limit", "2", "--limit", "2") == 0
        summary = summary_of(capsys)
        archived.append(summary["archived"])
        complete.append(summary["backfill_complete"])
    assert archived == [2, 2, 1, 0]            # newest window drains 2, then backfill 2, 1
    assert complete == [False, False, False, True]
    assert {p.name for p in roots.archive.iterdir() if p.is_dir()} == {
        f"z300-run-0{n}" for n in range(1, 6)}


def test_new_test_is_archived_before_older_backlog(roots, capsys):
    adb = SyncMockAdb(five_tests())
    assert run(roots, adb, "--list-limit", "2", "--limit", "2") == 0  # archives run-05, run-04
    summary_of(capsys)
    adb.add_test({"test_id": "run-06", "document": make_doc(), "blob": bundle(2), "sequence": 6})
    assert run(roots, adb, "--list-limit", "2", "--limit", "2") == 0
    summary = summary_of(capsys)
    assert summary["archived"] == 2
    assert (roots.archive / "z300-run-06").is_dir()      # newer test archived first
    assert not (roots.archive / "z300-run-02").exists()  # older backlog still waiting


def test_pending_backfill_test_holds_the_cursor(roots, capsys):
    now = 1_700_000_000.0
    young = now - 3600  # inside the retry window: pending, not skipped
    adb = SyncMockAdb([
        {"test_id": "top-05", "document": make_doc(), "blob": bundle(2), "sequence": 5},
        {"test_id": "top-04", "document": make_doc(), "blob": bundle(2), "sequence": 4},
        {"test_id": "hold-03", "document": make_doc(no_bundle=True, unix_time=young), "sequence": 3},
        {"test_id": "done-02", "document": make_doc(), "blob": bundle(2), "sequence": 2},
    ])
    for _ in range(2):
        assert run(roots, adb, "--list-limit", "2", "--limit", "5", now=now) == 0
        summary = summary_of(capsys)
        assert summary["backfill_cursor"] == 4  # never steps past the pending seq-3
        assert summary["backfill_complete"] is False
    assert (roots.archive / "z300-done-02").is_dir()  # resolvable neighbour still archived
    assert "hold-03" in json.loads((roots.state / "pending.json").read_text())


def test_rescan_rewalks_the_backfill(roots, capsys):
    adb = SyncMockAdb(five_tests())
    for _ in range(4):
        assert run(roots, adb, "--list-limit", "2", "--limit", "2") == 0
        last = summary_of(capsys)
    assert last["backfill_complete"] is True
    # A finished walk issues no second query.
    assert run(roots, adb, "--list-limit", "2", "--limit", "2") == 0
    normal = summary_of(capsys)
    assert normal["listed_backfill"] == 0 and normal["backfill_complete"] is True
    # --rescan resets the cursor and re-walks (everything already archived).
    assert run(roots, adb, "--list-limit", "2", "--limit", "2", "--rescan") == 0
    rescan = summary_of(capsys)
    assert rescan["listed_backfill"] > 0 and rescan["archived"] == 0


def test_legacy_bundle_test_is_archived(roots, capsys):
    digest = "ab" * 20
    adb = SyncMockAdb([
        {"test_id": "legacy-run", "document": make_doc(shots=2, legacy=digest),
         "blob": legacy_bundle()},
    ])
    assert run(roots, adb) == 0
    result = summary_of(capsys)
    assert result["archived"] == 1
    assert (roots.archive / "z300-legacy-run" / "native.zip").is_file()
