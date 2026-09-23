# Pantheum source-backup payload review — 2026-09-22

## Outcome and scope

No credential, private-key, binary-data, or private-runtime-configuration commit blockers were identified in the exact 52 files named by `provenance/pantheum-backup-audit-20260922.json`. All 52 originals were read under `/Users/mwhittaker/Projects/github/<directory>/<path>`, comprising 1,967,619 bytes across six dated backup folders. This is a bounded static archival review, not a deployment or runtime validation.

The review read `/Users/mwhittaker/Projects/github/pantheum-I/AGENTS.md`, particularly lines 53–55 (NVR credentials), 60–61 (tablet credentials), and 79–82 (credentials and TLS private keys excluded from Git). No source, backup, or Git metadata was edited. The only intentional file write is this report, beginning at `reports/2026-09-22-pantheum-backup-payload-review.md:1`. No provider switch occurred. No network or host `lrc` access occurred.

## Evidence

- SHA-256 was recomputed from the bytes of every listed original: **52/52 matched** the corresponding audit hash and all expected files were readable.
- UTF-8 decoding succeeded for **52/52** files. There were **0** NUL bytes or other control characters except ordinary tabs/newlines/carriage returns.
- File types: 15 Markdown, 22 Python, 1 JSON, 3 HTML, 3 CSS, 4 JavaScript, and 4 CommonJS files. No `.env`, credential store, key file, binary payload, or runtime database was present in the reviewed inventory.
- Private-key PEM header matches: **0**. Known AWS/GitHub/Slack/OpenAI/Google API token and JWT signature matches: **0**. URLs embedding a username/password pair: **0**.
- Broad credential/password/token/secret/API-key/authorization/cookie text scan: **126 line occurrences**, **38 distinct lines**, all triaged. These were documentation of authentication boundaries or credential storage rules; HTTP credential policy; camera-frame content hashes; and tests asserting that status responses omit secrets. Ordinary code names were not counted as credentials.
- Long opaque-looking runs: **42**, resolved as **17 comment separators** and **25 identifiers**. The entropy check covered **10,255 extracted string literals**. Its **2 unique mixed-case/digit high-entropy literal candidates** were dotted `unittest.mock.patch` targets, verified structurally as patch call arguments (4 occurrences); neither was a credential.
- The only JSON file is `pantheum-I-acquisition-backup-20260921T130028/config/alibz.example.json`. Its fields contain application settings, endpoint/path references, and motion/acquisition example settings. There are no credential-value fields or embedded credentials. `sync.rclone_config` refers to a runtime file path; it does not contain the credential file. The audit marks this example's content as already present in main history. It remains historical source configuration, not a captured private runtime configuration file.

## Commands and method

All review reads were scoped to the audit JSON, the specified Pantheum AGENTS.md, and the 52 exact payload files. Local `python3` inline scripts used `json`, `pathlib`, `hashlib`, `re`, `collections`, `ast`, and `math`; they decoded each file, checked audit SHA-256 hashes and control bytes, scanned token/key/credential patterns, extracted literals, and classified candidates. Reports to the terminal included only paths/line numbers, aggregate counts, and redacted candidate context. No candidate secret value was printed.

The substantive checks were equivalent to the following per-file operations; the inventory below fixes the exact input set:

```python
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == audit_file['sha256']
text = raw.decode('utf-8')
assert not any(ord(c) < 32 and c not in '\t\n\r' for c in text)
# Scan PEM private-key headers; provider token/JWT forms; credential URLs;
# credential-related words; opaque runs of >=64 characters.
# Parse Python string constants with ast; extract other quoted strings.
# Triage no-whitespace literals of >=20 characters with entropy >=4.3,
# mixed uppercase/lowercase/digits, excluding slash-containing paths.
# Resolve remaining candidates with redacted local context and AST call shape.
```

## Unrelated project-test attempt

The implementation-agent template requested project tests, so an alibz baseline was started despite this being a read-only archival review. The coordinating agent explicitly clarified that runtime tests are not an archive-validation requirement and requested stopping the owned test session and running no further tests. No post-change test run was performed; no runtime source change was made.

- Command: `PYTHONPATH=src python3 -m pytest tests/ -q`. Result: **30 collection errors, 0 executed tests**, exit **2**, 11.26 seconds. The reported collection errors include `ModuleNotFoundError: No module named 'scipy'`. Log: `/tmp/pantheum-payload-review-tests-before-base.log`.
- Command: `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`. Result at explicit interruption of this agent's own session: **69 passed, 22 subtests passed**, incomplete suite, exit **130**, 60.59 seconds. Log: `/tmp/pantheum-payload-review-tests-before-pmg.log`.
- These outcomes neither validate nor invalidate the archival payload. The archive-specific evidence is the file/hash/text/credential review above.

## Limits

This review does not mathematically prove that arbitrary text cannot encode a secret. It identified no secret indicators or unresolved candidates under the checks described. It does not verify the surrounding repository, ignored files, remote systems, Git reachability, the final archive copy, or deletion of originals. The coordinating agent owns independent verification of preservation and any later deletion. No live configuration was installed or exercised.

## Exact reviewed inventory

Every entry below passed SHA-256 equality, UTF-8 decoding, control-byte checks, and the scoped credential/private-configuration review. Paths are relative to `/Users/mwhittaker/Projects/github`.

| Original file | Bytes |
|---|---:|
| `pantheum-fire-fix-source-backup-20260921T153917/DECISIONS.md` | 36781 |
| `pantheum-fire-fix-source-backup-20260921T153917/pantheum/alibz/acquire.py` | 34764 |
| `pantheum-fire-fix-source-backup-20260921T153917/pantheum/alibz/z300.py` | 18269 |
| `pantheum-fire-fix-source-backup-20260921T153917/tests/test_alibz_acquire.py` | 27130 |
| `pantheum-fire-fix-source-backup-20260921T153917/tests/test_alibz_checkouts.py` | 24764 |
| `pantheum-fire-fix-source-backup-20260921T153917/docs/alibz-architecture.md` | 28572 |
| `pantheum-I-acquisition-backup-20260921T130028/README.md` | 16681 |
| `pantheum-I-acquisition-backup-20260921T130028/DECISIONS.md` | 33671 |
| `pantheum-I-acquisition-backup-20260921T130028/pantheum/alibz/service.py` | 21249 |
| `pantheum-I-acquisition-backup-20260921T130028/pantheum/alibz/motion.py` | 22666 |
| `pantheum-I-acquisition-backup-20260921T130028/pantheum/alibz/__main__.py` | 22972 |
| `pantheum-I-acquisition-backup-20260921T130028/config/alibz.example.json` | 3172 |
| `pantheum-I-acquisition-backup-20260921T130028/web/alibz/index.html` | 22428 |
| `pantheum-I-acquisition-backup-20260921T130028/web/alibz/styles.css` | 22464 |
| `pantheum-I-acquisition-backup-20260921T130028/web/alibz/app.js` | 159901 |
| `pantheum-I-acquisition-backup-20260921T130028/deploy/alibz/enable-gantry.py` | 4303 |
| `pantheum-I-acquisition-backup-20260921T130028/deploy/alibz/README.md` | 16942 |
| `pantheum-I-acquisition-backup-20260921T130028/tests/test_alibz_motion.py` | 22686 |
| `pantheum-I-acquisition-backup-20260921T130028/tests/test_alibz_ui.cjs` | 24849 |
| `pantheum-I-acquisition-backup-20260921T130028/docs/alibz-architecture.md` | 25915 |
| `pantheum-I-acquisition-backup-20260921T130028/reports/STATUS.md` | 40607 |
| `pantheum-I-acquisition-backup-20260921T141426/README.md` | 17109 |
| `pantheum-I-acquisition-backup-20260921T141426/DECISIONS.md` | 35165 |
| `pantheum-I-acquisition-backup-20260921T141426/pantheum/alibz/reservation.py` | 8897 |
| `pantheum-I-acquisition-backup-20260921T141426/pantheum/alibz/acquire.py` | 24966 |
| `pantheum-I-acquisition-backup-20260921T141426/pantheum/alibz/optimization.py` | 29298 |
| `pantheum-I-acquisition-backup-20260921T141426/pantheum/alibz/motion.py` | 35013 |
| `pantheum-I-acquisition-backup-20260921T141426/pantheum/alibz/__main__.py` | 24448 |
| `pantheum-I-acquisition-backup-20260921T141426/web/alibz/index.html` | 25490 |
| `pantheum-I-acquisition-backup-20260921T141426/web/alibz/styles.css` | 24317 |
| `pantheum-I-acquisition-backup-20260921T141426/web/alibz/app.js` | 185069 |
| `pantheum-I-acquisition-backup-20260921T141426/deploy/alibz/README.md` | 18481 |
| `pantheum-I-acquisition-backup-20260921T141426/tests/test_alibz.py` | 28288 |
| `pantheum-I-acquisition-backup-20260921T141426/tests/test_alibz_acquire.py` | 22406 |
| `pantheum-I-acquisition-backup-20260921T141426/tests/test_alibz_ui.cjs` | 33352 |
| `pantheum-I-acquisition-backup-20260921T141426/tests/test_alibz_reservation.py` | 10990 |
| `pantheum-I-acquisition-backup-20260921T141426/docs/alibz-architecture.md` | 26611 |
| `pantheum-I-acquisition-backup-20260921T142747/DECISIONS.md` | 35934 |
| `pantheum-I-acquisition-backup-20260921T142747/pantheum/alibz/optimization.py` | 29622 |
| `pantheum-I-acquisition-backup-20260921T142747/web/alibz/app.js` | 190292 |
| `pantheum-I-acquisition-backup-20260921T142747/tests/test_alibz_ui.cjs` | 46049 |
| `pantheum-I-acquisition-backup-20260921T142747/tests/test_alibz_optimization.py` | 17294 |
| `pantheum-I-acquisition-backup-20260921T142747/docs/acquisition-optimization.md` | 5448 |
| `pantheum-panel-errors-source-backup-20260921T165416/web/alibz/index.html` | 28992 |
| `pantheum-panel-errors-source-backup-20260921T165416/web/alibz/styles.css` | 24577 |
| `pantheum-panel-errors-source-backup-20260921T165416/web/alibz/app.js` | 193746 |
| `pantheum-panel-errors-source-backup-20260921T165416/tests/test_alibz_ui.cjs` | 51069 |
| `pantheum-query-fix-source-backup-20260921T162743/DECISIONS.md` | 37542 |
| `pantheum-query-fix-source-backup-20260921T162743/pantheum/alibz/acquire.py` | 39973 |
| `pantheum-query-fix-source-backup-20260921T162743/pantheum/alibz/z300.py` | 19259 |
| `pantheum-query-fix-source-backup-20260921T162743/tests/test_alibz_acquire.py` | 36408 |
| `pantheum-query-fix-source-backup-20260921T162743/docs/alibz-architecture.md` | 30728 |
