# Deploy wrapper: 2026-09-22 Optimize Acquisition / composition change

## What was written

1. `/Users/mwhittaker/Projects/github/alibz/scripts/deploy-alibz-fire-fix.py`
   extended minimally (diff, not a rewrite):
   - Lines 38-45: manifest validation loop now treats `before[name] == None`
     as "must not exist live" (raises `Live file already exists: <name>` if
     it does); otherwise unchanged (`digest(live) != before` -> `Live source
     changed`).
   - Lines 67-73: backup loop now `continue`s for any live file that does
     not exist (new files have nothing to back up).
   - Line 78: added `target.parent.mkdir(parents=True, exist_ok=True)`
     before the atomic install-write, needed the first time a file is
     installed into a brand-new path (e.g. `pantheum/alibz/references/`
     already exists on the live tree so this is a no-op there, but keeps
     the install path safe for any future new subdirectory).
   - Lines 86-93 (except-block rollback): now iterates `manifest['before']`
     -- for files whose `before` was `None`, `unlink()`s them if the failed
     install created them; for all other files, restores from backup only
     if a backup copy exists. This makes a failed `--apply` leave the tree
     exactly as it was, including removing partially-installed new files.
   - Every existing check (manifest path safety, bundle-hash check, config
     hash check, reservation lock, sqlite active-work/job/hardware queries,
     backup-then-stop-services-then-install-then-restart sequencing, hash
     verification after install) is untouched; for a manifest with no
     `null` befores the code path and behavior are byte-identical to before.

2. `/Users/mwhittaker/Projects/github/alibz/scripts/deploy-alibz-optimize-acquisition.sh`
   (new, executable, bash 3.2 compatible -- no associative arrays or
   `mapfile`, since `/bin/bash` on this Mac is GNU bash 3.2.57 and there is
   no Homebrew bash on PATH). Same shape as `deploy-alibz-rate-10hz.sh`:
   default dry run, `--apply` installs. Manifest is built at run time:
   - `REF_FILES` enumerated via `ls pantheum/alibz/references/*-v1.json |
     grep -v fe-v1.json | sort` in `$PANTHEUM_SRC` -- asserts exactly 82.
   - `after` hash = local `shasum -a 256` of each of the 87 files (4
     modified + `composition.py` + 82 references).
   - `before` for the 4 modified files = live `sha256sum` read over ssh
     from `~/pantheum-I`, but the script refuses unless that live hash
     equals `git show <commit>:<path>` for one of `6f7c654`, `97433bb`,
     `e4e79ee` (computed locally in `$PANTHEUM_SRC`); it prints which
     commit matched, per file.
   - `before` for the 83 new files (`composition.py` + 82 references) =
     `null`, after first verifying over ssh (`[ -e file ]` loop in
     `~/pantheum-I`) that none of them exist live.
   - `config_sha256` of `~/.config/pantheum/alibz.json`, same as the
     existing wrappers.
   - Remote bundle dir: `pantheum-optimize-acquisition-bundle-20260922`.
   - Header comment states what it deploys, that it supersedes
     `scripts/deploy-alibz-verify-min-shots.sh` (live `optimization.py` is
     at `e4e79ee`; this bundle also carries the `97433bb` change), and
     warns that the pending `pantheum-I-stage-6f7c654/` rsync described in
     `pantheum-I/reports/2026-09-22-alibz-field-deploy.md` would overwrite
     the 4 modified files back to HEAD (6f7c654) content if run after this
     script, so it must run first, or the composition change must be
     committed to HEAD before running both.
   - After `--apply`: prints `sha256sum` of the 4 modified files +
     `composition.py`, the count of `references/*-v1.json` files live, and
     `systemctl --user is-active` of both services (same pattern as
     `deploy-alibz-rate-10hz.sh`). Neither `deploy-alibz-rate-10hz.sh` nor
     `deploy-alibz-verify-min-shots.sh` actually contains a status-endpoint
     GET (checked: no `curl`/`GET`/socket reference in either file) --
     that pattern only exists elsewhere in this repo, in
     `scripts/check-optimization-readiness.py` and in
     `pantheum-I/reports/2026-09-22-alibz-field-deploy.md`'s operator step
     1, both of which GET over the alibz gateway unix socket at
     `~/.local/state/pantheum/alibz-gateway/alibz.sock`. I copied that
     pattern instead: `curl -s --unix-socket
     ~/.local/state/pantheum/alibz-gateway/alibz.sock
     http://alibz/api/optimization` (the optimization status endpoint used
     by `check-optimization-readiness.py`). Flagging this as a deviation
     from "copy that pattern if present" since it was not present in the
     two named files.

## Verification run

- `bash -n scripts/deploy-alibz-optimize-acquisition.sh` -> no output, exit 0.
- `python3 -m py_compile scripts/deploy-alibz-fire-fix.py` -> exit 0.
- `ssh -o BatchMode=yes moissanite 'echo SSH_OK && hostname'` -> `SSH_OK` /
  `mwhittaker-u10`. No permission-classifier refusal on ssh, scp, or the
  remote dry run at any point; nothing was worked around.
- Dry run (`bash scripts/deploy-alibz-optimize-acquisition.sh`, no
  `--apply`) run twice:
  - **Run 1**: succeeded end to end (verbatim, trimmed):
    ```
    == verifying local sources exist ==
    == reading live hashes for modified files over ssh ==
    == checking new files do not already exist live ==
    == matching each live modified-file hash against known-good commits 6f7c654 97433bb e4e79ee ==
      live pantheum/alibz/optimization.py matches commit e4e79ee
      live pantheum/alibz/optimization_metrics.py matches commit 6f7c654
      live web/alibz/app.js matches commit 97433bb
      live web/alibz/index.html matches commit 6f7c654
    == reading private config hash over ssh ==
    {"apply": false, "files": [...87 entries...], "active_work": 0, "configuration": "unchanged"}
    dry run only; rerun with --apply to install and restart the alibz services
    ```
    Manifest file count confirmed programmatically: `file_count 87`
    (4 modified + `composition.py` + 82 references).
  - **Run 2** (immediately after): the local pre-checks and the bundle
    scp all succeeded again, but the *remote* `deploy.py` dry run failed
    with `RuntimeError: Active work or unresolved hardware; refusing
    deployment` -- this is the existing sqlite active-work/job/hardware
    guard in `deploy-alibz-fire-fix.py` firing because moissanite had a
    real acquisition/hardware operation in flight at that moment (state
    flickered; not caused by this script, and not a permission-classifier
    refusal). A third run moments later succeeded again with
    `"active_work": 0`. No workaround was applied; this is the intended
    safety behavior of the unmodified guard.
  - `--apply` was never passed; no live file was touched, no service was
    stopped or restarted.

## Known-commit matches (live tree, as of this run)

| File | Matches commit |
|---|---|
| pantheum/alibz/optimization.py | e4e79ee |
| pantheum/alibz/optimization_metrics.py | 6f7c654 |
| web/alibz/app.js | 97433bb |
| web/alibz/index.html | 6f7c654 |

This matches the brief's stated live state (`optimization.py` at `e4e79ee`).

## Unverified / not attempted

- `--apply` was never run (out of scope per the brief); installed hashes,
  live reference-file count, service status, and the `/api/optimization`
  GET after a real install are therefore unverified against a live
  install -- only the dry-run path and the deploy.py unit-level control
  flow were exercised.
- The `/api/optimization` GET pattern is a best-effort substitution (see
  above) since the two named wrapper scripts do not contain a
  status-endpoint call; if a more specific "the" status endpoint was
  intended, that should be confirmed before relying on the `--apply`
  branch's output.
- Did not test the rollback path (a forced mid-install failure) live,
  since that would require `--apply` against production; the code change
  was verified by reading and by `py_compile` only.
