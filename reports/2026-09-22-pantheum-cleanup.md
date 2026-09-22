# Pantheum branch and worktree cleanup — 2026-09-22

User requested merge all Pantheum branches into main and remove extra worktree
folders, preserving the primary checkout and other repositories.

Local branches: main only. Remote branch heads queried directly from origin:
main only. Both main refs point to be7b3202c4ae6a191ce8bcdcb5fec2f7fbe8a60b.
No merge or push was necessary. Existing uncommitted alibz and RamanLab edits
in the primary checkout were left untouched.

The only other registered worktree was a missing detached temporary baseline:
/private/tmp/claude-501/-Users-mwhittaker-Projects-github-pantheum-I/71c7b443-80c1-4a58-b753-81c1a6a844aa/scratchpad/baseline
Its commit 6de6524c4a83e6e9eaf8f0d6d2e03468803a59bd is an ancestor of main.
Git worktree prune --dry-run identified only that stale registration;
git worktree prune --verbose removed it. Postcheck lists only the main checkout.

The six additional Pantheum-named folders under Projects/github are ordinary
source backups, not Git repositories or worktrees. Main independently verified
agent inventory: 52 regular files, 1,967,619 bytes, no symlinks or .git markers.
31 file snapshots have no matching blob reachable from main. These may be
intermediate versions whose changes were later committed; absent exact blobs
do not imply unmerged functionality. No historical source files were overlaid
onto the current working tree.

Inventory: reports/2026-09-22-pantheum-folder-audit.md.
File hashes/history coverage: provenance/pantheum-backup-audit-20260922.json.
Prepared exact move plan: provenance/pantheum-folder-move-plan-20260922.json.
Awaiting scope clarification before moving these backups outside /github.
No provider switch.
