# Opal camera reliability audit — 2026-09-22

Scope: read-only review of the deployed-task evidence and the Pantheum-I
installer/source. The live `OpalCamera` task and live camera process were not
changed or restarted by this audit.

## Finding

The installed task action is a PowerShell wrapper around Python:

```powershell
powershell.exe ... -Command "& '...\\python.exe' -B '...\\opal-camera.py' ..."
```

Windows PowerShell 5.1 documentation says that with `-Command`, the process
exit code is `1` when the last command sets `$?` false, and that `exit
$LASTEXITCODE` is needed to preserve the native command's exact code. The
existing task recorded `3221225786` (`0xC000013A`) directly, so there is no
evidence here that the wrapper swallowed failure status or prevented the
configured `RestartCount 3` / `RestartInterval PT5M` policy from applying.
The status proves an interruption/control termination, but does not identify
who or what caused it. Source inspection cannot establish that root cause.
See [about_Automatic_Variables](https://learn.microsoft.com/en-us/powershell/module/Microsoft.PowerShell.Core/about/about_automatic_variables?view=powershell-5.1)
and [about_PowerShell_exe](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_powershell_exe?view=powershell-5.1).

## Staged patch

An exploratory one-line patch was staged during the audit, but is withdrawn:

`/private/tmp/pantheum-camera-reliability-20260922/install-opal-camera.ps1`

Because the live task already reported the exact native status, preserving that
status is not materially useful for the observed failure. The staged file was
deleted; no installer change is recommended.

No source or installer diff remains.

## Verification

- Microsoft documentation review confirms `-Command` returns `1` for a false final command result and recommends `exit $LASTEXITCODE` only when preserving the exact native code is required.
- The exploratory staged file was deleted after that review; no source or installer diff remains.
- No live install, task mutation, camera restart, or source-tree edit was performed.
