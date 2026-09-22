#!/usr/bin/env bash
# Z300-0915 clock and launcher recovery after a power cycle.
#
# The analyzer's PCF8563 RTC has no working backup cell: every power-off resets
# the clock to 1970-01-02 (Android's earliest supported time). At boot the vendor
# launcher (com.sciaps.android.home, "LIBZ Home") sees the bad date, shows a
# "Loading..." dialog and opens Settings > Date & time. Setting the date there
# does NOT dismiss the dialog: the launcher stays idle behind it until it is
# restarted, and Geochem Pro cannot be opened from the handheld. The HTTP API
# (RemoteService) is unaffected, so data-API retrieval works throughout.
#
# Runs from the Mac via `ssh opal` (Windows host with the analyzer on USB ADB,
# root shell). Default mode is a read-only status; each change needs its flag.
#
#   scripts/z300-clock.sh                 status: Opal time, analyzer time, RTC,
#                                         uptime, skew, foreground window, dialog
#   scripts/z300-clock.sh --set           set the analyzer clock from Opal's clock
#                                         (local time, `date -s YYYYMMDD.HHMMSS`),
#                                         then broadcast TIME_SET; verifies skew
#   scripts/z300-clock.sh --restart-home  force-stop the launcher (Android relaunches
#                                         it); verifies the dialog is gone and saves
#                                         a screenshot
#   scripts/z300-clock.sh --screenshot    save a screenshot only
#   scripts/z300-clock.sh --reboot        warm reboot (RTC keeps time while powered;
#                                         laser disarms; needs --yes)
#
# Never fires, moves the stage, cancels a test, or touches RemoteService.
# --set and --restart-home refuse while Pantheum has queued/running acquisitions
# (checked on Moissanite) unless --force is given.
set -euo pipefail

ADB='C:\Users\Whittaker\AppData\Local\Temp\alibz-api-recovery-20260921\platform-tools\adb.exe'
ADB_PORT=5038
SERIAL=0123456789ABCDEF
HOME_PKG=com.sciaps.android.home
MAX_SKEW=120
OUT=${TMPDIR:-/tmp}

mode=status; yes=0; force=0
for a in "$@"; do
    case "$a" in
        --set) mode=set ;;
        --restart-home) mode=restart ;;
        --screenshot) mode=screenshot ;;
        --reboot) mode=reboot ;;
        --yes) yes=1 ;;
        --force) force=1 ;;
        -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

# --- Opal helpers -------------------------------------------------------------
ps_run() {  # run a PowerShell script on Opal; strip CLIXML noise
    local enc
    enc=$(python3 -c 'import sys,base64;print(base64.b64encode(sys.argv[1].encode("utf-16-le")).decode())' "$1")
    ssh -o BatchMode=yes -o ConnectTimeout=15 opal powershell.exe -NoProfile -NonInteractive -EncodedCommand "$enc" 2>&1 \
        | grep -v '^#< CLIXML' | sed -e 's/<Objs .*$//' | grep -v '^$' || true
}
adb_sh() {  # adb shell <cmd> on the analyzer. No double quotes in <cmd> (Windows argv eats them).
    local cmd=${1//\'/\'\'}
    ps_run "& '$ADB' -P $ADB_PORT -s $SERIAL shell '$cmd' 2>&1"
}
adb_raw() { # adb <args...>
    local args="" a
    for a in "$@"; do a=${a//\'/\'\'}; args="$args '$a'"; done
    ps_run "& '$ADB' -P $ADB_PORT -s $SERIAL $args 2>&1"
}
opal_now() { ps_run "Get-Date -Format yyyyMMdd.HHmmss"; }
opal_epoch() { ps_run "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()"; }  # not -UFormat %s: PS 5.1 makes it local-time based

analyzer_epoch() {  # system clock; toolbox `date -s` goes through /dev/alarm, which also writes the RTC
    adb_sh 'date +%s' | tr -d '\r' | grep -E '^[0-9]+$' || echo 0
}
home_windows() {  # 1 = launcher only, 2 = launcher + dialog (the stuck "Loading...")
    adb_sh 'dumpsys window windows' | grep -c "^ *Window #.* u0 $HOME_PKG/$HOME_PKG.HomeActivity}" || true
}
focus() { adb_sh 'dumpsys window windows' | grep mCurrentFocus | tr -d '\r' | sed 's/^ *//'; }

pantheum_active() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 moissanite python3 - <<'PY'
import os, sqlite3
db = sqlite3.connect(os.path.expanduser('~/.local/state/pantheum/alibz/alibz.sqlite'))
print(db.execute("SELECT COUNT(*) FROM acquisitions WHERE state IN ('queued','running','cancelling')").fetchone()[0])
PY
}
refuse_if_active() {
    local n
    n=$(pantheum_active || echo unknown)
    if [[ "$n" != "0" && $force -eq 0 ]]; then
        echo "refusing: Pantheum reports active acquisitions=$n (use --force to override)" >&2; exit 3
    fi
}

screenshot() {
    local ts f
    ts=$(date +%Y%m%dT%H%M%S); f="$OUT/z300-screen-$ts.png"
    adb_sh 'screencap -p /data/local/tmp/alibz-screen.png' >/dev/null
    adb_raw pull /data/local/tmp/alibz-screen.png 'C:\Users\Whittaker\AppData\Local\Temp\alibz-screen.png' >/dev/null
    scp -q -o BatchMode=yes 'opal:C:/Users/Whittaker/AppData/Local/Temp/alibz-screen.png' "$f"
    echo "screenshot: $f"
}

status() {
    local oe ae skew
    echo "opal_time:      $(ps_run 'Get-Date -Format o')"
    echo "analyzer_date:  $(adb_sh date | tr -d '\r')"
    echo "analyzer_uptime_s: $(adb_sh 'cat /proc/uptime' | tr -d '\r' | cut -d' ' -f1)"
    echo "rtc:            $(adb_sh 'cat /sys/class/rtc/rtc0/name /sys/class/rtc/rtc0/date /sys/class/rtc/rtc0/time' | tr -d '\r' | tr '\n' ' ')"
    oe=$(opal_epoch | tr -d '\r'); ae=$(analyzer_epoch); skew=$((ae - oe))
    echo "skew_s (analyzer - opal): $skew"
    echo "focus:          $(focus)"
    local n; n=$(home_windows)
    echo "home_windows:   $n $([[ "$n" -ge 2 ]] && echo '(launcher dialog present: stuck Loading...)' || echo '(no dialog)')"
    echo "date_prompts_since_boot: $(adb_sh 'logcat -d -b system -v time' | grep -c 'act=android.settings.DATE_SETTINGS' || true)"
    echo "pantheum_active_acquisitions: $(pantheum_active 2>/dev/null || echo unknown)"
    local abs=$(( skew < 0 ? -skew : skew ))
    if (( abs <= MAX_SKEW )); then echo "clock: OK"; else echo "clock: SKEWED by $skew s (run --set)"; fi
}

case $mode in
    status) status ;;
    screenshot) screenshot ;;
    set)
        refuse_if_active
        before=$(analyzer_epoch); oe=$(opal_epoch | tr -d '\r')
        echo "before: analyzer-opal skew ${before}-${oe} = $((before - oe)) s"
        now=$(opal_now | tr -d '\r')
        adb_sh "date -s $now" | tr -d '\r'
        adb_sh 'am broadcast -a android.intent.action.TIME_SET' | tr -d '\r' | head -2 || true
        after=$(analyzer_epoch); oe=$(opal_epoch | tr -d '\r'); skew=$((after - oe))
        echo "after:  analyzer-opal skew = $skew s"
        if (( skew < -MAX_SKEW || skew > MAX_SKEW )); then
            echo "clock still skewed by $skew s; check time zone (analyzer $(adb_sh 'getprop persist.sys.timezone' | tr -d '\r'))" >&2; exit 4
        fi
        echo "clock: OK"
        n=$(home_windows); [[ "$n" -ge 2 ]] && echo "launcher dialog still present; run --restart-home" || true
        ;;
    restart)
        refuse_if_active
        echo "before: home_windows=$(home_windows) focus: $(focus)"
        adb_sh "am force-stop $HOME_PKG" | tr -d '\r'
        sleep 6
        n=$(home_windows)
        echo "after:  home_windows=$n focus: $(focus)"
        screenshot
        if [[ "$n" -ge 2 ]]; then
            echo "dialog came back: the launcher still sees a bad clock or a missing service; run status" >&2; exit 5
        fi
        echo "launcher restarted; open Geochem Pro on the handheld and arm the laser (PIN) before firing"
        ;;
    reboot)
        [[ $yes -eq 1 ]] || { echo "--reboot needs --yes (warm reboot; laser disarms; clock survives because the RTC stays powered)" >&2; exit 2; }
        refuse_if_active
        echo "skew before reboot: $(( $(analyzer_epoch) - $(opal_epoch | tr -d '\r') )) s"
        adb_raw reboot | tr -d '\r' || true
        echo "rebooting; wait ~90 s, then run status"
        ;;
esac
