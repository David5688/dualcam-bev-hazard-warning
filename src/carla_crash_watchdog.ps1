# carla_crash_watchdog.ps1 -- kills a crashed CARLA server the moment it dies, so
# the "The UE4-CarlaUE4 Game has crashed and will close / Fatal error!" modal never
# sits on screen waiting for someone to click 確定.
#
# Why this is needed even though collect_v4_driver.ps1 already restarts the server
# every segment: on a crash the engine puts up a modal dialog and the process stays
# alive behind it.  The driver only kills it on its NEXT loop iteration, which is
# after the collector has finished waiting out --teardown_timeout -- up to ~40 s of
# a dialog sitting in the user's face, every single dead segment.  Killing it right
# away also shortens each dead segment, because the collector then gets a refused
# connection instead of waiting for the timeout.
#
# Trigger: a new directory under %LOCALAPPDATA%\CarlaUE4\Saved\Crashes.  UE4 creates
# it at crash time, so it is an exact "this server is dead" signal -- no guessing
# from window titles.  A crash dump means collection is over for that segment either
# way (mid-run crash = segment lost, teardown crash = every frame and the manifest
# are already on disk, see 資料收集規劃.md §13 結論二), so killing on this signal can
# never interrupt a healthy run.
#
# Guard: only servers that were already running when the dump appeared are killed,
# so a freshly started server can never be shot by a late-arriving detection.
#
# Side effect worth knowing: killing the engine early truncates UE4Minidump.dmp.
# The CrashContext.runtime-xml (thread, error message, callstack -- everything
# diag_crash_summary.py reads) is written first and stays intact.
#
#   powershell -File carla_crash_watchdog.ps1              # runs until stopped
#   powershell -File carla_crash_watchdog.ps1 -Once        # single sweep, for testing

param(
    [int]$PollSeconds = 2,
    [switch]$Once
)

$CRASHDIR = Join-Path $env:LOCALAPPDATA 'CarlaUE4\Saved\Crashes'
$LOG      = 'E:\dualcam_project\logs\crash_watchdog.log'

function Write-Log([string]$msg) {
    $line = "{0} {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $LOG -Value $line -Encoding utf8
}

function Get-CrashDirs {
    if (-not (Test-Path $CRASHDIR)) { return @{} }
    $h = @{}
    foreach ($d in (Get-ChildItem $CRASHDIR -Directory -ErrorAction SilentlyContinue)) {
        $h[$d.Name] = $d.CreationTime
    }
    return $h
}

# The crash reporter itself is never wanted: it is a pure UI process.
function Kill-CrashReporter {
    $procs = Get-Process | Where-Object { $_.Name -match 'CrashReportClient' }
    foreach ($p in $procs) {
        try {
            $p | Stop-Process -Force -ErrorAction Stop
            Write-Log "killed crash reporter pid $($p.Id)"
        } catch { }
    }
}

function Kill-DeadServer([datetime]$dumpTime) {
    foreach ($p in (Get-Process | Where-Object { $_.Name -like 'Carla*' })) {
        try {
            # Anything started after the dump is a new server, not the corpse.
            if ($p.StartTime -lt $dumpTime) {
                $age = [int]((Get-Date) - $p.StartTime).TotalSeconds
                $p | Stop-Process -Force -ErrorAction Stop
                Write-Log "killed crashed server $($p.Name) pid $($p.Id) (was up ${age}s)"
            }
        } catch { }
    }
}

$seen = Get-CrashDirs
Write-Log "watchdog started (poll ${PollSeconds}s, $($seen.Count) existing crash dumps ignored)"

while ($true) {
    Kill-CrashReporter

    $now = Get-CrashDirs
    foreach ($name in $now.Keys) {
        if (-not $seen.ContainsKey($name)) {
            Write-Log "new crash dump $name"
            Kill-DeadServer $now[$name]
            $seen[$name] = $now[$name]
        }
    }

    if ($Once) { break }
    Start-Sleep -Seconds $PollSeconds
}
