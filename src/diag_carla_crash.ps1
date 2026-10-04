# diag_carla_crash.ps1 -- one controlled CARLA collection trial, for bisecting the
# RenderThread crash that kills Town02 / Town05 on the first tick.
#
# Evidence this was written to chase (2026-07-30):
#   Town10HD/04/06/07/01  crash on TaskGraphThreadHP after 1499-7428 s, i.e. AFTER
#                         the run finished -> 2500-2700 frames collected.
#   Town02 / Town05       crash on RenderThread 1/2 after 38-107 s, i.e. on the
#                         first tick -> 0-36 frames.
# So the failure is render-side and town-specific, not load_world (Town04/06/07/01
# all collected fine while the collector did its own load_world, before prewarm
# even existed).
#
# One trial = kill server / start server with -ServerArgs / wait ready / collect
# -Frames frames into a scratch dir / report frames + which crash dump appeared.
#
#   powershell -File diag_carla_crash.ps1 -Town Town02 -Tag base -Frames 60
#   powershell -File diag_carla_crash.ps1 -Town Town02 -Tag epic -Frames 60 `
#              -ServerArgs '-quality-level=Epic'
#
# Never writes into data\v4: the scratch dir is deleted at the start of each trial.

param(
    [string]$Town       = 'Town02',
    [string]$Tag        = 'base',
    [int]$Frames        = 60,
    [string[]]$ServerArgs = @('-quality-level=Low'),
    [string[]]$ExtraCollect = @(),
    [int]$CamW          = 1280,
    [int]$CamH          = 720,
    [int]$Vehicles      = 40,
    [int]$Walkers       = 60,
    [switch]$NoFilter,
    [switch]$NoDepth,
    [int]$SpawnId       = -1,
    [switch]$ServerLog
)

$ErrorActionPreference = 'Stop'

$PY       = 'E:\miniforge3\envs\lss\python.exe'
$SRC      = 'E:\dualcam_project\src'
$LOGDIR   = 'E:\dualcam_project\logs\diag'
$CARLA    = 'G:\CARLA_0.9.16\CarlaUE4.exe'
$SCRATCH  = 'E:\dualcam_project\data\_diag_scratch'
$CRASHDIR = Join-Path $env:LOCALAPPDATA 'CarlaUE4\Saved\Crashes'
$SRVLOG   = Join-Path $env:LOCALAPPDATA 'CarlaUE4\Saved\Logs\CarlaUE4.log'

if (-not (Test-Path $LOGDIR)) { New-Item -ItemType Directory -Path $LOGDIR -Force | Out-Null }

function Stop-Carla {
    # CrashReportClient survives the server and holds the crash dialog open.
    $procs = Get-Process | Where-Object { $_.Name -like 'Carla*' -or $_.Name -match 'CrashReportClient' }
    if ($procs) {
        $procs | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 8
    }
}

# Crash dumps are the only server-side evidence in a Shipping build, so the trial
# is scored by "which dump is new", not by exit codes.
function Get-CrashSet {
    if (-not (Test-Path $CRASHDIR)) { return @() }
    return (Get-ChildItem $CRASHDIR -Directory | Select-Object -ExpandProperty Name)
}

$before = Get-CrashSet

if (Test-Path $SCRATCH) { Remove-Item $SCRATCH -Recurse -Force }

Write-Output "=== trial '$Tag' : $Town  frames=$Frames  server='$($ServerArgs -join ' ')' ==="

Stop-Carla
if (Test-Path $SRVLOG) { Remove-Item $SRVLOG -Force -ErrorAction SilentlyContinue }

$srvArgs = @('-carla-rpc-port=2000') + $ServerArgs
if ($ServerLog) { $srvArgs += '-log' }
Start-Process -FilePath $CARLA -ArgumentList $srvArgs

$rdy = Join-Path $LOGDIR "ready_$Tag.log"
Start-Process -FilePath $PY -ArgumentList '-u', "$SRC\wait_carla_ready.py", '--host', '127.0.0.1',
    '--port', '2000', '--streak', '3', '--timeout', '300' `
    -WorkingDirectory $SRC -Wait -NoNewWindow -RedirectStandardOutput $rdy -RedirectStandardError "$rdy.err"
if (-not (Select-String -Path $rdy -Pattern 'READY' -Quiet)) {
    Write-Output "RESULT $Tag : server never became ready"
    Stop-Carla
    exit 2
}

$a = @('-u', "$SRC\carla_dualcam_collect_gt.py",
       '--host', '127.0.0.1', '--port', '2000',
       '--town', $Town, '--out', $SCRATCH,
       '--frames', "$Frames",
       '--cam_w', "$CamW", '--cam_h', "$CamH",
       '--vehicles', "$Vehicles", '--walkers', "$Walkers",
       '--preview', (Join-Path $SCRATCH 'preview.mp4'))
if (-not $NoDepth)  { $a += '--depth' }
if (-not $NoFilter) { $a += @('--filter', '--near_ego', '150') }
if ($SpawnId -ge 0) { $a += @('--spawn_id', "$SpawnId") } else { $a += '--random_spawn' }
$a += $ExtraCollect

$log = Join-Path $LOGDIR "collect_$Tag.log"
$t0  = Get-Date
Start-Process -FilePath $PY -ArgumentList $a -WorkingDirectory $SRC -Wait -NoNewWindow `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

$saved = 0
$bev = Join-Path $SCRATCH 'bev_gt'
if (Test-Path $bev) { $saved = (Get-ChildItem $bev -Filter '*.png' -File).Count }

$alive = [bool](Get-Process | Where-Object { $_.Name -like 'Carla*' })

# Copy the server log aside before the next trial wipes it.
if ($ServerLog -and (Test-Path $SRVLOG)) {
    Copy-Item $SRVLOG (Join-Path $LOGDIR "server_$Tag.log") -Force
}

$new = Get-CrashSet | Where-Object { $before -notcontains $_ }

Write-Output "RESULT $Tag : saved=$saved / $Frames in ${secs}s  server_alive=$alive  new_crash_dumps=$($new.Count)"
foreach ($n in $new) {
    $xml = Join-Path $CRASHDIR "$n\CrashContext.runtime-xml"
    if (Test-Path $xml) {
        & $PY "$SRC\diag_crash_summary.py" $xml
    }
}

Stop-Carla
if (Test-Path $SCRATCH) { Remove-Item $SCRATCH -Recurse -Force -ErrorAction SilentlyContinue }
