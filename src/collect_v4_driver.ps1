# collect_v4_driver.ps1 -- segmented collection driver for the v4 dataset re-collection.
#
# CARLA 0.9.16 reliably crashes while destroying the sensors at the end of a run
# (verified on a freshly rebooted machine, 2026-07-29), and it can also crash
# mid-run on the big maps.  Data is safe either way: every frame is written as it
# is captured and poses/manifest are flushed every 50 frames.  So the strategy is
# to keep restarting the server and collecting into the SAME out directory with a
# fresh --prefix until the target frame count is reached.
#
# Every segment also pre-warms the map with prewarm_map.py before the collector
# runs, so the collector never calls load_world() itself -- see Prewarm-Map.
#
#   powershell -File collect_v4_driver.ps1 -Town Town10HD -Out ..\data\v4\train_town10hd `
#              -Target 2500 -Filter -NearEgo 150
#
# val/test towns must be one continuous run: pass -Single (no prefix, no retry).

param(
    [Parameter(Mandatory = $true)][string]$Town,
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][int]$Target,
    [switch]$Filter,
    [double]$NearEgo = 0,
    [switch]$Single,
    [string]$PrefixTag = 's',
    # Loose backstop only -- $MaxDead is the real give-up signal.  Town02's observed
    # yield per segment is wildly uneven (594, 72, 34, and plenty of 0), averaging
    # ~26 frames over the first four segments of the 2026-07-30 21:50 run, so filling
    # 2,500 can genuinely need ~70 segments.  At <=1 min for a dead segment
    # (--teardown_timeout) the old 40 stopped the grind long before it was actually
    # stuck.
    [int]$MaxSeg = 150,
    # How many consecutive empty segments to tolerate before giving up.
    #
    # Measured on Town02 over 13 attempts on 2026-07-30 (see 資料收集規劃.md §13.1):
    # 4 produced frames, 9 died on the first tick with 0 frames -> ~31% success, and
    # a surviving segment collects ~600 frames.  So a city needs ~4 good segments
    # out of ~13 attempts, and the give-up threshold only has to be high enough not
    # to fire on an unlucky streak: 0.69^8 = 5.1%, 0.69^12 = 1.2%.  12 it is -- a
    # dead segment now costs ~1 min rather than ~4.2 (--teardown_timeout), so the
    # extra patience is nearly free.
    [int]$MaxDead = 12,
    [int]$Vehicles = 40,
    [int]$Walkers = 60,
    [int]$CamW = 1280,
    [int]$CamH = 720
)

$ErrorActionPreference = 'Stop'

$PY      = 'E:\miniforge3\envs\lss\python.exe'
$SRC     = 'E:\dualcam_project\src'
$LOGDIR  = 'E:\dualcam_project\logs'
$CARLA   = 'G:\CARLA_0.9.16\CarlaUE4.exe'
$name    = Split-Path $Out -Leaf

if (-not (Test-Path $LOGDIR)) { New-Item -ItemType Directory -Path $LOGDIR | Out-Null }

function Stop-Carla {
    # CrashReportClient too: after a crash it survives the server and leaves a modal
    # "the game has crashed and will close" dialog sitting on the desktop.  See also
    # carla_crash_watchdog.ps1, which clears it within seconds instead of waiting for
    # the next segment to come around.
    $procs = Get-Process | Where-Object { $_.Name -like 'Carla*' -or $_.Name -match 'CrashReportClient' }
    if ($procs) {
        $procs | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 8
    }
}

function Start-Carla {
    Stop-Carla
    Start-Process -FilePath $CARLA -ArgumentList '-quality-level=Low', '-carla-rpc-port=2000'
    $rdy = Join-Path $LOGDIR "v4_ready_$name.log"
    Start-Process -FilePath $PY -ArgumentList '-u', "$SRC\wait_carla_ready.py", '--host', '127.0.0.1',
        '--port', '2000', '--streak', '3', '--timeout', '300' `
        -WorkingDirectory $SRC -Wait -NoNewWindow -RedirectStandardOutput $rdy -RedirectStandardError "$rdy.err"
    return (Select-String -Path $rdy -Pattern 'READY' -Quiet)
}

function Prewarm-Map {
    # Get the server onto the target map BEFORE the collector connects.
    #
    # Left to itself the collector calls load_world() (carla_dualcam_collect_gt.py
    # :1019), whose hard 60 s wait client.set_timeout() cannot extend -- and it then
    # spawns four sensors onto a map the engine may still be streaming in.  On
    # 2026-07-30 that combination killed the server in 3 of Town02's 4 segments:
    # 0 frames each, every log stopping right after "content filter ON".  Town10HD,
    # which is the server's own default map so no load happens at all, collected
    # 2,500 frames in a single segment.
    #
    # prewarm_map.py does the switch itself, swallows the timeout and polls until
    # the server reports the map; the collector then sees it is already there and
    # skips its own load (the skip branch at :1015).
    $prewarmLog = Join-Path $LOGDIR "v4_prewarm_${name}_${PrefixTag}${seg}.log"
    Start-Process -FilePath $PY -ArgumentList '-u', "$SRC\prewarm_map.py", '--host', '127.0.0.1',
        '--port', '2000', '--town', $Town, '--wait', '600' `
        -WorkingDirectory $SRC -Wait -NoNewWindow -RedirectStandardOutput $prewarmLog `
        -RedirectStandardError "$prewarmLog.err"
    return (Select-String -Path $prewarmLog -Pattern 'already on|server is on' -Quiet)
}

function Get-Saved {
    $d = Join-Path $Out 'bev_gt'
    if (-not (Test-Path $d)) { return 0 }
    return (Get-ChildItem $d -Filter '*.png' -File).Count
}

Write-Output "=== v4 collect: $Town -> $Out  target=$Target  filter=$Filter near_ego=$NearEgo single=$Single ==="

$seg      = 0
$deadRuns = 0

# Segments must start from DIFFERENT spawn points, or the autopilot retraces one
# route and the duplicate frames leak across split_contiguous (README 開頭那個
# 0.772 mIoU 的洩漏).  --random_spawn was the first fix, and it is not enough:
# drawing uniformly from a town's ~100 spawn points, 22 segments collide about
# 2.4 times by the birthday bound, and Town02 measured exactly that -- 18 distinct
# spawn points across 22 segments, with v4_check_dup.py flagging 3 duplicate route
# pairs (r1_/r5_, r28_/r32_, r15_/r21_) even though every segment ran with the
# flag.  So assign the spawn point explicitly instead: index 13*n, taken mod the
# town's spawn count inside the collector.  gcd(13, spawn_count) = 1 for every
# CARLA town used here, so the ids stay distinct for the first ~100 segments, and
# the choice is reproducible rather than lucky.
#
# $spawnBase offsets the sequence past whatever a previous run already collected,
# so resuming a half-finished city (which is the normal case on this machine)
# cannot hand a new segment the same spawn point an old one used.
# It must be derived from the spawn ids ALREADY USED, not from how many segments
# left frames behind.  Counting surviving prefixes undercounts badly: Town05's
# filtered pass ran 32 segments but only 17 produced any frames, so the negatives
# pass restarted at 13x18 and re-walked ids the filtered pass had already used --
# neg1/neg2/neg4 drew exactly the same spawn points as s18/s19/s21 and retraced
# their routes, which v4_check_dup.py then flagged as duplicate routes. Every
# attempted segment leaves a manifest snapshot, including the ones that crashed
# before saving a frame, so the snapshots are the reliable record.
$usedIds = @()
foreach ($mf in (Get-ChildItem $Out -Filter 'manifest_*.json' -File -ErrorAction SilentlyContinue)) {
    $m = Select-String -Path $mf.FullName -Pattern '--spawn_id (\d+)' -AllMatches
    foreach ($mm in $m.Matches) { $usedIds += [int]$mm.Groups[1].Value }
}
$spawnBase = 0
if ($usedIds.Count -gt 0) {
    $spawnBase = [math]::Floor((($usedIds | Measure-Object -Maximum).Maximum) / 13) + 1
}
Write-Output "spawn sequence starts at index 13 x $spawnBase ($($usedIds.Count) spawn ids already used in this dataset)"

while ($true) {
    $have = Get-Saved
    if ($have -ge $Target) {
        Write-Output "DONE $name : $have / $Target frames"
        break
    }
    if ($seg -ge $MaxSeg) {
        Write-Output "GIVING UP $name after $seg segments: $have / $Target frames"
        break
    }
    if ($deadRuns -ge $MaxDead) {
        Write-Output "GIVING UP $name : $MaxDead consecutive segments saved nothing ($have / $Target)"
        break
    }

    $need = $Target - $have

    # Never reuse a prefix that already has frames on disk.  $seg restarts at 0 on
    # every run, so re-running a city with the same -PrefixTag would renumber a new
    # segment over an old one's files (s1_000000.png etc.) and silently destroy
    # them -- that is how Town02's first 594 frames nearly died on 2026-07-30, and
    # this machine gets interrupted often enough (crash, sleep, Ctrl-C) that
    # resuming a half-collected city is the normal case, not the exception.
    if (-not $Single) {
        while ((Get-ChildItem (Join-Path $Out 'bev_gt') -Filter "${PrefixTag}${seg}_*.png" `
                              -File -ErrorAction SilentlyContinue).Count -gt 0) {
            Write-Output "prefix ${PrefixTag}${seg}_ already has frames, skipping to next"
            $seg++
        }
    }
    $prefix = if ($Single) { '' } else { "${PrefixTag}${seg}_" }
    $log    = Join-Path $LOGDIR "v4_${name}_${PrefixTag}${seg}.log"

    Write-Output "--- segment $seg : need $need more (have $have) ---"

    if (-not (Start-Carla)) {
        Write-Output "server never became ready, retrying"
        $seg++
        continue
    }

    if (-not (Prewarm-Map)) {
        Write-Output "server never reached $Town, retrying"
        $seg++
        continue
    }

    $a = @('-u', "$SRC\carla_dualcam_collect_gt.py",
           '--host', '127.0.0.1', '--port', '2000',
           '--town', $Town, '--out', $Out,
           '--frames', "$need", '--depth',
           '--cam_w', "$CamW", '--cam_h', "$CamH",
           '--vehicles', "$Vehicles", '--walkers', "$Walkers",
           # Explicit, distinct spawn point per segment -- see $spawnBase above.
           # (--spawn_id takes precedence over --random_spawn in the collector:
           # carla_dualcam_collect_gt.py:1035.  Without either, the ego always
           # spawns at spawn_points[0] and every segment retraces one route:
           # measured on Town02, 84% of s1_/s3_ frames were near-identical to s0_.
           # Those duplicates survive split_contiguous, which only guards against
           # ADJACENT frames, and leak train frames into val/test.)
           '--spawn_id', "$(13 * ($spawnBase + $seg))",
           '--preview', (Join-Path $Out "preview_${PrefixTag}${seg}.mp4"))
    if ($prefix)  { $a += @('--prefix', $prefix) }
    if ($Filter)  { $a += '--filter' }
    # Cut a segment loose once the ego has been stationary for 20 s, so the next
    # segment can try a different spawn point.  200 ticks at 10 fps sits above the
    # longest legitimate stop measured in the finished cities (Town04's worst red
    # light was 146 frames / 15 s) and well below a real deadlock, which never
    # ends.  NOT for -Single: val/test must stay one continuous run (規劃 §3), and
    # truncating them mid-drive would break the TTC/hazard demo.
    if (-not $Single) { $a += @('--stuck_abort', '200') }
    if ($NearEgo -gt 0) { $a += @('--near_ego', "$NearEgo") }

    $t0 = Get-Date
    Start-Process -FilePath $PY -ArgumentList $a -WorkingDirectory $SRC -Wait -NoNewWindow `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

    # The collector rebuilds poses/manifest from scratch each run and overwrites
    # the file, so a later segment destroys the earlier segments' records.  Keep
    # a per-segment snapshot; v4_merge_segments.py stitches them back together.
    # NOTE: do not name a local $src here -- PowerShell variables are
    # case-insensitive, so it would clobber the script-level $SRC (the source
    # directory) and silently break wait_carla_ready.py and the merge step for
    # every later segment.  That bug cost two cities on 2026-07-30.
    foreach ($f in @('poses.json', 'manifest.json')) {
        $snapFrom = Join-Path $Out $f
        if (Test-Path $snapFrom) {
            $stem = [IO.Path]::GetFileNameWithoutExtension($f)
            Copy-Item $snapFrom (Join-Path $Out "${stem}_${PrefixTag}${seg}.json") -Force
        }
    }

    $now  = Get-Saved
    $gain = $now - $have
    Write-Output "segment $seg : +$gain frames in $mins min (total $now / $Target)"

    if ($gain -le 0) { $deadRuns++ } else { $deadRuns = 0 }

    if ($Single) {
        if ($now -ge $Target) {
            Write-Output "DONE $name (single continuous run): $now frames"
        } else {
            Write-Output "INCOMPLETE $name : single run stopped at $now / $Target -- must be re-collected as one run, not resumed"
        }
        break
    }
    $seg++
}

Stop-Carla

# rebuild the full poses.json / per-segment provenance from the snapshots
& $PY "$SRC\v4_merge_segments.py" $Out

Write-Output "=== driver finished for $name ==="
