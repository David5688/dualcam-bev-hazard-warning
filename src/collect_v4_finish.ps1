# collect_v4_finish.ps1 -- collects the two training cities that collect_v4_all.ps1
# failed to produce on 2026-07-30 (Town02 and Town05).
#
# Two separate failures, both now fixed in collect_v4_driver.ps1:
#
#  1. The per-segment snapshot loop used a local $src which, PowerShell variables
#     being case-insensitive, clobbered the script-level $SRC.  Every segment after
#     the first then failed to find wait_carla_ready.py, so the server was never
#     seen as ready and the city stalled (Town02 186 frames, Town05 36).
#     Fixed 09:17; the partial runs are parked in train_town0{2,5}_crash_partial.
#
#  2. With retries working, Town02 still lost 3 of 4 segments to a server crash on
#     the FIRST tick (manifest stats: ticks 1, saved 0).
#
#     ~~The cause is the collector's own load_world() followed immediately by
#     spawning four sensors; prewarm_map.py fixes it by making the collector skip
#     its load.~~  REFUTED 2026-07-30 21:00 -- the r0_/r1_ segments both ran WITH
#     prewarm (their logs confirm "server already on Carla/Maps/Town02, skipping
#     load_world") and still died on tick 1, while Town04/06/07/01 collected
#     2,500-2,700 frames each doing their own load_world before prewarm existed.
#     load_world is not the trigger; prewarm is harmless but does nothing here.
#
#     What the 119 crash dumps in %LOCALAPPDATA%\CarlaUE4\Saved\Crashes actually
#     say: every first-tick death is on RenderThread 38-50 s after server start,
#     and the largest crash family project-wide (44 dumps, going back to 06-22) is
#     "LowLevelFatalError [Line: 139] Shader compilation failures are Fatal."  It
#     is a render/shader failure, not an RPC or map-loading one -- and no
#     nvlddmkm/TDR events accompany it, so the GPU is not resetting.
#
#     Bisected with diag_carla_crash.ps1: Town02, identical parameters, depth
#     cameras OFF -> 60/60 frames in 28 s, server still alive afterwards, zero
#     crash dumps.  With --depth (4 sensors at 1280x720) every single run leaves a
#     crash dump.  The two depth cameras are the trigger.
#
# Town02 keeps the 594 good frames from segment s1_ and resumes under a fresh
# prefix tag ('r'): the driver restarts its segment counter at 0 on every run, so
# reusing 's' would have renumbered a second segment as s1_000000.png and
# overwritten those 594 frames.
#
# Recipe is otherwise identical to the three cities that did succeed
# (§6/§12/§14 of 資料收集規劃.md):
#   filtered pass  2,500 frames total, --filter, --near_ego 150
#   negatives pass   500 frames, unfiltered, near_ego off, prefix neg0_

$ErrorActionPreference = 'Stop'
$DRV  = 'E:\dualcam_project\src\collect_v4_driver.ps1'
$DATA = 'E:\dualcam_project\data\v4'
$MAIN = 2500   # filtered frames per training city
$NEG  = 500    # natural-distribution frames on top

function Count-Frames([string]$dir) {
    $d = Join-Path $dir 'bev_gt'
    if (-not (Test-Path $d)) { return 0 }
    return (Get-ChildItem $d -Filter '*.png' -File).Count
}

function Add-Negatives([string]$town, [string]$out) {
    $have = Count-Frames $out
    # Only worth adding once the filtered pass is actually complete: 500 unfiltered
    # frames on top of a short main pass would quietly change that city's filtered/
    # natural mix, which §14 wants at ~17%.
    if ($have -lt $MAIN) {
        Write-Output "SKIP negatives for $town : filtered pass short ($have / $MAIN)"
        return
    }
    $target = $have + $NEG
    Write-Output ">>> negatives pass: $town  $have -> $target (unfiltered, near_ego off)"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $DRV -Town $town -Out $out `
        -Target $target -NearEgo 0 -PrefixTag 'neg'
}

function Collect-TrainCity([string]$town, [string]$out, [string]$prefixTag) {
    Write-Output ">>> train city: $town -> $out  (prefix tag '$prefixTag')"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $DRV -Town $town -Out $out `
        -Target $MAIN -Filter -NearEgo 150 -PrefixTag $prefixTag
    Add-Negatives $town $out
}

# Town02 resumes on top of 594 existing s1_ frames -> new tag 'r'.
Collect-TrainCity 'Town02' "$DATA\train_town02" 'r'
# Town05 starts from an empty directory -> the default 's' tag is free.
Collect-TrainCity 'Town05' "$DATA\train_town05" 's'

Write-Output '=== collect_v4_finish finished ==='
foreach ($d in (Get-ChildItem $DATA -Directory)) {
    "{0,-30} {1}" -f $d.Name, (Count-Frames $d.FullName)
}
