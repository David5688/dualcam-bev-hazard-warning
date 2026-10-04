# collect_v4_town05.ps1 -- re-collect Town05 after the gridlock write-off.
#
# The first attempt (2026-07-30, parked in data/v4/retired/train_town05_gridlock)
# had to be discarded: the ego spent most of it standing in a deadlocked junction,
# and 70.9% of its frames were near-duplicates -- against 5.0% for Town04 and
# 25.6% for Town10HD under identical settings.  See that directory's
# PROVENANCE_NOTE.md and 資料收集規劃.md §15.3.
#
# What changed for this run:
#   --stuck_abort 200   the collector ends a segment once the ego has been
#                       stationary for 20 s, so the driver can restart it from a
#                       different spawn point.  Measured headroom: the longest
#                       genuine standstill seen while probing Town05 was 26 ticks
#                       (2.6 s), so a red light cannot trip it.
#   --spawn_id 13*n     distinct, reproducible spawn point per segment (replaces
#                       --random_spawn, whose draws collided on Town02).
#   Vehicle count stays 40, matching the other four training cities: halving it to
#   20 was probed and came out WORSE (73.2% duplicates), so the gridlock is about
#   where the ego ends up, not how many cars are around it.
#
# Both changes live in collect_v4_driver.ps1; this script only sequences the two
# passes for this one city.

$ErrorActionPreference = 'Stop'
$DRV  = 'E:\dualcam_project\src\collect_v4_driver.ps1'
$OUT  = 'E:\dualcam_project\data\v4\train_town05'
$MAIN = 2500   # filtered frames
$NEG  = 500    # natural-distribution frames on top (規劃 §14)

function Count-Frames {
    $d = Join-Path $OUT 'bev_gt'
    if (-not (Test-Path $d)) { return 0 }
    return (Get-ChildItem $d -Filter '*.png' -File).Count
}

Write-Output ">>> Town05 filtered pass -> $MAIN frames"
& powershell -NoProfile -ExecutionPolicy Bypass -File $DRV -Town Town05 -Out $OUT `
    -Target $MAIN -Filter -NearEgo 150

$have = Count-Frames
if ($have -lt $MAIN) {
    Write-Output "SKIP negatives: filtered pass short ($have / $MAIN)"
} else {
    Write-Output ">>> Town05 negatives pass -> $($have + $NEG) frames (unfiltered, near_ego off)"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $DRV -Town Town05 -Out $OUT `
        -Target ($have + $NEG) -NearEgo 0 -PrefixTag 'neg'
}

Write-Output "=== Town05 finished: $(Count-Frames) frames ==="
