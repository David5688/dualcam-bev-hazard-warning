# collect_v4_all.ps1 -- runs the rest of the v4 collection plan unattended.
#
# Per 資料收集規劃.md, with the 2026-07-30 amendment: each training city gets a
# second pass of 500 UNFILTERED, near_ego-OFF frames (prefix neg0_) so that
# natural-distribution / low-actor frames are ~17% of that city's data.  §4 asks
# for 15-25% negatives; the filtered pass alone produced 0.9%.
#
# val/test are one continuous unfiltered run each (§3), never segmented.

$ErrorActionPreference = 'Stop'
$DRV  = 'E:\dualcam_project\src\collect_v4_driver.ps1'
$DATA = 'E:\dualcam_project\data\v4'

function Count-Frames([string]$dir) {
    $d = Join-Path $dir 'bev_gt'
    if (-not (Test-Path $d)) { return 0 }
    return (Get-ChildItem $d -Filter '*.png' -File).Count
}

# 500 natural-distribution frames on top of whatever the filtered pass produced.
function Add-Negatives([string]$town, [string]$out) {
    $have = Count-Frames $out
    if ($have -eq 0) {
        Write-Output "SKIP negatives for $town : main pass produced nothing"
        return
    }
    $target = $have + 500
    Write-Output ">>> negatives pass: $town  $have -> $target (unfiltered, near_ego off)"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $DRV -Town $town -Out $out `
        -Target $target -NearEgo 0 -PrefixTag 'neg'
}

function Collect-TrainCity([string]$town, [string]$out) {
    Write-Output ">>> train city: $town -> $out"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $DRV -Town $town -Out $out `
        -Target 2500 -Filter -NearEgo 150
    Add-Negatives $town $out
}

# --- Town10HD and Town02 are both re-collected from scratch with the fixed
#     driver.  Town02 because its four segments all started at spawn_points[0]
#     and 84% of s1_/s3_ frames were near-identical to s0_; Town10HD because the
#     lost poses.json for its main pass cannot be rebuilt any other way.
#     The old data is parked in train_town10hd_superseded / train_town02_bad_spawn.
Collect-TrainCity 'Town10HD' "$DATA\train_town10hd"
Collect-TrainCity 'Town02'   "$DATA\train_town02"

# --- remaining training cities: filtered pass + negatives pass
Collect-TrainCity 'Town04' "$DATA\train_town04"
Collect-TrainCity 'Town05' "$DATA\train_town05"
Collect-TrainCity 'Town06' "$DATA\train_town06"

# --- val / test: single continuous unfiltered run, natural distribution (§3, §14)
Write-Output '>>> val city: Town07 (single continuous, unfiltered)'
& powershell -NoProfile -ExecutionPolicy Bypass -File $DRV -Town Town07 -Out "$DATA\val_town07" `
    -Target 2700 -NearEgo 0 -Single

Write-Output '>>> test city: Town01 (single continuous, unfiltered)'
& powershell -NoProfile -ExecutionPolicy Bypass -File $DRV -Town Town01 -Out "$DATA\test_town01" `
    -Target 2700 -NearEgo 0 -Single

Write-Output '=== collect_v4_all finished ==='
foreach ($d in (Get-ChildItem $DATA -Directory)) {
    "{0,-20} {1}" -f $d.Name, (Count-Frames $d.FullName)
}
