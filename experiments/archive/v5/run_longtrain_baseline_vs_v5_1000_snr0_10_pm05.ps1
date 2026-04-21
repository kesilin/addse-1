$ErrorActionPreference = 'Stop'
Set-Location -Path "f:/ksl/addse"
$py = "f:/ksl/.venv312/Scripts/python.exe"
$script = "experiments/archive/v5/phase_fusion_layered_v5_only.py"

$cfgBaseline = "experiments/archive/v5/configs/v5_longtrain_baseline_discrete_1000_snr0_10_pm05.yaml"
$cfgV5 = "experiments/archive/v5/configs/v5_longtrain_parallel_v5_1000_snr0_10_pm05.yaml"

Write-Host "==== PREFLIGHT baseline_discrete (sanity) ===="
& $py $script --config $cfgBaseline --mode quick_compare --variants baseline --train-steps 1 --train-eval-every-steps 1 --train-eval-examples 5 --eval-examples 5 --pesq-max-examples 5 --report-json experiments/archive/v5/reports/v5_longtrain_baseline_discrete_preflight.json

Write-Host "==== PREFLIGHT parallel_v5 (sanity) ===="
& $py $script --config $cfgV5 --mode quick_compare --variants baseline --train-steps 1 --train-eval-every-steps 1 --train-eval-examples 5 --eval-examples 5 --pesq-max-examples 5 --report-json experiments/archive/v5/reports/v5_longtrain_parallel_v5_preflight.json

Write-Host "==== LONG TRAIN 1/2 baseline_discrete ===="
& $py $script --config $cfgBaseline --mode train_eval --variants baseline --report-json experiments/archive/v5/reports/v5_longtrain_baseline_discrete_1000_snr0_10_pm05_train_eval.json

Write-Host "==== LONG TRAIN 2/2 parallel_v5 ===="
& $py $script --config $cfgV5 --mode train_eval --variants baseline --report-json experiments/archive/v5/reports/v5_longtrain_parallel_v5_1000_snr0_10_pm05_train_eval.json

$reports = @(
  "experiments/archive/v5/reports/v5_longtrain_baseline_discrete_preflight.json",
  "experiments/archive/v5/reports/v5_longtrain_parallel_v5_preflight.json",
  "experiments/archive/v5/reports/v5_longtrain_baseline_discrete_1000_snr0_10_pm05_train_eval.json",
  "experiments/archive/v5/reports/v5_longtrain_parallel_v5_1000_snr0_10_pm05_train_eval.json"
)
$missing = @()
foreach ($r in $reports) {
  if (-not (Test-Path $r)) { $missing += $r }
}
if ($missing.Count -eq 0) {
  Write-Host "READY"
} else {
  Write-Host "MISSING"
  $missing | ForEach-Object { Write-Host $_ }
}
