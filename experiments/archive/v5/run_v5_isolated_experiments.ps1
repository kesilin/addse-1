$ErrorActionPreference = 'Stop'
Set-Location -Path "f:/ksl/addse"
$py = "f:/ksl/.venv312/Scripts/python.exe"
$script = "experiments/archive/v5/phase_fusion_layered_v5_only.py"
$cfgEval = "experiments/archive/v5/configs/v5_isolated_eval100.yaml"
$cfgDirect = "experiments/archive/v5/configs/v5_isolated_residual_weight_train40_eval100.yaml"

$jobs = @(
    @{ Name = 'fixed_add_100'; Cmd = "$py $script --config $cfgEval --mode eval_only --variants baseline --fusion-postprocess-mode fixed_add --report-json experiments/archive/v5/reports/v5_isolated_fixed_add_eval100.json" },
    @{ Name = 'mag_pm05_100'; Cmd = "$py $script --config $cfgEval --mode eval_only --variants baseline --fusion-postprocess-mode mag_scale_perturb --fusion-perturb-min 0.95 --fusion-perturb-max 1.05 --report-json experiments/archive/v5/reports/v5_isolated_mag_perturb_pm05_eval100.json" },
    @{ Name = 'mag_pm10_100'; Cmd = "$py $script --config $cfgEval --mode eval_only --variants baseline --fusion-postprocess-mode mag_scale_perturb --fusion-perturb-min 0.90 --fusion-perturb-max 1.10 --report-json experiments/archive/v5/reports/v5_isolated_mag_perturb_pm10_eval100.json" },
    @{ Name = 'mag_pm15_100'; Cmd = "$py $script --config $cfgEval --mode eval_only --variants baseline --fusion-postprocess-mode mag_scale_perturb --fusion-perturb-min 0.85 --fusion-perturb-max 1.15 --report-json experiments/archive/v5/reports/v5_isolated_mag_perturb_pm15_eval100.json" },
    @{ Name = 'residual_weight_eval100'; Cmd = "$py $script --config $cfgEval --mode eval_only --variants baseline --fusion-postprocess-mode residual_weight_direct --phase-fusion-weight-min 0.05 --phase-fusion-weight-max 0.35 --report-json experiments/archive/v5/reports/v5_isolated_residual_weight_direct_eval100.json" },
    @{ Name = 'residual_weight_train40_eval100'; Cmd = "$py $script --config $cfgDirect --mode quick_compare --variants residual_aware_lite --report-json experiments/archive/v5/reports/v5_isolated_residual_weight_direct_train40_eval100.json" }
)

foreach ($j in $jobs) {
    Write-Host "==== RUN $($j.Name) ===="
    Invoke-Expression $j.Cmd
}

$reports = @(
    "experiments/archive/v5/reports/v5_isolated_fixed_add_eval100.json",
    "experiments/archive/v5/reports/v5_isolated_mag_perturb_pm05_eval100.json",
    "experiments/archive/v5/reports/v5_isolated_mag_perturb_pm10_eval100.json",
    "experiments/archive/v5/reports/v5_isolated_mag_perturb_pm15_eval100.json",
    "experiments/archive/v5/reports/v5_isolated_residual_weight_direct_eval100.json",
    "experiments/archive/v5/reports/v5_isolated_residual_weight_direct_train40_eval100.json"
)

$missing = @()
foreach ($r in $reports) {
    if (-not (Test-Path $r)) {
        $missing += $r
    }
}

if ($missing.Count -eq 0) {
    Write-Host "READY"
} else {
    Write-Host "MISSING"
    $missing | ForEach-Object { Write-Host $_ }
}
