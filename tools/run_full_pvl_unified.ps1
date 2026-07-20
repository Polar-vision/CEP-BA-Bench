$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $repo "example_v2\build\Release\example.exe"
$problems = Join-Path $repo "PVL-BA-Bench\public_release\extracted\pvl-ba"
$output = Join-Path $repo "benchmark_public_release_unified_clean"

if (-not (Test-Path -LiteralPath $executable)) {
    throw "Benchmark executable not found: $executable"
}
if (-not (Test-Path -LiteralPath $problems)) {
    throw "Extracted PVL-BA-Bench directory not found: $problems"
}

New-Item -ItemType Directory -Path $output -Force | Out-Null

$env:CEP_NUM_THREADS = "1"
$env:CEP_MAX_NUM_ITERATIONS = "100"
$env:CEP_FIX_MONOCULAR_GAUGE = "1"
Remove-Item Env:CEP_FUNCTION_TOLERANCE -ErrorAction SilentlyContinue
Remove-Item Env:CEP_GRADIENT_TOLERANCE -ErrorAction SilentlyContinue
Remove-Item Env:CEP_PARAMETER_TOLERANCE -ErrorAction SilentlyContinue
Remove-Item Env:CEP_INITIAL_TRUST_REGION_RADIUS -ErrorAction SilentlyContinue

$command = "`"$executable`" --problems `"$problems`" --out `"$output`" --mode clean --resume"
$command | Set-Content -LiteralPath (Join-Path $output "run_command.txt") -Encoding UTF8
(Get-Date -Format o) | Set-Content -LiteralPath (Join-Path $output "started_at.txt") -Encoding UTF8

& $executable --problems $problems --out $output --mode clean --resume
$exitCode = $LASTEXITCODE

@(
    "completed_at=$(Get-Date -Format o)"
    "exit_code=$exitCode"
) | Set-Content -LiteralPath (Join-Path $output "completed.txt") -Encoding UTF8

exit $exitCode
