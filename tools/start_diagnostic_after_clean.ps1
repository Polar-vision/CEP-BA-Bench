param(
    [Parameter(Mandatory = $true)]
    [string]$CleanOutDir,

    [Parameter(Mandatory = $true)]
    [int]$CleanProcessId,

    [Parameter(Mandatory = $true)]
    [string]$DiagnosticOutDir,

    [string]$ProblemsDir = "E:\zuo\projects\CEP\benchmark_initial_value_problems",
    [string]$OrchestrationLog = "E:\zuo\projects\CEP\benchmark_sequential_tasklog_20260920_timestamped.log",
    [int]$TotalTasks = 1470,
    [int]$IntervalSeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = "E:\zuo\projects\CEP"
$releaseDir = Join-Path $repoRoot "example_v2\build\Release"
$problemsPath = [System.IO.Path]::GetFullPath($ProblemsDir)
$cleanPath = [System.IO.Path]::GetFullPath($CleanOutDir)
$diagnosticPath = [System.IO.Path]::GetFullPath($DiagnosticOutDir)
$monitorScript = Join-Path $repoRoot "tools\monitor_benchmark_taskinfo.ps1"
$globalWriter = New-Object System.IO.StreamWriter($OrchestrationLog, $true, [System.Text.Encoding]::UTF8)
$globalWriter.AutoFlush = $true

function Write-GlobalLog {
    param([string]$Phase, [string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $globalWriter.WriteLine("[$timestamp][$Phase] $Message")
}

function Get-CompletedRows {
    $summaryPath = Join-Path $cleanPath "summary.csv"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        return 0
    }
    return [Math]::Max(0, @(Get-Content -LiteralPath $summaryPath).Count - 1)
}

function Assert-NewOutput {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        $items = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue)
        if ($items.Count -gt 0) {
            throw "Diagnostic output directory is not empty: $Path"
        }
    } else {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

try {
    Write-GlobalLog "coordinator" "Waiting for clean-logged PID=$CleanProcessId to finish before starting diagnostic."
    while (Get-Process -Id $CleanProcessId -ErrorAction SilentlyContinue) {
        $rows = Get-CompletedRows
        $percent = if ($TotalTasks -gt 0) { 100.0 * $rows / $TotalTasks } else { 0.0 }
        Write-GlobalLog "clean-wait" ("progress={0}/{1} ({2:N2}%)" -f $rows, $TotalTasks, $percent)
        Start-Sleep -Seconds $IntervalSeconds
    }

    $rows = Get-CompletedRows
    if ($rows -lt $TotalTasks) {
        throw "Clean process exited before completion: $rows/$TotalTasks."
    }
    Write-GlobalLog "coordinator" "Clean-logged completed with $rows/$TotalTasks rows."

    Assert-NewOutput $diagnosticPath
    $exePath = Join-Path $releaseDir "example_strict.exe"
    $arguments = "--problems `"$problemsPath`" --out `"$diagnosticPath`" --mode diagnostic --resume --point-condition-sample 128 --schur-sample 32"
    $commandText = "`"$exePath`" $arguments"
    Set-Content -LiteralPath (Join-Path $diagnosticPath "run_command.txt") -Value $commandText -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $diagnosticPath "thread_config.txt") -Value @(
        "CEP_NUM_THREADS=1"
        "CEP_STRICT_VECTOR_DIAGNOSTICS=1"
    ) -Encoding UTF8

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exePath
    $psi.Arguments = $arguments
    $psi.WorkingDirectory = $releaseDir
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.EnvironmentVariables["CEP_NUM_THREADS"] = "1"
    $psi.EnvironmentVariables["CEP_STRICT_VECTOR_DIAGNOSTICS"] = "1"
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $started = Get-Date
    Set-Content -LiteralPath (Join-Path $diagnosticPath "started_at_timestamped.txt") -Value $started.ToString("o") -Encoding UTF8
    if (-not $process.Start()) {
        throw "Failed to start strict diagnostic."
    }
    Set-Content -LiteralPath (Join-Path $diagnosticPath "process.pid") -Value ([string]$process.Id) -Encoding ASCII
    Write-GlobalLog "coordinator" "Strict diagnostic started; pid=$($process.Id); threads=1."

    $monitor = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -WorkingDirectory $repoRoot -PassThru -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $monitorScript,
        "-OutDir", $diagnosticPath, "-ProcessId", [string]$process.Id,
        "-TotalTasks", [string]$TotalTasks, "-IntervalSeconds", [string]$IntervalSeconds
    )
    Write-GlobalLog "coordinator" "Diagnostic task monitor started; monitor_pid=$($monitor.Id)."

    while (-not $process.HasExited) {
        Start-Sleep -Seconds $IntervalSeconds
    }
    $process.WaitForExit()
    Set-Content -LiteralPath (Join-Path $diagnosticPath "exit_code.txt") -Value ([string]$process.ExitCode) -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $diagnosticPath "finished_at_timestamped.txt") -Value (Get-Date).ToString("o") -Encoding UTF8
    Write-GlobalLog "coordinator" "Strict diagnostic finished; exit_code=$($process.ExitCode)."
    $process.Dispose()
} catch {
    Write-GlobalLog "coordinator" ("FAILED: " + $_.Exception.Message)
    throw
} finally {
    $globalWriter.Dispose()
}
