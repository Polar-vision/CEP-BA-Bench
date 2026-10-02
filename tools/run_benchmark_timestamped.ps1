param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("clean-logged", "diagnostic")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [string]$ProblemsDir = "E:\zuo\projects\CEP\benchmark_initial_value_problems",
    [int]$PointConditionSample = 128,
    [int]$SchurSample = 32
)

$ErrorActionPreference = "Stop"

$repoRoot = "E:\zuo\projects\CEP"
$releaseDir = if ($Mode -eq "clean-logged") {
    Join-Path $repoRoot "example_v2\build_logged\Release"
} else {
    Join-Path $repoRoot "example_v2\build\Release"
}
$exePath = if ($Mode -eq "diagnostic") {
    Join-Path $releaseDir "example_strict.exe"
} else {
    Join-Path $releaseDir "example.exe"
}

$outPath = [System.IO.Path]::GetFullPath($OutDir)
$problemsPath = [System.IO.Path]::GetFullPath($ProblemsDir)
$progressPath = Join-Path $outPath "progress_timestamped.log"
$stdoutPath = Join-Path $outPath "stdout_timestamped.log"
$stderrPath = Join-Path $outPath "stderr_timestamped.log"

if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
    throw "Executable not found: $exePath"
}
if (-not (Test-Path -LiteralPath $problemsPath -PathType Container)) {
    throw "Problems directory not found: $problemsPath"
}
if (Test-Path -LiteralPath $outPath) {
    $existing = Get-ChildItem -LiteralPath $outPath -Force -ErrorAction SilentlyContinue
    if ($existing) {
        throw "Output directory is not empty; refusing to overwrite: $outPath"
    }
} else {
    New-Item -ItemType Directory -Path $outPath | Out-Null
}

$arguments = @(
    "--problems", "`"$problemsPath`"",
    "--out", "`"$outPath`"",
    "--mode", $Mode,
    "--resume"
)
if ($Mode -eq "diagnostic") {
    $arguments += @(
        "--point-condition-sample", [string]$PointConditionSample,
        "--schur-sample", [string]$SchurSample
    )
}
$argumentText = $arguments -join " "

$env:CEP_NUM_THREADS = "1"
if ($Mode -eq "diagnostic") {
    $env:CEP_STRICT_VECTOR_DIAGNOSTICS = "1"
} else {
    Remove-Item Env:CEP_STRICT_VECTOR_DIAGNOSTICS -ErrorAction SilentlyContinue
}

$commandText = "`"$exePath`" $argumentText"
Set-Content -LiteralPath (Join-Path $outPath "run_command.txt") -Value $commandText -Encoding UTF8
Set-Content -LiteralPath (Join-Path $outPath "thread_config.txt") -Value @(
    "CEP_NUM_THREADS=1"
    if ($Mode -eq "diagnostic") { "CEP_STRICT_VECTOR_DIAGNOSTICS=1" }
    else { "CEP_STRICT_VECTOR_DIAGNOSTICS=(unset)" }
) -Encoding UTF8

$progressWriter = New-Object System.IO.StreamWriter($progressPath, $false, [System.Text.Encoding]::UTF8)
$stdoutWriter = New-Object System.IO.StreamWriter($stdoutPath, $false, [System.Text.Encoding]::UTF8)
$stderrWriter = New-Object System.IO.StreamWriter($stderrPath, $false, [System.Text.Encoding]::UTF8)
$progressWriter.AutoFlush = $true
$stdoutWriter.AutoFlush = $true
$stderrWriter.AutoFlush = $true
$sync = New-Object object
$started = Get-Date
$summaryPath = Join-Path $outPath "summary.csv"

function Write-TimestampedLine {
    param(
        [System.IO.StreamWriter]$Writer,
        [string]$StreamName,
        [string]$Line
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $elapsed = ((Get-Date) - $started).TotalSeconds.ToString("F3", [Globalization.CultureInfo]::InvariantCulture)
    [System.Threading.Monitor]::Enter($sync)
    try {
        $Writer.WriteLine("[$timestamp][+${elapsed}s][$StreamName] $Line")
    } finally {
        [System.Threading.Monitor]::Exit($sync)
    }
}

Set-Content -LiteralPath (Join-Path $outPath "started_at_timestamped.txt") -Value $started.ToString("o") -Encoding UTF8
Write-TimestampedLine $progressWriter "launcher" "Starting mode=$Mode, threads=1"
Write-TimestampedLine $progressWriter "launcher" "Executable=$exePath"
Write-TimestampedLine $progressWriter "launcher" "Problems=$problemsPath"
Write-TimestampedLine $progressWriter "launcher" "Output=$outPath"
Write-TimestampedLine $progressWriter "launcher" "Command=$commandText"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exePath
$psi.Arguments = $argumentText
$psi.WorkingDirectory = $releaseDir
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$process = New-Object System.Diagnostics.Process
$process.StartInfo = $psi

$stdoutHandler = [System.Diagnostics.DataReceivedEventHandler]{
    param($sender, $event)
    if ($null -ne $event.Data) {
        Write-TimestampedLine $stdoutWriter "stdout" $event.Data
        Write-TimestampedLine $progressWriter "stdout" $event.Data
    }
}
$stderrHandler = [System.Diagnostics.DataReceivedEventHandler]{
    param($sender, $event)
    if ($null -ne $event.Data) {
        Write-TimestampedLine $stderrWriter "stderr" $event.Data
        Write-TimestampedLine $progressWriter "stderr" $event.Data
    }
}
$process.add_OutputDataReceived($stdoutHandler)
$process.add_ErrorDataReceived($stderrHandler)

try {
    if (-not $process.Start()) {
        throw "Failed to start benchmark process."
    }
    Set-Content -LiteralPath (Join-Path $outPath "process.pid") -Value ([string]$process.Id) -Encoding ASCII
    Write-TimestampedLine $progressWriter "launcher" "Process started with PID=$($process.Id)"
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
    while (-not $process.HasExited) {
        Start-Sleep -Seconds 30
        if (Test-Path -LiteralPath $summaryPath) {
            $summaryLineCount = @((Get-Content -LiteralPath $summaryPath -ReadCount 0)).Count
            $summaryRows = [Math]::Max(0, $summaryLineCount - 1)
            $summaryUpdated = (Get-Item -LiteralPath $summaryPath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
            Write-TimestampedLine $progressWriter "heartbeat" "PID=$($process.Id); summary_rows=$summaryRows; summary_last_write=$summaryUpdated"
        } else {
            Write-TimestampedLine $progressWriter "heartbeat" "PID=$($process.Id); summary.csv not created yet"
        }
    }
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    Write-TimestampedLine $progressWriter "launcher" "Process exited with code=$exitCode"
    Set-Content -LiteralPath (Join-Path $outPath "exit_code.txt") -Value ([string]$exitCode) -Encoding ASCII
    $finished = Get-Date
    Set-Content -LiteralPath (Join-Path $outPath "finished_at_timestamped.txt") -Value $finished.ToString("o") -Encoding UTF8
    if ($exitCode -ne 0) {
        exit $exitCode
    }
} finally {
    $process.Dispose()
    $progressWriter.Dispose()
    $stdoutWriter.Dispose()
    $stderrWriter.Dispose()
}
