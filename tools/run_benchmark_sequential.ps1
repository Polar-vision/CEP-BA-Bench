param(
    [string]$ProblemsDir = "E:\zuo\projects\CEP\benchmark_initial_value_problems",
    [string]$CleanOutDir = "E:\zuo\projects\CEP\benchmark_initial_value_clean_logged_20260920_singlethread_alone",
    [string]$DiagnosticOutDir = "E:\zuo\projects\CEP\benchmark_initial_value_diagnostic_strict_v2_20260920_singlethread_alone",
    [string]$OrchestrationLog = "E:\zuo\projects\CEP\benchmark_sequential_20260920_timestamped.log",
    [int]$TotalTasks = 1470,
    [int]$IntervalSeconds = 30
)

$ErrorActionPreference = "Stop"

$repoRoot = "E:\zuo\projects\CEP"
$releaseDir = Join-Path $repoRoot "example_v2\build\Release"
$loggedReleaseDir = Join-Path $repoRoot "example_v2\build_logged\Release"
$problemsPath = [System.IO.Path]::GetFullPath($ProblemsDir)
$cleanPath = [System.IO.Path]::GetFullPath($CleanOutDir)
$diagnosticPath = [System.IO.Path]::GetFullPath($DiagnosticOutDir)
$orchestrationLog = [System.IO.Path]::GetFullPath($OrchestrationLog)

function Assert-NewDirectory {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        $items = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue)
        if ($items.Count -gt 0) {
            throw "Output directory is not empty: $Path"
        }
    } else {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Write-Log {
    param(
        [System.IO.StreamWriter]$Writer,
        [string]$Phase,
        [string]$Message
    )
    $now = Get-Date
    $timestamp = $now.ToString("yyyy-MM-dd HH:mm:ss.fff")
    $Writer.WriteLine("[$timestamp][$Phase] $Message")
}

function Get-SummaryProgress {
    param([string]$OutPath)
    $summaryPath = Join-Path $OutPath "summary.csv"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        return [pscustomobject]@{
            Rows = 0
            Percent = 0.0
            LastWrite = "n/a"
            LastCompleted = "n/a"
            LastDataset = "n/a"
            LastQuality = "n/a"
            LastMethod = "n/a"
        }
    }

    $lines = @(Get-Content -LiteralPath $summaryPath)
    $rows = [Math]::Max(0, $lines.Count - 1)
    $percent = if ($TotalTasks -gt 0) {
        [Math]::Min(100.0, 100.0 * $rows / $TotalTasks)
    } else {
        0.0
    }
    $lastCompleted = if ($rows -gt 0) {
        $last = $lines[$lines.Count - 1]
        if ($last.Length -gt 180) { $last.Substring(0, 180) + "..." } else { $last }
    } else {
        "n/a"
    }
    $lastParts = if ($rows -gt 0) { $lines[$lines.Count - 1] -split "," } else { @() }
    [pscustomobject]@{
        Rows = $rows
        Percent = $percent
        LastWrite = (Get-Item -LiteralPath $summaryPath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        LastCompleted = $lastCompleted
        LastDataset = if ($lastParts.Count -gt 0) { $lastParts[0] } else { "n/a" }
        LastQuality = if ($lastParts.Count -gt 1) { $lastParts[1] } else { "n/a" }
        LastMethod = if ($lastParts.Count -gt 2) { $lastParts[2] } else { "n/a" }
    }
}

function Run-Phase {
    param(
        [string]$Phase,
        [string]$ExePath,
        [string]$WorkingDirectory,
        [string]$OutPath,
        [string]$Arguments,
        [hashtable]$Environment
    )

    Assert-NewDirectory $OutPath
    $phaseLog = Join-Path $OutPath "monitor_timestamped.log"
    $phaseWriter = New-Object System.IO.StreamWriter($phaseLog, $false, [System.Text.Encoding]::UTF8)
    $phaseWriter.AutoFlush = $true
    $stdoutWriter = New-Object System.IO.StreamWriter((Join-Path $OutPath "stdout_timestamped.log"), $false, [System.Text.Encoding]::UTF8)
    $stderrWriter = New-Object System.IO.StreamWriter((Join-Path $OutPath "stderr_timestamped.log"), $false, [System.Text.Encoding]::UTF8)
    $stdoutWriter.AutoFlush = $true
    $stderrWriter.AutoFlush = $true
    $commandText = "`"$ExePath`" $Arguments"
    Set-Content -LiteralPath (Join-Path $OutPath "run_command.txt") -Value $commandText -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $OutPath "thread_config.txt") -Value @(
        "CEP_NUM_THREADS=1"
        if ($Environment.ContainsKey("CEP_STRICT_VECTOR_DIAGNOSTICS")) {
            "CEP_STRICT_VECTOR_DIAGNOSTICS=$($Environment.CEP_STRICT_VECTOR_DIAGNOSTICS)"
        } else {
            "CEP_STRICT_VECTOR_DIAGNOSTICS=(unset)"
        }
    ) -Encoding UTF8

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ExePath
    $psi.Arguments = $Arguments
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($name in $Environment.Keys) {
        $psi.Environment[$name] = [string]$Environment[$name]
    }

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $state = [hashtable]::Synchronized(@{
        CurrentIndex = 0
        CurrentTotal = $TotalTasks
        CurrentDataset = "n/a"
        CurrentQuality = "n/a"
        CurrentMethod = "n/a"
    })
    $stdoutHandler = [System.Diagnostics.DataReceivedEventHandler]{
        param($sender, $event)
        if ($null -ne $event.Data) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
            $stdoutWriter.WriteLine("[$timestamp][stdout] $($event.Data)")
            $stdoutWriter.Flush()
            if ($event.Data -match '^\[(\d+)\/(\d+)\]\s+(.+?)\s+\/\s+(.+?)\s+\/\s+(.+?)\s+\(([^)]+)\)') {
                $state.CurrentIndex = [int]$matches[1]
                $state.CurrentTotal = [int]$matches[2]
                $state.CurrentDataset = $matches[3]
                $state.CurrentQuality = $matches[4]
                $state.CurrentMethod = $matches[5]
            }
        }
    }.GetNewClosure()
    $stderrHandler = [System.Diagnostics.DataReceivedEventHandler]{
        param($sender, $event)
        if ($null -ne $event.Data) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
            $stderrWriter.WriteLine("[$timestamp][stderr] $($event.Data)")
            $stderrWriter.Flush()
        }
    }.GetNewClosure()
    $process.add_OutputDataReceived($stdoutHandler)
    $process.add_ErrorDataReceived($stderrHandler)
    $started = Get-Date
    Set-Content -LiteralPath (Join-Path $OutPath "started_at_timestamped.txt") -Value $started.ToString("o") -Encoding UTF8
    Write-Log $phaseWriter $Phase "Starting task; total_tasks=$TotalTasks; threads=1"
    Write-Log $phaseWriter $Phase "Executable=$ExePath"
    Write-Log $phaseWriter $Phase "Problems=$problemsPath"
    Write-Log $phaseWriter $Phase "Output=$OutPath"
    Write-Log $phaseWriter $Phase "Command=$commandText"

    try {
        if (-not $process.Start()) {
            throw "Failed to start $Phase."
        }
        Set-Content -LiteralPath (Join-Path $OutPath "process.pid") -Value ([string]$process.Id) -Encoding ASCII
        Write-Log $phaseWriter $Phase "Process started; pid=$($process.Id)"
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()

        while (-not $process.HasExited) {
            Start-Sleep -Seconds $IntervalSeconds
            $progress = Get-SummaryProgress $OutPath
            Write-Log $phaseWriter $Phase ("progress={0}/{1} ({2:N2}%); current_task={3}/{4}; current_dataset={5}; current_quality={6}; current_method={7}; last_completed_dataset={8}; last_completed_quality={9}; last_completed_method={10}; pid={11}; summary_last_write={12}" -f `
                $progress.Rows, $TotalTasks, $progress.Percent, $state.CurrentIndex, $state.CurrentTotal, $state.CurrentDataset, $state.CurrentQuality, $state.CurrentMethod, `
                $progress.LastDataset, $progress.LastQuality, $progress.LastMethod, $process.Id, $progress.LastWrite)
        }

        $process.WaitForExit()
        $process.WaitForExit()
        $progress = Get-SummaryProgress $OutPath
        $exitCode = $process.ExitCode
        Write-Log $phaseWriter $Phase ("finished; exit_code={0}; progress={1}/{2} ({3:N2}%); last_completed_dataset={4}; last_completed_quality={5}; last_completed_method={6}" -f `
            $exitCode, $progress.Rows, $TotalTasks, $progress.Percent, $progress.LastDataset, $progress.LastQuality, $progress.LastMethod)
        Set-Content -LiteralPath (Join-Path $OutPath "exit_code.txt") -Value ([string]$exitCode) -Encoding ASCII
        Set-Content -LiteralPath (Join-Path $OutPath "finished_at_timestamped.txt") -Value (Get-Date).ToString("o") -Encoding UTF8
        if ($exitCode -ne 0) {
            throw "$Phase exited with code $exitCode."
        }
    } finally {
        $process.Dispose()
        $phaseWriter.Dispose()
        $stdoutWriter.Dispose()
        $stderrWriter.Dispose()
    }
}

Assert-NewDirectory $cleanPath
Assert-NewDirectory $diagnosticPath
$globalWriter = New-Object System.IO.StreamWriter($orchestrationLog, $true, [System.Text.Encoding]::UTF8)
$globalWriter.AutoFlush = $true
try {
    Write-Log $globalWriter "orchestrator" "Sequential benchmark started; clean then diagnostic; total_tasks=$TotalTasks; threads=1"
    $envClean = @{ CEP_NUM_THREADS = "1" }
    $cleanArguments = "--problems `"$problemsPath`" --out `"$cleanPath`" --mode clean-logged --resume"
    Run-Phase "clean-logged" (Join-Path $loggedReleaseDir "example.exe") $loggedReleaseDir $cleanPath $cleanArguments $envClean
    Write-Log $globalWriter "orchestrator" "clean-logged completed; starting strict diagnostic"

    $envDiagnostic = @{
        CEP_NUM_THREADS = "1"
        CEP_STRICT_VECTOR_DIAGNOSTICS = "1"
    }
    $diagnosticArguments = "--problems `"$problemsPath`" --out `"$diagnosticPath`" --mode diagnostic --resume --point-condition-sample 128 --schur-sample 32"
    Run-Phase "diagnostic-strict-v2" (Join-Path $releaseDir "example_strict.exe") $releaseDir $diagnosticPath $diagnosticArguments $envDiagnostic
    Write-Log $globalWriter "orchestrator" "Sequential benchmark completed successfully"
} catch {
    Write-Log $globalWriter "orchestrator" ("FAILED: " + $_.Exception.Message)
    throw
} finally {
    $globalWriter.Dispose()
}
