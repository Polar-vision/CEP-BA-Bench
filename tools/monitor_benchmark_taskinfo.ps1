param(
    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [Parameter(Mandatory = $true)]
    [int]$ProcessId,

    [int]$TotalTasks = 1470,
    [int]$IntervalSeconds = 30
)

$ErrorActionPreference = "Stop"
$outPath = [System.IO.Path]::GetFullPath($OutDir)
$logPath = Join-Path $outPath "monitor_timestamped.log"
$writer = New-Object System.IO.StreamWriter($logPath, $true, [System.Text.Encoding]::UTF8)
$writer.AutoFlush = $true

function Get-ProgressSnapshot {
    $summaryPath = Join-Path $outPath "summary.csv"
    $rows = 0
    $lastDataset = "n/a"
    $lastQuality = "n/a"
    $lastMethod = "n/a"
    $summaryLastWrite = "n/a"
    if (Test-Path -LiteralPath $summaryPath) {
        $lines = @(Get-Content -LiteralPath $summaryPath)
        $rows = [Math]::Max(0, $lines.Count - 1)
        if ($rows -gt 0) {
            $parts = $lines[$lines.Count - 1] -split ","
            if ($parts.Count -gt 0) { $lastDataset = $parts[0] }
            if ($parts.Count -gt 1) { $lastQuality = $parts[1] }
            if ($parts.Count -gt 2) { $lastMethod = $parts[2] }
        }
        $summaryLastWrite = (Get-Item -LiteralPath $summaryPath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    }

    $active = Get-ChildItem -LiteralPath $outPath -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            $datasetDir = $_
            $qualityDir = Join-Path $datasetDir.FullName "Initial Value"
            if (Test-Path -LiteralPath $qualityDir) {
                Get-ChildItem -LiteralPath $qualityDir -Directory -ErrorAction SilentlyContinue |
                    ForEach-Object {
                        [pscustomobject]@{
                            Dataset = $datasetDir.Name
                            Quality = "Initial Value"
                            Method = $_.Name
                            Created = $_.CreationTime
                            Updated = $_.LastWriteTime
                            Files = @(Get-ChildItem -LiteralPath $_.FullName -File -ErrorAction SilentlyContinue).Count
                        }
                    }
            }
        } | Sort-Object Created, Updated -Descending | Select-Object -First 1

    [pscustomobject]@{
        Rows = $rows
        Percent = if ($TotalTasks -gt 0) { [Math]::Min(100.0, 100.0 * $rows / $TotalTasks) } else { 0.0 }
        LastDataset = $lastDataset
        LastQuality = $lastQuality
        LastMethod = $lastMethod
        SummaryLastWrite = $summaryLastWrite
        ActiveDataset = if ($null -ne $active) { $active.Dataset } else { "n/a" }
        ActiveQuality = if ($null -ne $active) { $active.Quality } else { "n/a" }
        ActiveMethod = if ($null -ne $active) { $active.Method } else { "n/a" }
        ActiveFiles = if ($null -ne $active) { $active.Files } else { 0 }
    }
}

try {
    while ($true) {
        $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        $snapshot = Get-ProgressSnapshot
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
        $state = if ($null -ne $process) { "running" } else { "exited" }
        $writer.WriteLine(("[{0}][monitor] state={1}; progress={2}/{3} ({4:N2}%); active_dataset={5}; active_quality={6}; active_method={7}; active_files={8}; last_completed_dataset={9}; last_completed_quality={10}; last_completed_method={11}; summary_last_write={12}" -f `
            $timestamp, $state, $snapshot.Rows, $TotalTasks, $snapshot.Percent, $snapshot.ActiveDataset, $snapshot.ActiveQuality, `
            $snapshot.ActiveMethod, $snapshot.ActiveFiles, $snapshot.LastDataset, $snapshot.LastQuality, $snapshot.LastMethod, $snapshot.SummaryLastWrite))
        if ($null -eq $process) {
            break
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
} finally {
    $writer.Dispose()
}
