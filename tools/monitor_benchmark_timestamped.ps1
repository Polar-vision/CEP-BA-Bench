param(
    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [Parameter(Mandatory = $true)]
    [int]$ProcessId,

    [int]$IntervalSeconds = 30
)

$ErrorActionPreference = "Stop"
$outPath = [System.IO.Path]::GetFullPath($OutDir)
$logPath = Join-Path $outPath "monitor_timestamped.log"
$summaryPath = Join-Path $outPath "summary.csv"
$writer = New-Object System.IO.StreamWriter($logPath, $false, [System.Text.Encoding]::UTF8)
$writer.AutoFlush = $true

try {
    while ($true) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
        $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $summaryPath) {
            $lineCount = @(Get-Content -LiteralPath $summaryPath).Count
            $rows = [Math]::Max(0, $lineCount - 1)
            $updated = (Get-Item -LiteralPath $summaryPath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
            $state = if ($null -eq $process) { "exited" } else { "running" }
            $writer.WriteLine("[$timestamp][monitor] pid=$ProcessId; state=$state; summary_rows=$rows; summary_last_write=$updated")
        } else {
            $state = if ($null -eq $process) { "exited" } else { "running" }
            $writer.WriteLine("[$timestamp][monitor] pid=$ProcessId; state=$state; summary.csv not created yet")
        }
        if ($null -eq $process) {
            break
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
} finally {
    $writer.Dispose()
}
