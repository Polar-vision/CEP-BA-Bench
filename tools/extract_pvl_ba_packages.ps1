param(
    [string]$PackageRoot = "E:\zuo\projects\CEP\PVL-BA-Bench\public_release\packages\pvl-ba",
    [string]$OutputRoot = "E:\zuo\projects\CEP\PVL-BA-Bench\public_release\extracted\pvl-ba",
    [int]$ThrottleLimit = 4,
    [int]$Limit = 0,
    [string]$Dataset = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$packageRootPath = [System.IO.Path]::GetFullPath($PackageRoot)
$outputRootPath = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not (Test-Path -LiteralPath $packageRootPath -PathType Container)) {
    throw "Package root does not exist: $packageRootPath"
}

New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null
$archives = @(
    Get-ChildItem -LiteralPath $packageRootPath -Recurse -Filter *.zip -File |
        Sort-Object FullName
)
if ($Dataset) {
    $archives = @(
        $archives |
            Where-Object {
                [System.IO.Path]::GetFileNameWithoutExtension($_.Name) -eq $Dataset
            }
    )
}
if ($Limit -gt 0) {
    $archives = @($archives | Select-Object -First $Limit)
}
if ($archives.Count -eq 0) {
    throw "No ZIP packages found under $packageRootPath"
}

$startedAt = Get-Date
Write-Host "Packages: $($archives.Count)"
Write-Host "Output:   $outputRootPath"
Write-Host "Workers:  $ThrottleLimit"

$results = @(
    $archives | ForEach-Object -Parallel {
        $archive = $_
        $outputRoot = $using:outputRootPath
        $dataset = [System.IO.Path]::GetFileNameWithoutExtension($archive.Name)
        $datasetRoot = Join-Path $outputRoot $dataset
        $marker = Join-Path $datasetRoot ".extracted.ok.json"
        $category = Split-Path $archive.DirectoryName -Leaf
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $extractor = ""

        try {
            if (
                (Test-Path -LiteralPath $marker -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $datasetRoot "original") -PathType Container) -and
                (Test-Path -LiteralPath (Join-Path $datasetRoot "quality") -PathType Container)
            ) {
                $qualityCount = @(
                    Get-ChildItem -LiteralPath (Join-Path $datasetRoot "quality") -Directory
                ).Count
                [PSCustomObject]@{
                    category = $category
                    dataset = $dataset
                    archive = $archive.FullName
                    status = "skipped"
                    elapsed_sec = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
                    quality_count = $qualityCount
                    error = ""
                }
                return
            }

            $resolvedOutput = [System.IO.Path]::GetFullPath($outputRoot)
            $resolvedDataset = [System.IO.Path]::GetFullPath($datasetRoot)
            $outputPrefix = $resolvedOutput.TrimEnd(
                [System.IO.Path]::DirectorySeparatorChar,
                [System.IO.Path]::AltDirectorySeparatorChar
            ) + [System.IO.Path]::DirectorySeparatorChar
            if (-not $resolvedDataset.StartsWith(
                $outputPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
                throw "Refusing to clean dataset outside output root: $resolvedDataset"
            }
            if (Test-Path -LiteralPath $resolvedDataset) {
                Remove-Item -LiteralPath $resolvedDataset -Recurse -Force
            }

            $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
            $startInfo.FileName = "tar.exe"
            $startInfo.UseShellExecute = $false
            $startInfo.CreateNoWindow = $true
            $startInfo.RedirectStandardOutput = $true
            $startInfo.RedirectStandardError = $true
            [void]$startInfo.ArgumentList.Add("-xf")
            [void]$startInfo.ArgumentList.Add($archive.FullName)
            [void]$startInfo.ArgumentList.Add("-C")
            [void]$startInfo.ArgumentList.Add($outputRoot)
            $process = [System.Diagnostics.Process]::new()
            $process.StartInfo = $startInfo
            [void]$process.Start()
            $standardOutput = $process.StandardOutput.ReadToEnd()
            $standardError = $process.StandardError.ReadToEnd()
            $process.WaitForExit()

            if ($process.ExitCode -eq 0) {
                $extractor = "tar"
            }
            else {
                if (Test-Path -LiteralPath $resolvedDataset) {
                    Remove-Item -LiteralPath $resolvedDataset -Recurse -Force
                }
                Add-Type -AssemblyName System.IO.Compression.FileSystem
                [System.IO.Compression.ZipFile]::ExtractToDirectory(
                    $archive.FullName,
                    $outputRoot,
                    $true
                )
                $extractor = "dotnet"
            }

            $originalRoot = Join-Path $datasetRoot "original"
            $qualityRoot = Join-Path $datasetRoot "quality"
            if (-not (Test-Path -LiteralPath $originalRoot -PathType Container)) {
                throw "Missing original directory after extraction"
            }
            if (-not (Test-Path -LiteralPath $qualityRoot -PathType Container)) {
                throw "Missing quality directory after extraction"
            }
            $qualityCount = @(Get-ChildItem -LiteralPath $qualityRoot -Directory).Count
            if ($qualityCount -eq 0) {
                throw "No quality datasets found after extraction"
            }

            $markerData = [ordered]@{
                dataset = $dataset
                category = $category
                archive = $archive.FullName
                archive_bytes = $archive.Length
                quality_count = $qualityCount
                completed_at = (Get-Date).ToString("o")
            }
            $markerData |
                ConvertTo-Json |
                Set-Content -LiteralPath $marker -Encoding UTF8

            [PSCustomObject]@{
                category = $category
                dataset = $dataset
                archive = $archive.FullName
                status = "extracted"
                elapsed_sec = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
                quality_count = $qualityCount
                extractor = $extractor
                error = ""
            }
        }
        catch {
            [PSCustomObject]@{
                category = $category
                dataset = $dataset
                archive = $archive.FullName
                status = "failed"
                elapsed_sec = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
                quality_count = 0
                extractor = $extractor
                error = $_.Exception.Message
            }
        }
        finally {
            $stopwatch.Stop()
        }
    } -ThrottleLimit $ThrottleLimit
)

$logPath = Join-Path $outputRootPath "extraction_log.csv"
$results |
    Sort-Object category, dataset |
    Export-Csv -LiteralPath $logPath -NoTypeInformation -Encoding UTF8

$failed = @($results | Where-Object status -eq "failed")
$extracted = @($results | Where-Object status -eq "extracted")
$skipped = @($results | Where-Object status -eq "skipped")
$elapsed = (Get-Date) - $startedAt

Write-Host "Extracted: $($extracted.Count)"
Write-Host "Skipped:   $($skipped.Count)"
Write-Host "Failed:    $($failed.Count)"
Write-Host "Elapsed:   $([math]::Round($elapsed.TotalMinutes, 2)) min"
Write-Host "Log:       $logPath"

if ($failed.Count -gt 0) {
    $failed | Select-Object category, dataset, error | Format-Table -AutoSize
    exit 1
}
