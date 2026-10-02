param(
    [string]$RepoRoot = "E:\zuo\projects\CEP",
    [string]$Suffix = (Get-Date -Format "yyyyMMdd_HHmmss"),
    [switch]$ResumeExisting,
    [switch]$SkipClean
)

$ErrorActionPreference = "Stop"

$repo = [System.IO.Path]::GetFullPath($RepoRoot)
$exe = Join-Path $repo "example_v2\build_gcp_control\Release\example.exe"
$problems = Join-Path $repo "PVL-BA-Bench\public_release\extracted\pvl-ba"
$dataset = "BA-problem-000144-i2823-p188150-o1073331-g23-c0"
$qualityRoot = Join-Path $problems "$dataset\quality"
$gcp = Join-Path $repo "gcp_splits\$dataset\gcp_13control_10checkpoint_uniform_xy.txt"
$tools = Join-Path $repo "tools"
$cleanOut = Join-Path $repo "d144_clean_singlethread_confirm_$Suffix"
$diagOut = Join-Path $repo "d144_diag_singlethread_confirm_$Suffix"
$reportDir = Join-Path $repo "d144_singlethread_confirm_$Suffix"
$logPath = Join-Path $reportDir "runner.log"
$oldClean = Join-Path $repo "d144_clean\summary.csv"
$oldDiag = Join-Path $repo "d144_diag\summary.csv"
$oldCheckpointDir = Join-Path $repo "d144_diag\absolute_accuracy"

function New-EmptyDirectory {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        $items = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue)
        if ($items.Count -gt 0) {
            throw "Refusing to overwrite non-empty directory: $Path"
        }
    } else {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Ensure-OutputDirectory {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        return
    }
    New-Item -ItemType Directory -Path $Path | Out-Null
}

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $logPath -Value "[$timestamp] $Message" -Encoding UTF8
}

function Quote-Arg {
    param([string]$Value)
    if ($Value -match "\s") {
        return '"' + $Value + '"'
    }
    return $Value
}

function Get-SummaryRows {
    param([string]$OutDir)
    $summary = Join-Path $OutDir "summary.csv"
    if (-not (Test-Path -LiteralPath $summary)) {
        return 0
    }
    return [Math]::Max(0, @((Get-Content -LiteralPath $summary)).Count - 1)
}

function Write-Manifest {
    param(
        [string]$OutDir,
        [string]$ModeLabel,
        [string]$CommandText
    )
    $lines = @(
        "started=$((Get-Date).ToString('o'))",
        "mode=$ModeLabel",
        "root=$problems",
        "dataset_filter=$dataset",
        "gcp=$gcp",
        "stdout=$(Join-Path $OutDir 'benchmark.stdout.log')",
        "stderr=$(Join-Path $OutDir 'benchmark.stderr.log')",
        "threads=1",
        "command=$CommandText"
    )
    Set-Content -LiteralPath (Join-Path $OutDir "run_manifest_singlethread.txt") -Value $lines -Encoding UTF8
}

function Run-D144 {
    param(
        [string]$Mode,
        [string]$OutDir
    )
    $stdout = Join-Path $OutDir "benchmark.stdout.log"
    $stderr = Join-Path $OutDir "benchmark.stderr.log"
    $args = @(
        "--problems", $problems,
        "--out", $OutDir,
        "--dataset", $dataset,
        "--mode", $Mode,
        "--use-gcp-control",
        "--gcp", $gcp
    )
    $argText = (($args | ForEach-Object { Quote-Arg $_ }) -join " ")
    $modeLabel = if ($Mode -eq "clean") { "clean-gcp-control-singlethread" } else { "diagnostic-gcp-control-singlethread" }
    $cmdText = '"' + $exe + '" ' + $argText
    Write-Manifest $OutDir $modeLabel $cmdText
    Write-Log "START $Mode out=$OutDir"

    $env:CEP_NUM_THREADS = "1"
    Remove-Item Env:CEP_STRICT_VECTOR_DIAGNOSTICS -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $exe `
        -ArgumentList $argText `
        -WorkingDirectory (Split-Path -Parent $exe) `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -LiteralPath (Join-Path $OutDir "process.pid") -Value ([string]$proc.Id) -Encoding ASCII

    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 60
        $proc.Refresh()
        $rows = Get-SummaryRows $OutDir
        $summary = Join-Path $OutDir "summary.csv"
        $lastWrite = if (Test-Path -LiteralPath $summary) {
            (Get-Item -LiteralPath $summary).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        } else {
            "n/a"
        }
        Write-Log "HEARTBEAT $Mode rows=$rows/112 last_write=$lastWrite pid=$($proc.Id)"
    }

    $proc.WaitForExit()
    $proc.Refresh()
    $exitCode = $proc.ExitCode
    Set-Content -LiteralPath (Join-Path $OutDir "exit_code.txt") -Value ([string]$exitCode) -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $OutDir "finished_at.txt") -Value (Get-Date).ToString("o") -Encoding UTF8
    $rowsFinal = Get-SummaryRows $OutDir
    Write-Log "FINISH $Mode exit=$exitCode rows=$rowsFinal/112 out=$OutDir"
    if ($null -eq $exitCode -and $rowsFinal -eq 112) {
        Write-Log "Treating empty ExitCode as success because summary has 112/112 rows."
        $exitCode = 0
        Set-Content -LiteralPath (Join-Path $OutDir "exit_code.txt") -Value "0" -Encoding ASCII
    }
    if ($exitCode -ne 0) {
        throw "$Mode exited with code $exitCode"
    }
}

function To-DoubleOrNull {
    param([string]$Value)
    $number = 0.0
    if ([double]::TryParse($Value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$number)) {
        return $number
    }
    return $null
}

function Abs-Diff {
    param($A, $B)
    if ($null -eq $A -and $null -eq $B) {
        return 0.0
    }
    if ($null -eq $A -or $null -eq $B) {
        return [double]::PositiveInfinity
    }
    if ([double]::IsNaN($A) -and [double]::IsNaN($B)) {
        return 0.0
    }
    return [Math]::Abs([double]$A - [double]$B)
}

function Compare-Summary {
    param(
        [string]$OldPath,
        [string]$NewPath,
        [string]$OutCsv
    )
    $oldRows = @(Import-Csv -LiteralPath $OldPath)
    $newRows = @(Import-Csv -LiteralPath $NewPath)
    $newByKey = @{}
    foreach ($row in $newRows) {
        $newByKey["$($row.quality_dataset)|$($row.method)"] = $row
    }

    $fields = @(
        "status",
        "initial_cost",
        "final_cost",
        "initial_rmse_px",
        "final_rmse_px",
        "iterations",
        "accepted_steps",
        "rejected_steps",
        "linear_solver_iterations",
        "initial_gradient_max_norm",
        "final_gradient_max_norm",
        "final_gradient_norm",
        "gradient_reduction_ratio_final",
        "reached_gradient_tolerance",
        "iterations_to_gradient_tolerance",
        "final_relative_function_decrease",
        "final_relative_step_size",
        "final_lm_gain_ratio",
        "final_gradient_lipschitz_estimate",
        "final_direction_quality",
        "termination_type"
    )
    $rows = foreach ($old in $oldRows) {
        $key = "$($old.quality_dataset)|$($old.method)"
        $new = $newByKey[$key]
        if ($null -eq $new) {
            [pscustomobject]@{
                quality_dataset = $old.quality_dataset
                method = $old.method
                field = "__row__"
                old_value = "present"
                new_value = "missing"
                abs_diff = ""
            }
            continue
        }
        foreach ($field in $fields) {
            $oldValue = [string]$old.$field
            $newValue = [string]$new.$field
            $oldNumber = To-DoubleOrNull $oldValue
            $newNumber = To-DoubleOrNull $newValue
            if ($null -ne $oldNumber -or $null -ne $newNumber) {
                $diff = Abs-Diff $oldNumber $newNumber
                [pscustomobject]@{
                    quality_dataset = $old.quality_dataset
                    method = $old.method
                    field = $field
                    old_value = $oldValue
                    new_value = $newValue
                    abs_diff = $diff
                }
            } elseif ($oldValue -ne $newValue) {
                [pscustomobject]@{
                    quality_dataset = $old.quality_dataset
                    method = $old.method
                    field = $field
                    old_value = $oldValue
                    new_value = $newValue
                    abs_diff = ""
                }
            }
        }
    }
    $rows | Export-Csv -LiteralPath $OutCsv -NoTypeInformation -Encoding UTF8
    return [pscustomobject]@{
        old_rows = $oldRows.Count
        new_rows = $newRows.Count
        compared_cells = @($rows).Count
        nonzero_cells = @($rows | Where-Object {
            ($_.abs_diff -eq "") -or ([double]$_.abs_diff -gt 0.0)
        }).Count
        max_abs_diff = (@($rows | Where-Object { $_.abs_diff -ne "" } | ForEach-Object { [double]$_.abs_diff }) | Measure-Object -Maximum).Maximum
        max_final_rmse_diff = (@($rows | Where-Object { $_.field -eq "final_rmse_px" } | ForEach-Object { [double]$_.abs_diff }) | Measure-Object -Maximum).Maximum
        max_final_cost_diff = (@($rows | Where-Object { $_.field -eq "final_cost" } | ForEach-Object { [double]$_.abs_diff }) | Measure-Object -Maximum).Maximum
    }
}

function Compare-Convergence {
    param(
        [string]$OldSummaryPath,
        [string]$NewSummaryPath,
        [string]$OutCsv
    )
    $oldRows = @(Import-Csv -LiteralPath $OldSummaryPath)
    $newRows = @(Import-Csv -LiteralPath $NewSummaryPath)
    $newByKey = @{}
    foreach ($row in $newRows) {
        $newByKey["$($row.quality_dataset)|$($row.method)"] = $row
    }

    $outRows = foreach ($old in $oldRows) {
        $key = "$($old.quality_dataset)|$($old.method)"
        $new = $newByKey[$key]
        $oldConv = Join-Path (Split-Path -Parent $old.report) "convergence.txt"
        $newConv = if ($null -ne $new) { Join-Path (Split-Path -Parent $new.report) "convergence.txt" } else { "" }
        if ($null -eq $new -or -not (Test-Path -LiteralPath $oldConv) -or -not (Test-Path -LiteralPath $newConv)) {
            [pscustomobject]@{
                quality_dataset = $old.quality_dataset
                method = $old.method
                old_rows = ""
                new_rows = ""
                max_abs_diff = ""
                differing_cells = ""
                issue = "missing_convergence"
            }
            continue
        }
        $oldConvRows = @(Import-Csv -LiteralPath $oldConv)
        $newConvRows = @(Import-Csv -LiteralPath $newConv)
        $maxDiff = 0.0
        $differing = 0
        $issue = ""
        if ($oldConvRows.Count -ne $newConvRows.Count) {
            $issue = "row_count"
        }
        $n = [Math]::Min($oldConvRows.Count, $newConvRows.Count)
        for ($i = 0; $i -lt $n; $i++) {
            foreach ($field in $oldConvRows[$i].PSObject.Properties.Name) {
                $a = To-DoubleOrNull ([string]$oldConvRows[$i].$field)
                $b = To-DoubleOrNull ([string]$newConvRows[$i].$field)
                if ($null -ne $a -or $null -ne $b) {
                    $diff = Abs-Diff $a $b
                    if ($diff -gt $maxDiff) {
                        $maxDiff = $diff
                    }
                    if ($diff -gt 0.0) {
                        $differing++
                    }
                } elseif ([string]$oldConvRows[$i].$field -ne [string]$newConvRows[$i].$field) {
                    $differing++
                    $issue = "string_diff"
                }
            }
        }
        [pscustomobject]@{
            quality_dataset = $old.quality_dataset
            method = $old.method
            old_rows = $oldConvRows.Count
            new_rows = $newConvRows.Count
            max_abs_diff = $maxDiff
            differing_cells = $differing
            issue = $issue
        }
    }
    $outRows | Export-Csv -LiteralPath $OutCsv -NoTypeInformation -Encoding UTF8
    return [pscustomobject]@{
        files_compared = @($outRows).Count
        files_with_issue = @($outRows | Where-Object { $_.issue -ne "" }).Count
        files_with_diff = @($outRows | Where-Object { $_.differing_cells -ne "" -and [int]$_.differing_cells -gt 0 }).Count
        max_abs_diff = (@($outRows | Where-Object { $_.max_abs_diff -ne "" } | ForEach-Object { [double]$_.max_abs_diff }) | Measure-Object -Maximum).Maximum
    }
}

function Compare-Checkpoint {
    param(
        [string]$OldCsv,
        [string]$NewCsv,
        [string]$OutCsv
    )
    $oldRows = @(Import-Csv -LiteralPath $OldCsv)
    $newRows = @(Import-Csv -LiteralPath $NewCsv)
    $newByKey = @{}
    foreach ($row in $newRows) {
        $newByKey[$row.method] = $row
    }
    $fields = @(
        "mean_final_rmse_px",
        "checkpoint_count",
        "checkpoint_observation_count",
        "checkpoint_rmse_horizontal_m",
        "checkpoint_rmse_z_m",
        "checkpoint_rmse_3d_m",
        "checkpoint_reproj_rmse_px"
    )
    $outRows = foreach ($old in $oldRows) {
        $new = $newByKey[$old.method]
        if ($null -eq $new) {
            [pscustomobject]@{
                method = $old.method
                field = "__row__"
                old_value = "present"
                new_value = "missing"
                abs_diff = ""
            }
            continue
        }
        foreach ($field in $fields) {
            $oldNumber = To-DoubleOrNull ([string]$old.$field)
            $newNumber = To-DoubleOrNull ([string]$new.$field)
            [pscustomobject]@{
                method = $old.method
                field = $field
                old_value = [string]$old.$field
                new_value = [string]$new.$field
                abs_diff = (Abs-Diff $oldNumber $newNumber)
            }
        }
    }
    $outRows | Export-Csv -LiteralPath $OutCsv -NoTypeInformation -Encoding UTF8
    return [pscustomobject]@{
        methods_old = $oldRows.Count
        methods_new = $newRows.Count
        compared_cells = @($outRows).Count
        nonzero_cells = @($outRows | Where-Object { $_.abs_diff -eq "" -or [double]$_.abs_diff -gt 0.0 }).Count
        max_abs_diff = (@($outRows | Where-Object { $_.abs_diff -ne "" } | ForEach-Object { [double]$_.abs_diff }) | Measure-Object -Maximum).Maximum
        max_rmse3d_diff = (@($outRows | Where-Object { $_.field -eq "checkpoint_rmse_3d_m" } | ForEach-Object { [double]$_.abs_diff }) | Measure-Object -Maximum).Maximum
    }
}

foreach ($path in @($exe, $problems, $qualityRoot, $gcp, $oldClean, $oldDiag)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path not found: $path"
    }
}

if ($ResumeExisting) {
    Ensure-OutputDirectory $reportDir
    Ensure-OutputDirectory $cleanOut
    Ensure-OutputDirectory $diagOut
} else {
    New-EmptyDirectory $reportDir
    New-EmptyDirectory $cleanOut
    New-EmptyDirectory $diagOut
}
if (-not (Test-Path -LiteralPath $logPath)) {
    Set-Content -LiteralPath $logPath -Value @(
        "D144 single-thread confirmation",
        "suffix=$Suffix",
        "clean_out=$cleanOut",
        "diag_out=$diagOut",
        "report_dir=$reportDir"
    ) -Encoding UTF8
}

try {
    Write-Log "Runner started"
    if ($SkipClean) {
        $cleanRows = Get-SummaryRows $cleanOut
        if ($cleanRows -ne 112) {
            throw "SkipClean requested but clean summary has $cleanRows/112 rows: $cleanOut"
        }
        Write-Log "Skipping clean; existing clean summary has 112/112 rows."
    } else {
        Run-D144 "clean" $cleanOut
    }
    Run-D144 "diagnostic" $diagOut

    $newCheckpointDir = Join-Path $diagOut "absolute_accuracy"
    New-EmptyDirectory $newCheckpointDir
    $checkpointScript = Join-Path $tools "checkpoint_absolute_accuracy_batch.py"
    Write-Log "START checkpoint absolute accuracy"
    & python $checkpointScript --diagnostic-root $diagOut --quality-root $qualityRoot --gcp $gcp --out-dir $newCheckpointDir 1>> (Join-Path $reportDir "checkpoint_stdout.log") 2>> (Join-Path $reportDir "checkpoint_stderr.log")
    if ($LASTEXITCODE -ne 0) {
        throw "checkpoint_absolute_accuracy_batch.py exited with code $LASTEXITCODE"
    }
    Write-Log "FINISH checkpoint absolute accuracy"

    Write-Log "START comparisons"
    $cleanStats = Compare-Summary $oldClean (Join-Path $cleanOut "summary.csv") (Join-Path $reportDir "clean_summary_compare.csv")
    $diagStats = Compare-Summary $oldDiag (Join-Path $diagOut "summary.csv") (Join-Path $reportDir "diag_summary_compare.csv")
    $convStats = Compare-Convergence $oldDiag (Join-Path $diagOut "summary.csv") (Join-Path $reportDir "diag_convergence_compare.csv")
    $checkpointStats = Compare-Checkpoint `
        (Join-Path $oldCheckpointDir "checkpoint_review_table.csv") `
        (Join-Path $newCheckpointDir "checkpoint_review_table.csv") `
        (Join-Path $reportDir "checkpoint_review_compare.csv")

    $report = @(
        "# D144 Single-Thread Confirmation",
        "",
        "- Completed: $((Get-Date).ToString('o'))",
        "- Clean output: ``$cleanOut``",
        "- Diagnostic output: ``$diagOut``",
        "- Report directory: ``$reportDir``",
        "- Thread setting: ``CEP_NUM_THREADS=1``",
        "",
        "## Summary Comparison",
        "",
        "| Scope | Old rows | New rows | Compared cells | Nonzero cells | Max abs diff | Max final RMSE diff | Max final cost diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| clean | $($cleanStats.old_rows) | $($cleanStats.new_rows) | $($cleanStats.compared_cells) | $($cleanStats.nonzero_cells) | $($cleanStats.max_abs_diff) | $($cleanStats.max_final_rmse_diff) | $($cleanStats.max_final_cost_diff) |",
        "| diagnostic | $($diagStats.old_rows) | $($diagStats.new_rows) | $($diagStats.compared_cells) | $($diagStats.nonzero_cells) | $($diagStats.max_abs_diff) | $($diagStats.max_final_rmse_diff) | $($diagStats.max_final_cost_diff) |",
        "",
        "## Diagnostic Convergence Comparison",
        "",
        "- Files compared: $($convStats.files_compared)",
        "- Files with issue: $($convStats.files_with_issue)",
        "- Files with numeric/string differences: $($convStats.files_with_diff)",
        "- Max absolute numeric difference: $($convStats.max_abs_diff)",
        "",
        "## Checkpoint Review Comparison",
        "",
        "- Methods old/new: $($checkpointStats.methods_old) / $($checkpointStats.methods_new)",
        "- Compared cells: $($checkpointStats.compared_cells)",
        "- Nonzero cells: $($checkpointStats.nonzero_cells)",
        "- Max absolute difference: $($checkpointStats.max_abs_diff)",
        "- Max checkpoint RMSE3D difference: $($checkpointStats.max_rmse3d_diff)",
        "",
        "Detailed CSV files:",
        "",
        '- `clean_summary_compare.csv`',
        '- `diag_summary_compare.csv`',
        '- `diag_convergence_compare.csv`',
        '- `checkpoint_review_compare.csv`'
    )
    Set-Content -LiteralPath (Join-Path $reportDir "comparison_report.md") -Value $report -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $reportDir "DONE.txt") -Value (Get-Date).ToString("o") -Encoding UTF8
    Write-Log "Runner completed"
} catch {
    Write-Log ("FAILED: " + $_.Exception.Message)
    Set-Content -LiteralPath (Join-Path $reportDir "FAILED.txt") -Value @(
        (Get-Date).ToString("o"),
        $_.Exception.Message,
        $_.ScriptStackTrace
    ) -Encoding UTF8
    throw
}
