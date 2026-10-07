param(
    [string]$PythonExe = "",
    [switch]$NoShowcase,
    [switch]$NoInteractive,
    [switch]$AllowDownload,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$demoScript = Join-Path $repoRoot "demo\try_gujinbridge.py"

if (-not $PythonExe) {
    $gujinBridgePython = "E:\Anaconda\envs\GujinBridge\python.exe"
    if (Test-Path -LiteralPath $gujinBridgePython) {
        $PythonExe = $gujinBridgePython
    }
    else {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }
}

$scriptArgs = @($demoScript)
if (-not $AllowDownload) {
    # 当前发布候选为 V5 merged + DPO V1 LoRA，默认离线可避免无意义的联网重试。
    $scriptArgs += "--local-files-only"
}
if ($NoShowcase) {
    $scriptArgs += "--no-showcase"
}
if ($NoInteractive) {
    $scriptArgs += "--no-interactive"
}
if ($ExtraArgs) {
    $scriptArgs += $ExtraArgs
}

Write-Host "使用 Python：$PythonExe"
& $PythonExe @scriptArgs
exit $LASTEXITCODE
