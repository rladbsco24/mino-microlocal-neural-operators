param(
    [string[]]$Campaigns = @("proxy_sweep", "ablation", "stage3"),
    [string]$Seeds = "7,11,19",
    [string]$Device = "cuda",
    [int]$ProxyEpochs = 16,
    [int]$AblationEpochs = 16,
    [int]$Stage3Epochs = 24
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$LogDir = Join-Path $Root "generated\empirical_closure\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Location $Root

function Invoke-Campaign {
    param(
        [string]$Campaign,
        [int]$Epochs
    )

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logPath = Join-Path $LogDir ("resume_{0}_{1}.log" -f $Campaign, $timestamp)
    Write-Host ("[resume] campaign={0} epochs={1} seeds={2} log={3}" -f $Campaign, $Epochs, $Seeds, $logPath)

    $args = @(
        "scripts/run_mino_empirical_closure.py",
        "--campaign", $Campaign,
        "--epochs", "$Epochs",
        "--seeds", $Seeds,
        "--device", $Device,
        "--skip-existing"
    )

    & python @args 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "Campaign failed: $Campaign"
    }
}

foreach ($campaign in $Campaigns) {
    switch ($campaign) {
        "proxy_sweep" { Invoke-Campaign -Campaign $campaign -Epochs $ProxyEpochs }
        "ablation" { Invoke-Campaign -Campaign $campaign -Epochs $AblationEpochs }
        "stage3" { Invoke-Campaign -Campaign $campaign -Epochs $Stage3Epochs }
        default { throw "Unknown campaign: $campaign" }
    }
}

Write-Host "[resume] complete"
