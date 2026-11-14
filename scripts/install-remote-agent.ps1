# Llama RPC Farm — one-line Windows peer-agent installer:
#   irm https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/install-remote-agent.ps1 | iex
$ErrorActionPreference = "Stop"

$Repo = if ($env:LLAMA_FARM_REPO) { $env:LLAMA_FARM_REPO } else { "dakotacookson/Llama-RPC-Farm" }
$api = "https://api.github.com/repos/$Repo/releases?per_page=1"
$release = Invoke-RestMethod -Uri $api -Headers @{ "User-Agent" = "llama-farm-installer"; "Accept" = "application/vnd.github+json" }
$asset = $release[0].assets | Where-Object {
  $_.name -like "llama-peer-agent_*_windows-x64-setup.exe"
} | Select-Object -First 1
if (-not $asset) { throw "No Windows peer-agent installer found in the latest release." }

$tmp = Join-Path $env:TEMP "llama-peer-agent-setup.exe"
Write-Host "==> Downloading $($asset.name)"
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $tmp
Write-Host "==> Installing the peer agent"
Start-Process -FilePath $tmp -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait
Remove-Item $tmp -Force -ErrorAction SilentlyContinue
Write-Host "Peer agent installed. The farm app will pair it automatically when it discovers this host."
