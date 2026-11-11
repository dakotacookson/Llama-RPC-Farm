# Llama RPC Farm — one-line Windows source installer:
#   irm https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/install-remote.ps1 | iex
$ErrorActionPreference = "Stop"

$Repo = $env:LLAMA_FARM_REPO
if (-not $Repo) { $Repo = "https://github.com/dakotacookson/Llama-RPC-Farm.git" }
$Dest = $env:LLAMA_FARM_DIR
if (-not $Dest) { $Dest = Join-Path $HOME "Llama-RPC-Farm" }

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "Llama RPC Farm needs git. Install Git for Windows, then run this again."
}

if (Test-Path (Join-Path $Dest ".git")) {
  Write-Host "==> Updating existing checkout at $Dest"
  git -C $Dest pull --ff-only
} else {
  if ((Test-Path $Dest) -and (Get-ChildItem -Force $Dest | Select-Object -First 1)) {
    throw "$Dest exists and is not an empty directory. Move it aside or set LLAMA_FARM_DIR."
  }
  Write-Host "==> Cloning Llama RPC Farm into $Dest"
  git clone --depth 1 $Repo $Dest
}

Set-Location $Dest
& .\install.bat
