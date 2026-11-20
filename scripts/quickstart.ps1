# One-command solo quickstart for Windows.
#
#   irm https://raw.githubusercontent.com/dakotacookson/Llama-RPC-Farm/main/scripts/quickstart.ps1 | iex
#
# Existing checkout is reused. Extra arguments are passed to src/quickstart.py.
$ErrorActionPreference = "Stop"
$Repo = if ($env:LLAMA_FARM_REPO) { $env:LLAMA_FARM_REPO } else { "https://github.com/dakotacookson/Llama-RPC-Farm.git" }
$Dest = if ($env:LLAMA_FARM_DIR) { $env:LLAMA_FARM_DIR } else { Join-Path $env:USERPROFILE "Llama-RPC-Farm" }
$AppUrl = if ($env:FARM_URL) { $env:FARM_URL } else { "http://127.0.0.1:8199" }

function Say($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

if (Test-Path (Join-Path $Dest ".git")) {
  Say "Using existing checkout: $Dest"
} else {
  Say "Cloning Llama RPC Farm into $Dest"
  git clone --depth 1 $Repo $Dest
}
Set-Location $Dest

$Py = Join-Path $Dest ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  Say "Installing the farm app (one time)"
  & (Join-Path $Dest "install.bat")
}

try {
  Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "$AppUrl/api/machines" | Out-Null
} catch {
  Say "Starting the farm UI"
  Start-Process -FilePath (Join-Path $Dest "run.bat") -WindowStyle Hidden
  for ($i = 0; $i -lt 60; $i++) {
    try {
      Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "$AppUrl/api/machines" | Out-Null
      break
    } catch { Start-Sleep -Seconds 1 }
  }
}

Say "Running quickstart"
$ArgsList = @()
if ($args) { $ArgsList = $args }
& $Py (Join-Path $Dest "src\quickstart.py") @ArgsList
