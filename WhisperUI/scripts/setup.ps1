$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
$PythonExe = Get-Command python -ErrorAction SilentlyContinue

if (-not $PythonLauncher -and -not $PythonExe) {
    Write-Error "Python was not found. Install Python 3, then rerun .\scripts\setup.ps1."
    exit 1
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment in .venv ..."
    if ($PythonLauncher) {
        & py -3 -m venv .venv
    }
    else {
        & python -m venv .venv
    }
}

Write-Host "Upgrading pip ..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt ..."
& $VenvPython -m pip install -r requirements.txt

Write-Host "Setup complete."
Write-Host "Next step: run .\scripts\run.ps1"
