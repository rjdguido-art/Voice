$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment in .venv ..."
    py -3 -m venv .venv
}

Write-Host "Upgrading pip ..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt ..."
& $VenvPython -m pip install -r requirements.txt

Write-Host "Setup complete."
Write-Host "Next step: run .\scripts\run.ps1"
