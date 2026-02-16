$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Run .\scripts\setup.ps1 first."
    exit 1
}

$FfmpegExe = Join-Path $ProjectRoot "ffmpeg\bin\ffmpeg.exe"
$FfprobeExe = Join-Path $ProjectRoot "ffmpeg\bin\ffprobe.exe"
if (-not (Test-Path $FfmpegExe) -or -not (Test-Path $FfprobeExe)) {
    Write-Warning "ffmpeg.exe and/or ffprobe.exe are missing in .\ffmpeg\bin."
    Write-Warning "The app can start, but transcription will fail until portable ffmpeg is added."
}

& $VenvPython app.py
