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
$BundledFfmpegReady = (Test-Path $FfmpegExe) -and (Test-Path $FfprobeExe)
if (-not $BundledFfmpegReady) {
    $SystemFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    $SystemFfprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
    if (-not $SystemFfmpeg -or -not $SystemFfprobe) {
        Write-Warning "ffmpeg and/or ffprobe were not found."
        Write-Warning "Add ffmpeg+ffprobe to PATH or place portable binaries in .\ffmpeg\bin."
    }
}

& $VenvPython app.py
