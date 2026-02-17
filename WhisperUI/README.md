# Echo by Concentrix

A Windows-friendly local transcription app using Gradio + OpenAI Whisper.

## Download Windows EXE (GitHub Actions)

1. Open the repo on GitHub and go to **Actions**.
2. Run workflow **Build Echo Windows EXE** (or push changes under `WhisperUI/` to trigger it).
3. Open the workflow run and download artifact **Echo-windows-x64**.
4. Extract the zip and run `Echo.exe`.

## Project structure

```text
WhisperUI/
  app.py
  requirements.txt
  README.md
  scripts/
    setup.ps1
    run.ps1
  outputs/            (created at runtime if missing)
  ffmpeg/
    bin/
      ffmpeg.exe
      ffprobe.exe
```

## Setup (no admin rights required)

1. Open **PowerShell**.
2. Go to the project folder:
   ```powershell
   cd .\WhisperUI
   ```
3. If FFmpeg is not already on your PATH, put portable FFmpeg binaries in:
   - `.\ffmpeg\bin\ffmpeg.exe`
   - `.\ffmpeg\bin\ffprobe.exe`
4. Run setup script:
   ```powershell
   .\scripts\setup.ps1
   ```

This creates a local virtual environment in `.\.venv` and installs dependencies using `.\.venv\Scripts\python.exe` directly (no need to activate the venv).

## Run locally

```powershell
.\scripts\run.ps1
```

Then open the local Gradio URL shown in PowerShell (usually `http://127.0.0.1:7860`).

### UI controls

- Audio upload (filepath passed to backend)
- Model: `tiny`, `base`, `small`, `medium`, `large`
- Language textbox (default `es`)
- Auto-detect language checkbox (uses `language=None`)
- Word timestamps checkbox (enables Whisper `word_timestamps`)
- Output format: `txt`, `srt`, `vtt`, `all` (saves all three)
- Output directory (default `./outputs`)
- Model cache directory (optional, default empty)
- Button: **Pre-download model** to download and cache selected model weights in advance
- Transcript box includes a built-in **Copy** button for clipboard copy

The app saves outputs as `<original_name>_<YYYYMMDD_HHMMSS>.<ext>` in the selected output directory.
When `all` is selected, it writes `.txt`, `.srt`, and `.vtt`.
Processing time is shown in the status panel.

## Troubleshooting

### "ffmpeg not found"

- If you see an error about missing FFmpeg, either add `ffmpeg` + `ffprobe` to PATH, or place portable binaries at `./ffmpeg/bin/ffmpeg.exe` and `./ffmpeg/bin/ffprobe.exe`.
- Confirm these files exist:
  - `.\ffmpeg\bin\ffmpeg.exe`
  - `.\ffmpeg\bin\ffprobe.exe`
- Re-run:
  ```powershell
  .\scripts\run.ps1
  ```

### Model download on first run

- Whisper downloads the selected model weights the first time you use that model.
- This can take time and requires internet access once per model.
- Use **Pre-download model** to cache weights before transcription.
- After the model is pre-downloaded, transcription works fully offline.

### PATH notes

- You can use either system FFmpeg on PATH or portable FFmpeg in `.\ffmpeg\bin`.
- `app.py` prepends `.\ffmpeg\bin` to PATH at runtime when bundled FFmpeg is present.
- Scripts call `.\.venv\Scripts\python.exe` explicitly, so PowerShell execution policy restrictions on `Activate.ps1` do not block usage.
