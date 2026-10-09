$ErrorActionPreference = "Stop"
$rustBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (Test-Path $rustBin) {
  $env:PATH = "$rustBin;$env:PATH"
}

npx tauri build
