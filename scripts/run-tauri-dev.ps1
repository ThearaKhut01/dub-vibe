$ErrorActionPreference = "Stop"
$rustBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (Test-Path $rustBin) {
  $env:PATH = "$rustBin;$env:PATH"
}

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $projectPython) {
  $env:PYTHON = $projectPython
} else {
  throw "DubVibe Python environment not found. Run npm run setup:backend first."
}

npx tauri dev
