$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "DubVibe Python environment not found. Run npm run setup:backend first."
}

$ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
$ffprobe = Get-Command ffprobe.exe -ErrorAction SilentlyContinue
if (-not $ffmpeg -or -not $ffprobe) {
  throw "FFmpeg and FFprobe must be installed and available on PATH to build the desktop app."
}

$binaryDir = Join-Path $projectRoot "src-tauri\binaries"
$workDir = Join-Path $projectRoot "src-tauri\target\pyinstaller"
New-Item -ItemType Directory -Path $binaryDir -Force | Out-Null
New-Item -ItemType Directory -Path $workDir -Force | Out-Null

# resemblyzer's metadata declares the obsolete 'typing' backport as a
# dependency, so reinstalling packages can silently bring it back — and
# PyInstaller refuses to build while it is present. Remove it on every build.
$typingPkg = Join-Path $projectRoot ".venv\Lib\site-packages\typing"
if (Test-Path $typingPkg) {
  Write-Output "Removing obsolete 'typing' backport (incompatible with PyInstaller)..."
  & $python -m pip uninstall typing -y | Out-Null
}

$arguments = @(
  "--noconfirm",
  "--clean",
  "--onedir",
  "--name", "dubvibe-backend-complete",
  "--distpath", $binaryDir,
  "--workpath", $workDir,
  "--specpath", $workDir,
  "--collect-all", "faster_whisper",
  "--collect-all", "ctranslate2",
  "--collect-all", "av",
  "--collect-all", "onnxruntime",
  "--collect-all", "tokenizers",
  "--collect-all", "edge_tts",
  "--collect-all", "argostranslate",
  "--collect-all", "pyttsx3",
  "--collect-all", "demucs",
  "--collect-all", "resemblyzer",
  "--collect-all", "webrtcvad",
  "--collect-all", "einops",
  "--collect-all", "julius",
  "--collect-all", "lameenc",
  "--collect-all", "sphn",
  "--collect-all", "torch",
  # NVIDIA runtime DLLs so CTranslate2 (Faster-Whisper) can use CUDA on GPU PCs
  "--collect-binaries", "nvidia.cublas",
  "--collect-binaries", "nvidia.cudnn",
  "--collect-binaries", "nvidia.cuda_runtime",
  "--collect-all", "tqdm",
  "--collect-all", "llama_cpp",
  "--collect-all", "huggingface_hub",
  # Khmer MMS-VITS and NLLB are loaded lazily through transformers; the
  # tokenizers need sentencepiece, and the .bin checkpoints are read with soundfile/yaml.
  "--collect-all", "transformers",
  "--collect-all", "sentencepiece",
  "--collect-all", "soundfile",
  "--collect-all", "yaml",
  "--collect-all", "voxcpm",
  "--collect-all", "librosa",
  "--collect-all", "wetext",
  "--add-binary", "$($ffmpeg.Source);.",
  "--add-binary", "$($ffprobe.Source);.",
  (Join-Path $projectRoot "backend\run_server.py")
)

& $python -m PyInstaller @arguments
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed with exit code $LASTEXITCODE."
}

# Onedir layout: the exe plus its _internal folder travel together.
$sidecar = Join-Path $binaryDir "dubvibe-backend-complete\dubvibe-backend-complete.exe"
if (-not (Test-Path $sidecar)) {
  throw "PyInstaller finished without creating $sidecar."
}

# Ship the small Base transcription model inside the bundle so transcription
# works offline on first launch. Larger models are downloaded via the Model
# Manager into %LOCALAPPDATA%\DubVibe Studio\models.
$baseSource = Join-Path $env:LOCALAPPDATA "DubVibe Studio\models\faster-whisper\base"
$bundleModels = Join-Path (Split-Path $sidecar -Parent) "models\faster-whisper"
$baseTarget = Join-Path $bundleModels "base"
$baseSourceFlat = Test-Path (Join-Path $baseSource "model.bin")
$baseSourceNested = (Test-Path $baseSource) -and ((Get-ChildItem -Path $baseSource -Recurse -Filter "model.bin" | Measure-Object).Count -gt 0)
if ($baseSourceNested) {
  Remove-Item -Recurse -Force $baseTarget -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Path $bundleModels -Force | Out-Null
  Copy-Item -Recurse -Force $baseSource $baseTarget
  Write-Output "Bundled Base transcription model: $baseTarget"
} elseif (Test-Path $baseTarget) {
  Write-Output "Keeping previously bundled Base model: $baseTarget"
} else {
  Write-Warning "Base model not found in AppData - the app will download it on first transcription."
}

# Prune redundant heavy folders so the NSIS data file stays under the 32-bit
# makensis mmap limit (~1.7 GB). torch/lib keeps its CUDA DLLs (the loader
# chain needs them; unused at runtime without an NVIDIA GPU - the app falls
# back to CPU), but the pip-wheel duplicate nvidia/ folder and the CUDA
# llama.cpp backend are dead weight, as are torch's build-time headers.
$internal = Join-Path (Split-Path $sidecar -Parent) "_internal"
Remove-Item -Recurse -Force (Join-Path $internal "nvidia") -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $internal "llama_cpp\lib\ggml-cuda.dll") -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force (Join-Path $internal "torch\include") -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force (Join-Path $internal "torch\share") -ErrorAction SilentlyContinue
Write-Output "Pruned nvidia/, ggml-cuda.dll, torch headers from the bundle."

Write-Output "Bundled backend ready: $sidecar"
Write-Output "Bundle folder (ship as a whole): $(Split-Path $sidecar -Parent)"
