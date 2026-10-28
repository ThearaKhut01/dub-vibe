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

# llama.cpp must be a CPU build for the bundle. A CUDA-linked llama.dll cannot
# load on a machine without the NVIDIA driver (Qwen dies before translating),
# and deleting ggml-cuda.dll is not a fix: ggml.dll imports it directly, which
# breaks loading on every PC. Rebuild from source with CUDA off when the venv
# still holds a CUDA wheel; NVIDIA machines get GPU support at runtime from the
# optional GPU Acceleration Pack instead.
$llamaLib = Join-Path $projectRoot ".venv\Lib\site-packages\llama_cpp\lib"
if (Test-Path (Join-Path $llamaLib "ggml-cuda.dll")) {
  $llamaVersion = (& $python -c "import importlib.metadata; print(importlib.metadata.version('llama-cpp-python'))").Trim()
  if (-not $llamaVersion) { throw "llama-cpp-python is not installed in the build venv." }
  Write-Output "Building CPU-only llama.cpp $llamaVersion for the bundle (a few minutes)..."
  $previousCmakeArgs = $env:CMAKE_ARGS
  $previousForceCmake = $env:FORCE_CMAKE
  $env:CMAKE_ARGS = "-DGGML_CUDA=OFF -DGGML_NATIVE=ON"
  $env:FORCE_CMAKE = "1"
  try {
    & $python -m pip install --force-reinstall --no-deps --no-binary llama-cpp-python "llama-cpp-python==$llamaVersion"
    if ($LASTEXITCODE -ne 0) { throw "CPU llama.cpp build failed (pip exit $LASTEXITCODE)." }
  } finally {
    $env:CMAKE_ARGS = $previousCmakeArgs
    $env:FORCE_CMAKE = $previousForceCmake
  }
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
# back to CPU), but the pip-wheel duplicate nvidia/ folder is dead weight, as
# are torch's build-time headers. llama.cpp is already a CPU build (see above),
# so there is no CUDA backend to delete here.
$internal = Join-Path (Split-Path $sidecar -Parent) "_internal"
Remove-Item -Recurse -Force (Join-Path $internal "nvidia") -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force (Join-Path $internal "torch\include") -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force (Join-Path $internal "torch\share") -ErrorAction SilentlyContinue
Write-Output "Pruned nvidia/, torch headers from the bundle."

Write-Output "Bundled backend ready: $sidecar"
Write-Output "Bundle folder (ship as a whole): $(Split-Path $sidecar -Parent)"
