param(
    [switch]$Help,
    [switch]$AllModels,
    [string]$ModelsDir = "models"
)

$ErrorActionPreference = "Stop"

function Write-Usage {
    @"
Usage: .\scripts\download-models.ps1 [-AllModels] [-ModelsDir <path>]

Download MioTTS model files using the Hugging Face CLI (hf).

Options:
  -AllModels        Download all MioTTS LLM GGUF models from Aratako/MioTTS-GGUF.
                    Default behavior downloads only MioTTS-0.1B-Q8_0.gguf.
  -ModelsDir <path> Output directory for downloaded files (default: models).
"@ | Write-Output
}

if ($Help) {
    Write-Usage
    exit 0
}

$hfCmd = Get-Command hf -ErrorAction SilentlyContinue
if (-not $hfCmd) {
    Write-Error @"
'hf' command not found.
Set up the Hugging Face CLI first:
https://huggingface.co/docs/huggingface_hub/guides/cli
"@
    exit 1
}

New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null

Write-Output "Downloading codec model..."
& hf download mmnga-o/miotts-cpp-gguf miocodec.gguf --local-dir $ModelsDir

Write-Output "Downloading all voice embedding models..."
& hf download mmnga-o/miotts-cpp-gguf --include "*.emb.gguf" --local-dir $ModelsDir

if ($AllModels) {
    Write-Output "Downloading all MioTTS LLM GGUF models..."
    & hf download Aratako/MioTTS-GGUF --include "*.gguf" --local-dir $ModelsDir
}
else {
    Write-Output "Downloading required MioTTS LLM model (MioTTS-0.1B-Q8_0.gguf)..."
    & hf download Aratako/MioTTS-GGUF MioTTS-0.1B-Q8_0.gguf --local-dir $ModelsDir
}

Write-Output "Done. Models are in '$ModelsDir'."
