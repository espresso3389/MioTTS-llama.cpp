#!/usr/bin/env bash
set -euo pipefail

ALL_MODELS=0
MODELS_DIR="models"

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Download MioTTS model files using the Hugging Face CLI (hf).

Options:
  --all-models        Download all MioTTS LLM GGUF models from Aratako/MioTTS-GGUF.
                      Default behavior downloads only MioTTS-0.1B-Q8_0.gguf.
  --models-dir DIR    Output directory for downloaded files (default: models).
  -h, --help          Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-models)
      ALL_MODELS=1
      shift
      ;;
    --models-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --models-dir requires a value." >&2
        exit 1
      fi
      MODELS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if ! command -v hf >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Error: 'hf' command not found.
Set up the Hugging Face CLI first:
https://huggingface.co/docs/huggingface_hub/guides/cli
MSG
  exit 1
fi

mkdir -p "$MODELS_DIR"

echo "Downloading codec model..."
hf download mmnga-o/miotts-cpp-gguf miocodec.gguf --local-dir "$MODELS_DIR"

echo "Downloading all voice embedding models..."
hf download mmnga-o/miotts-cpp-gguf --include "*.emb.gguf" --local-dir "$MODELS_DIR"

if [[ "$ALL_MODELS" -eq 1 ]]; then
  echo "Downloading all MioTTS LLM GGUF models..."
  hf download Aratako/MioTTS-GGUF --include "*.gguf" --local-dir "$MODELS_DIR"
else
  echo "Downloading required MioTTS LLM model (MioTTS-0.1B-Q8_0.gguf)..."
  hf download Aratako/MioTTS-GGUF MioTTS-0.1B-Q8_0.gguf --local-dir "$MODELS_DIR"
fi

echo "Done. Models are in '$MODELS_DIR'."
