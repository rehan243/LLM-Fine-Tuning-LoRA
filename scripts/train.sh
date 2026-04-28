#!/usr/bin/env bash
# one entrypoint for lora runs so folks stop copy pasting wandb keys in slack
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
  echo "usage: $0 [--gpu ID] [--dry-run]"
  echo "  expects TRAIN_DATA and OUTPUT_DIR in env or defaults below"
}

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo -e "${RED}bad arg${NC} $1"; usage; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TRAIN_DATA="${TRAIN_DATA:-data/train.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/lora_out}"

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo -e "${YELLOW}warning:${NC} $TRAIN_DATA missing, training will likely fail"
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo -e "${YELLOW}WANDB_API_KEY not set; offline logging only${NC}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

echo -e "${GREEN}gpu${NC} $GPU_ID"
echo -e "${GREEN}output${NC} $OUTPUT_DIR"

if [[ "$DRY" -eq 1 ]]; then
  echo -e "${YELLOW}dry run; not invoking trainer${NC}"; exit 0
fi

mkdir -p "$OUTPUT_DIR"

python -m train_lora \
  --config configs/lora.yaml \
  --train_data "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --report_to wandb

echo -e "${GREEN}training finished${NC}"
