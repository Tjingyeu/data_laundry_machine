#!/bin/bash
#
# LoRA Chunk Router - Full Pipeline
#
# This script runs the complete pipeline:
#   1. Preprocess - Parse transcripts into chunks and generate training data
#   2. Train LoRA - Fine-tune Qwen2.5-3B with LoRA
#   3. Build Vector Index - Build FAISS index with BGE-small embeddings
#   4. Evaluate - Compare LoRA vs Vector search
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default config
CONFIG="${CONFIG:-configs/config.yaml}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Parse arguments
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_VECTOR=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-vector)
            SKIP_VECTOR=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --skip-preprocess   Skip preprocessing step"
            echo "  --skip-train        Skip LoRA training step"
            echo "  --skip-vector       Skip vector index building"
            echo "  --skip-eval         Skip evaluation"
            echo "  --config FILE      Use config FILE (default: configs/config.yaml)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log "Starting LoRA Chunk Router Pipeline"
log "Config: $CONFIG"

# Step 1: Preprocess
if [ "$SKIP_PREPROCESS" = false ]; then
    log "=========================================="
    log "Step 1: Preprocessing transcripts"
    log "=========================================="

    python -m src.preprocess --config "$CONFIG"

    if [ $? -ne 0 ]; then
        error "Preprocessing failed!"
        exit 1
    fi

    log "Preprocessing complete!"
else
    warn "Skipping preprocessing (--skip-preprocess)"
fi

# Step 2: Train LoRA
if [ "$SKIP_TRAIN" = false ]; then
    log "=========================================="
    log "Step 2: Training LoRA model"
    log "=========================================="

    python -m src.train_lora --config "$CONFIG"

    if [ $? -ne 0 ]; then
        error "LoRA training failed!"
        exit 1
    fi

    log "LoRA training complete!"
else
    warn "Skipping LoRA training (--skip-train)"
fi

# Step 3: Build Vector Index
if [ "$SKIP_VECTOR" = false ]; then
    log "=========================================="
    log "Step 3: Building Vector Search Index"
    log "=========================================="

    python -m src.vector_search --mode build --config "$CONFIG"

    if [ $? -ne 0 ]; then
        error "Vector index building failed!"
        exit 1
    fi

    log "Vector index built successfully!"
else
    warn "Skipping vector index (--skip-vector)"
fi

# Step 4: Evaluate
if [ "$SKIP_EVAL" = false ]; then
    log "=========================================="
    log "Step 4: Evaluating LoRA vs Vector"
    log "=========================================="

    python -m src.evaluate --config "$CONFIG"

    if [ $? -ne 0 ]; then
        error "Evaluation failed!"
        exit 1
    fi

    log "Evaluation complete!"
else
    warn "Skipping evaluation (--skip-eval)"
fi

log "=========================================="
log "Pipeline complete!"
log "=========================================="
log ""
log "Results:"
log "  - Chunks: data/chunks.json"
log "  - Training data: data/train.jsonl"
log "  - Test data: data/test.json"
log "  - LoRA model: models/lora/"
log "  - Vector index: models/embeddings/"
