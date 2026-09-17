#!/bin/bash
# setup_colab.sh - Complete Google Colab Automation for GLV Kangaroo
# Usage: chmod +x setup_colab.sh && ./setup_colab.sh
#        (or run directly in Colab: bash setup_colab.sh)

set -euo pipefail

# ─── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# ─── 1. Detect GPU & Compute Capability ────────────────────────────────────
log "🔍 Detecting GPU..."
GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | head -1)
if [ -z "$GPU_INFO" ]; then
    error "No NVIDIA GPU detected! Enable GPU in Colab: Runtime → Change runtime type → GPU"
fi

GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
GPU_CC=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
GPU_MEM=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)

log "GPU: $GPU_NAME"
log "Compute Capability: $GPU_CC"
log "Memory: $GPU_MEM"

# Map compute capability to ARCH
case "$GPU_CC" in
    "7.5") ARCH="sm_75" ;;   # T4
    "8.0") ARCH="sm_80" ;;   # A100 40GB
    "8.6") ARCH="sm_86" ;;   # A100 80GB
    "8.9") ARCH="sm_89" ;;   # H100
    "9.0") ARCH="sm_90" ;;   # H100
    *) ARCH="sm_86"; warn "Unknown CC $GPU_CC, defaulting to sm_86" ;;
esac

log "Using ARCH: $ARCH"

# ─── 2. Mount Google Drive ────────────────────────────────────────────────
log "🔗 Mounting Google Drive..."
python3 -c "
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
" 2>/dev/null || warn "Drive mount failed (run manually if needed: from google.colab import drive; drive.mount('/content/drive'))"

# Create checkpoint directory
CHECKPOINT_DIR="/content/drive/MyDrive/Kangaroo_Checkpoints"
mkdir -p "$CHECKPOINT_DIR"
success "Checkpoint directory: $CHECKPOINT_DIR"

# ─── 3. Install Dependencies ──────────────────────────────────────────────
log "📦 Installing dependencies..."
apt-get update -qq && apt-get install -y -qq \
    cmake build-essential git wget python3-pip 2>/dev/null
success "Dependencies installed"

# ─── 4. Clone Repository ──────────────────────────────────────────────────
REPO_DIR="/content/BITCOIN_PUZZLE_LAB"
if [ ! -d "$REPO_DIR" ]; then
    log "📥 Cloning repository..."
    git clone https://github.com/JeanLucPons/Kangaroo.git "$REPO_DIR"_orig 2>/dev/null || \
    git clone https://github.com/your-org/BITCOIN_PUZZLE_LAB.git "$REPO_DIR" 2>/dev/null
    cp -r "$REPO_DIR"_orig "$REPO_DIR" 2>/dev/null || true
    success "Repository cloned"
else
    log "Repository exists, pulling latest..."
    cd "$REPO_DIR" && git pull 2>/dev/null || true
fi

cd "$REPO_DIR/production/native"
success "Working in $(pwd)"

# ─── 5. Build ─────────────────────────────────────────────────────────────
log "🔨 Building with ARCH=$ARCH..."
nvcc -O3 -arch=$ARCH -Xcompiler=/O2 -Xptxas -O3 \
    -I. -std=c++17 -o kangaroo_glv_gpu kangaroo_glv_gpu.cu -lcudart 2>&1 | tail -10

if [ -f "kangaroo_glv_gpu" ]; then
    success "Build successful! Binary: ./kangaroo_glv_gpu"
else
    error "Build failed! Check errors above."
fi

# ─── 6. Run Sanity Test ───────────────────────────────────────────────────
log "🧪 Running Sanity Test (-test flag)..."
./kangaroo_glv_gpu -test

if [ $? -eq 0 ]; then
    success "Sanity test PASSED!"
else
    error "Sanity test FAILED! Check output above."
fi

# ─── 6. Auto-detect best puzzle for this GPU ──────────────────────────────
log "🎯 Selecting optimal puzzle for $GPU_NAME..."
case "$GPU_CC" in
    "7.5") PUZZLE=135; BUDGET=30 ;;   # T4: Puzzle #135 (R2, 66-bit)
    "8.0") PUZZLE=140; BUDGET=32 ;;   # A100: Puzzle #140 (R2, 66-bit)
    "8.6") PUZZLE=145; BUDGET=33 ;;   # A100 80GB
    "8.9") PUZZLE=150; BUDGET=34 ;;   # H100
    "9.0") PUZZLE=150; BUDGET=34 ;;   # H100
    *) PUZZLE=135; BUDGET=30 ;;
esac

log "Selected: Puzzle #$PUZZLE (Budget: 2^$BUDGET ops)"

# ─── 7. Setup Checkpoint Path ─────────────────────────────────────────────
CHECKPOINT_DIR="/content/drive/MyDrive/Kangaroo_Checkpoints"
mkdir -p "$CHECKPOINT_DIR"
CHECKPOINT_FILE="$CHECKPOINT_DIR/puzzle_${PUZZLE}.work"
success "Checkpoint: $CHECKPOINT_FILE"

# ─── 7. Launch! ───────────────────────────────────────────────────────────
log "🚀 Launching Kangaroo GLV on Puzzle #$PUZZLE..."
log "Checkpoint: $CHECKPOINT_FILE (auto-resume enabled)"
log "GPU: $GPU_NAME ($ARCH) | Budget: 2^$BUDGET ops | DP bits: 26"

# Launch with checkpoint sync (auto-resume on reconnect)
exec ./kangaroo_glv_gpu \
    -puzzle $PUZZLE \
    -gpu 0 \
    -checkpoint "$CHECKPOINT_DIR/puzzle_${PUZZLE}.work" \
    -dpbits 26 \
    -budget $BUDGET \
    -sleep 300 \
    2>&1 | tee -a "/content/kangaroo_${PUZZLE}.log"
EOF
echo "setup_colab.sh created"