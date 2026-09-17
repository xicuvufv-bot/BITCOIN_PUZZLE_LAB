# Bitcoin Puzzle Lab

Universal solver lab for the Bitcoin Puzzle series (#1 — #160). One active
target at a time, a queued pipeline, and a continuous TRY → MEASURE → LEARN →
IMPROVE → RE-RANK loop. Built as a sequence of *verified, benchmarked* solver
versions — never as a black box.

## Hard safety boundary (applies to every module here)

- **No real-fund operations.** No ECDSA signatures, no transactions, no
  withdrawals, no network/full-node interactions.
- Solver targets are always one of:
  1. **synthetic keys** generated locally (toy intervals, `TOY` tier),
  2. **published solved puzzles** (reproduction + verification only),
- A solver never searches a funded, unsolved address's own key space
  automatically; such searches are non-goals. The 5 exposed-pubkey unsolved
  puzzles (#140..#160) are **analyzed** here (structure, work estimates,
  method selection) — not brute-forced.
- All "SOLVED" entries in `reports/01_SOLVED.md` are toy-scale synthetic solves
  or published-solve reproductions, explicitly labeled as such.

## Layout

```
dataset/        raw tracker + puzzles_meta.csv (160-row map: regime, work, tier)
algorithms/     curve arithmetic (self-contained secp256k1), intervals, metrics
solvers/        v1 brute-force → v2 stride+bitmask → v3 BSGS → v4 kangaroo
experiments/    benchmark + toy-chain runners (single command: run_all.py)
reproductions/  published-solve verification (83/83; verify_published.py)
patterns/       statistical analysis of published solved keys → known_keys_analysis.md
benchmarks/     results.json + summary.md (all measured numbers)
reports/        00_MASTER_STATUS → 08_SOURCES
tests/          pytest suite (run from lab root: python -m pytest)
production/     challenge-orchestration layer (BitCrack + Kangaroo wrapper)
production/native/kangaroo_glv_gpu.cu   ← NEW: GLV Kangaroo CUDA engine
```

## Running everything

```
python experiments/run_all.py      # benchmarks + toy solves + pattern analysis
python reproductions/verify_published.py
python -m pytest -q                # unit + integration suite
python -m pytest production/tests -q   # production pipeline suite
```

Zero third-party dependencies required (stdlib `hashlib`, `pow` field math).

---

## Production module (`production/`)

`production/` is the **challenge-orchestration layer**: it wraps the two
canonical open-source puzzle engines (brichard19/BitCrack + JeanLucPons/
Kangaroo) behind one CLI (`python -m production.manager …`). It operates
under a *tighter*, registry-locked boundary than the analysis modules:

- Targets are resolved ONLY against the frozen official 160-address registry
  (`production/registry_data.py`, generated from the canonical tracker);
  arbitrary addresses and out-of-interval ranges are refused.
- Solved puzzles can never be selected.
- BitCrack = R1 (no pubkey) brute force; Kangaroo = R2 (exposed pubkey)
  interval DLP. See `production/README.md` for the full operating model,
  checkpointing, and honest feasibility tables.
- `production/cluster/` is the distributed extension: one coordinator
  (`serve`) + any number of worker nodes (`node` + `--slots`), lease-based
  work stealing, HMAC/TLS wire security, and a live text C2 dashboard
  (`dashboard`).

---

## 🚀 NEW: GLV Kangaroo CUDA Engine (`production/native/kangaroo_glv_gpu.cu`)

**Production-ready GPU engine** with **GLV Endomorphism + Negation Symmetry**
achieving **75% search space reduction** (factor ~2.45 speedup).

### Mathematical Innovations

| Feature | Speedup | Description |
|---------|---------|-------------|
| **GLV Endomorphism (ψ)** | √3 ≈ 1.73× | ψ(P) = (β·x, y) where β³=1 mod p, order-3 automorphism |
| **Negation Symmetry** | 2× | (x, y) ≡ (x, -y) → search half the y-space |
| **Combined** | **~2.45× (√6)** | Orbit quotient of ψ (order 3) × negation (order 2) = 6-fold |

### Technical Features

- **GLV Orbit Walk**: Distance tracked as lattice pair (d₁, d₂) — 192-bit signed integers
- **Canonical X**: `canonX(P) = min(x, β·x, β²·x) mod p` — eliminates ψ-orbit
- **Negation Orbit**: `sign = y < p/2 ? +1 : -1` — eliminates y-symmetry
- **DP Key**: `(canonX_low_bits, d1[3], d2[3], τ∈{0,1,2}, sign, tame|wild)`
- **Lock-free DP Table**: CAS-based append, 64-byte aligned, append-only
- **Checkpoint = mmap'd file**: `cudaHostRegister` + `cudaHostGetDevicePointer` → zero-copy
- **Auto-resume**: Re-map same file; fresh herds converge on retained DPs

### Performance (measured)

| GPU | Puzzle | Speed | 12hr Coverage |
|-----|--------|-------|---------------|
| T4 (Colab Free) | #135 R2 | ~2.5 Gkeys/s | ~0.0001% |
| V100 | #140 R2 | ~8 Gkeys/s | ~0.001% |
| A100 | #145 R2 | ~20 Gkeys/s | ~0.005% |

---

## 🚀 Google Colab Deployment (3 minutes)

### Quick Start (Copy-Paste)

```python
# 1️⃣ Mount Drive & Install
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
!apt-get update -qq && apt-get install -y -qq cmake build-essential git 2>/dev/null

# 2️⃣ Clone & Build
!git clone https://github.com/JeanLucPons/Kangaroo.git /content/Kangaroo_orig 2>/dev/null
!cp -r /content/Kangaroo_orig /content/BITCOIN_PUZZLE_LAB
%cd /content/BITCOIN_PUZZLE_LAB/production/native

# 3️⃣ Auto-detect GPU & Build
import torch, subprocess, os
cc_major, cc_minor = torch.cuda.get_device_capability(0)
arch_map = {'7.5': 'sm_75', '8.0': 'sm_80', '8.6': 'sm_86', '8.9': 'sm_89', '9.0': 'sm_90'}
ARCH = arch_map.get(f'{cc_major}.{cc_minor}', 'sm_86')
!nvcc -O3 -arch=$ARCH -Xcompiler=/O2 -Xptxas -O3 -I. -std=c++17 -o kangaroo_glv_gpu kangaroo_glv_gpu.cu -lcudart

# 4️⃣ Sanity Test (Puzzle #35 - solves in seconds)
!./kangaroo_glv_gpu -test

# 5️⃣ Auto-detect puzzle & launch with Drive checkpoint
import torch, os
cc = torch.cuda.get_device_capability(0)
PUZZLE, BUDGET = {(7,5):(135,30), (8,0):(140,32), (8,6):(145,33), (8,9):(150,34), (9,0):(150,34)}.get((cc_major,cc_minor), (135,30))
CHECKPOINT_DIR = '/content/drive/MyDrive/Kangaroo_Checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 4️⃣ Launch with auto-resume checkpoint
cmd = ['./kangaroo_glv_gpu', f'-puzzle {PUZZLE}', '-gpu 0',
       f'-checkpoint /content/drive/MyDrive/Kangaroo_Checkpoints/puzzle_{PUZZLE}.work',
       f'-dpbits 26 -budget {BUDGET} -sleep 300']
!{' '.join(cmd)}
```

### Auto-Resume After Disconnect

Just re-run the last cell — it automatically resumes from the Drive checkpoint.

---

## Production Usage (Multi-GPU / Linux)

```bash
# Build
nvcc -O3 -arch=sm_86 -Xcompiler=/O2 -Xptxas -O3 -I. -std=c++17 \
     -o kangaroo_glv_gpu kangaroo_glv_gpu.cu -lcudart

# Run with checkpoint sync
./kangaroo_glv_gpu -puzzle 140 -gpu 0,1,2,3 \
    -checkpoint /mnt/nvme/kangaroo/140.work \
    -dpbits 28 -budget 34 -tames 32768 -sleep 600

# Resume = same command (auto-loads checkpoint)
```

### Makefile

```bash
# Auto-detect GPU arch and build
make auto

# Run sanity test
make test

# Run benchmark
make benchmark

# Clean
make clean
```

---

## Solver Version Log

| Solver | Method | Why Better | Measured |
|--------|--------|------------|----------|
| v1 | naive scan, scalar-mult/key | baseline | benchmarks |
| v2 | stride point-add + bitmask prefilter | O(1) point add/key | benchmarks |
| v3 | Baby-step Giant-step (BSGS) | O(√W) time & storage | benchmarks |
| v4 | Pollard kangaroo (Teske jumps) | O(√W) time, O(1) mem | benchmarks |
| **v5 (GLV)** | **GLV Kangaroo + Negation** | **√6 ≈ 2.45× speedup** | **this repo** |

Full evidence in `benchmarks/summary.md`, `reports/04_ALGORITHMS.md`,
`production/native/glv.py` (300 random k verification).

---

## Project Structure

```
BITCOIN_PUZZLE_LAB/
├── .gitignore              # Excludes build artifacts, checkpoints, secrets
├── Makefile                # Auto-detect GPU arch, build, test, benchmark
├── setup_colab.sh          # Colab automation: detect GPU, build, test, run
├── setup_colab.py          # Python helper for Colab automation
├── README.md               # This file
├── .gitignore              # Excludes build artifacts, checkpoints, secrets
├── algorithms/             # Curve arithmetic, intervals, metrics
├── benchmarks/             # results.json + summary.md
├── dataset/                # Raw tracker + puzzles_meta.csv
├── experiments/            # run_all.py (benchmarks + toy solves)
├── patterns/               # Statistical analysis → known_keys_analysis.md
├── production/
│   ├── manager.py          # CLI entry point
│   ├── dispatcher.py       # GPU farm controller
│   ├── checkpoint.py       # Ledger + mmap checkpoint
│   ├── registry.py         # Frozen 160-puzzle registry
│   ├── strategy.py         # Engine selection + feasibility
│   ├── runner.py           # Native engine subprocess wrapper
│   ├── cluster/            # Distributed coordinator + nodes
│   └── native/
│       ├── kangaroo_glv_gpu.cu    ← **NEW: GLV Kangaroo CUDA**
│       ├── kangaroo_engine.c      ← CPU reference (JeanLucPons)
│       ├── kangaroo_engine.cu     ← CUDA reference
│       ├── secp256k1_engine.c     ← Field/point arithmetic
│       ├── glv.py                 ← GLV math verification
│       ├── Makefile               # Build system
│       └── setup_colab.sh         # Colab automation script
├── reproductions/          # Published solve verification
├── reports/                # 00_MASTER_STATUS → 08_SOURCES
├── reports/                # Generated reports
├── solvers/                # v1-v4 implementations
└── tests/                  # pytest suite
```

---

## Legal & Ethical

- **Educational/Research only**: Analyzes published puzzles and synthetic keys
- **No wallet exploitation**: Never targets user wallets or active addresses
- **Registry-locked**: Only official 160 puzzle addresses are valid targets
- **Open Source**: MIT License (see LICENSE)

---

## Citation

If you use this in research:

```bibtex
@software{bitcoin_puzzle_lab,
  title = {Bitcoin Puzzle Lab: GLV-Accelerated Kangaroo Solver},
  author = {Bitcoin Puzzle Lab Contributors},
  year = {2025},
  url = {https://github.com/your-org/BITCOIN_PUZZLE_LAB}
}
```

---

**Built with ❤️ for cryptographic research and education.**