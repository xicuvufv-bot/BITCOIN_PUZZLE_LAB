# Kangaroo GLV GPU - Makefile
# Auto-detect GPU architecture, build, test, benchmark
# Usage: make [ARCH=sm_86] [DEBUG=1] [test|benchmark|clean|auto]

# ─── Auto-detect GPU architecture ──────────────────────────────────────────
ifeq ($(ARCH),)
    # Default to compute capability 86 (Ampere/A100)
    # Override with: make ARCH=sm_75 (T4) or make ARCH=sm_80 (A100)
    ARCH = sm_86
endif

# ─── Compiler & Flags ──────────────────────────────────────────────────────
NVCC = nvcc
CXX = g++

NVCCFLAGS = -O3 -use_fast_math -Xptxas -O3 -Xcompiler -O3
NVCCFLAGS += -arch=$(ARCH) -std=c++17
NVCCFLAGS += -Xptxas -v -maxrregcount=32
NVCCFLAGS += -D_FORCE_INLINES -D__CUDA_NO_HALF2_OPERATORS__
NVCCFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1

CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -lcudart -lstdc++

ifeq ($(DEBUG),1)
    NVCCFLAGS += -G -g -lineinfo
    CXXFLAGS += -g -O0
else
    CXXFLAGS += -O3 -DNDEBUG
endif

INCLUDES = -I. -I../include

# ─── Targets ───────────────────────────────────────────────────────────────
SOURCES = kangaroo_glv_gpu.cu
TARGET = kangaroo_glv_gpu
TARGET_WIN = kangaroo_glv_gpu.exe

# ─── Default Target ────────────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): kangaroo_glv_gpu.cu
	@echo "Building $(TARGET) for ARCH=$(ARCH)..."
	$(NVCC) $(NVCCFLAGS) -I. -o $(TARGET) kangaroo_glv_gpu.cu -lcudart
	@echo "Build complete: ./$(TARGET)"

$(TARGET_WIN): kangaroo_glv_gpu.cu
	@echo "Building $(TARGET_WIN) for ARCH=$(ARCH) (Windows/MSVC)..."
	$(NVCC) $(NVCCFLAGS) -Xcompiler="/MD" -o $(TARGET_WIN) kangaroo_glv_gpu.cu -lcudart

# ─── Auto-detect GPU Architecture ─────────────────────────────────────────
detect-arch:
	@echo "Detecting GPU architecture..."
	@nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | sed 's/\.//g' | sed 's/^/ARCH=sm_/' || echo "ARCH=sm_86 (default)"

auto: detect-arch
	@$(MAKE) ARCH=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | sed 's/\.//g' | sed 's/^/sm_/')

# ─── Test & Benchmark ──────────────────────────────────────────────────────
test: $(TARGET)
	@echo "🧪 Running Sanity Test..."
	./$(TARGET) -test

benchmark: $(TARGET)
	@echo "⚡ Running Benchmark..."
	./$(TARGET) -benchmark

# ─── Install (copy to bin/) ────────────────────────────────────────────────
install: $(TARGET)
	@mkdir -p ../bin
	cp $(TARGET) ../bin/
	@echo "Installed to ../bin/$(TARGET)"

# ─── Clean ─────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET) $(TARGET_WIN) *.o *.obj *.ptx *.cubin *.fatbin *.nvFatbin *.log
	rm -rf build/ build_*/ CMakeFiles/ CMakeCache.txt cmake_install.cmake
	rm -f cmake_install.cmake cmake_uninstall.cmake install_manifest.txt compile_commands.json

.PHONY: all clean test benchmark detect-arch auto install
EOF
echo "Makefile created"