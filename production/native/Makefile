# Kangaroo GLV GPU - Makefile
# Auto-detect GPU architecture, build, test, benchmark
# Usage: make [ARCH=sm_75] [WITHGPU=1] [DEBUG=1] [test|benchmark|clean|auto]

# ─── Auto-detect GPU architecture ──────────────────────────────────────────
ifeq ($(ARCH),)
    # Default to compute capability 75 (Turing/T4)
    # Override with: make ARCH=sm_75 (T4) or make ARCH=sm_80 (A100) or make ARCH=sm_86 (Ampere)
    ARCH = sm_75
endif

# ─── CUDA Path Configuration ───────────────────────────────────────────────
# Use default CUDA path in Colab / standard Linux
CUDA_PATH ?= /usr/local/cuda
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64

# ─── Compiler & Flags ──────────────────────────────────────────────────────
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Base flags
NVCCFLAGS = -O3 -use_fast_math -Xptxas -O3 -Xcompiler -O3
NVCCFLAGS += -arch=$(ARCH) -std=c++17
NVCCFLAGS += -Xptxas -v -maxrregcount=32
NVCCFLAGS += -D_FORCE_INLINES -D__CUDA_NO_HALF2_OPERATORS__
NVCCFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1

# GPU Enable flag
ifeq ($(WITHGPU),1)
    NVCCFLAGS += -DWITHGPU=1
endif

# Debug flags
ifeq ($(DEBUG),1)
    NVCCFLAGS += -G -g -lineinfo
    CXXFLAGS += -g -O0
else
    NVCCFLAGS += -Xptxas -v -maxrregcount=32
    NVCCFLAGS += -D_FORCE_INLINES -D__CUDA_NO_HALF2_OPERATORS__
    NVCCFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
    CXXFLAGS += -O3 -DNDEBUG
endif

CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -lcudart -lstdc++

INCLUDES = -I. -I../include $(CUDA_INC)

# ─── Targets ───────────────────────────────────────────────────────────────
SOURCES = kangaroo_glv_gpu.cu
TARGET = kangaroo_glv_gpu
TARGET_WIN = kangaroo_glv_gpu.exe

# ─── Default Target ────────────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): kangaroo_glv_gpu.cu
	@echo "Building $(TARGET) for ARCH=$(ARCH) WITHGPU=$(WITHGPU)..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(TARGET) kangaroo_glv_gpu.cu $(CUDA_LIB) -lcudart
	@echo "Build complete: ./$(TARGET)"

$(TARGET_WIN): kangaroo_glv_gpu.cu
	@echo "Building $(TARGET_WIN) for ARCH=$(ARCH) (Windows/MSVC)..."
	$(NVCC) $(NVCCFLAGS) -Xcompiler="/MD" -o $(TARGET_WIN) kangaroo_glv_gpu.cu $(CUDA_LIB) -lcudart

# ─── Auto-detect GPU Architecture ─────────────────────────────────────────
detect-arch:
	@echo "Detecting GPU architecture..."
	@nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | sed 's/\.//g' | sed 's/^/ARCH=sm_/' || echo "ARCH=sm_75 (default T4)"

auto: detect-arch
	@$(MAKE) ARCH=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | sed 's/\.//g' | sed 's/^/sm_/') WITHGPU=1

# ─── GPU Test ──────────────────────────────────────────────────────────────
gpu-test:
	@echo "Testing GPU availability..."
	@nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
	@nvcc --version

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

# ─── Full Clean & Rebuild ──────────────────────────────────────────────────
rebuild: clean
	@$(MAKE) WITHGPU=1 ARCH=sm_75

.PHONY: all clean test benchmark detect-arch auto install gpu-test rebuild