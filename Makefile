INCLUDE_PATH := $(abspath ./)
LIBRARY_PATH := $(abspath ./)

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

#
# Compile flags
#

BUILD_TYPE?=
# keep standard at C11 and C++11
CFLAGS   = -I./ggllm.cpp -I. -O3 -DNDEBUG -std=c11 -fPIC
CXXFLAGS = -I./ggllm.cpp -I. -I./ggllm.cpp/examples -I./examples -O3 -DNDEBUG -std=c++11 -fPIC
LDFLAGS  =

# warnings
CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function

# OS specific
# TODO: support Windows
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Darwin)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),FreeBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),NetBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),OpenBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Haiku)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	# Use all CPU extensions that are available:
	CFLAGS += -march=native -mtune=native
endif
ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS += -mcpu=power9
		CXXFLAGS += -mcpu=power9
	endif
	# Require c++23's std::byteswap for big-endian support.
	ifeq ($(UNAME_M),ppc64)
		CXXFLAGS += -std=c++23 -DGGML_BIG_ENDIAN
	endif
endif
ifndef LLAMA_NO_ACCELERATE
	# Mac M1 - include Accelerate framework.
	# `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE
		LDFLAGS += -framework Accelerate
	endif
endif
ifdef LLAMA_OPENBLAS
	CFLAGS  += -DGGML_USE_OPENBLAS -I/usr/local/include/openblas
	LDFLAGS += -lopenblas
endif
ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifneq ($(filter aarch64%,$(UNAME_M)),)
	CFLAGS += -mcpu=native
	CXXFLAGS += -mcpu=native
endif
ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, 2, 3
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif
ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

ifeq ($(BUILD_TYPE),openblas)
	EXTRA_LIBS=
	CMAKE_ARGS+=-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DBLAS_INCLUDE_DIRS=/usr/include/openblas
endif

ifeq ($(BUILD_TYPE),blis)
	EXTRA_LIBS=
	CMAKE_ARGS+=-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=FLAME
endif

ifeq ($(BUILD_TYPE),cublas)
	EXTRA_LIBS=
	CMAKE_ARGS+=-DLLAMA_CUBLAS=ON
	EXTRA_TARGETS+=ggllm.cpp/ggml-cuda.o
endif

ifeq ($(BUILD_TYPE),clblas)
	EXTRA_LIBS=
	CMAKE_ARGS+=-DLLAMA_CLBLAST=ON
	EXTRA_TARGETS+=ggllm.cpp/ggml-opencl.o
endif

ifeq ($(BUILD_TYPE),metal)
	EXTRA_LIBS=
	CGO_LDFLAGS+="-framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
	CMAKE_ARGS+=-DLLAMA_METAL=ON
	EXTRA_TARGETS+=ggllm.cpp/ggml-metal.o
endif

ifdef CLBLAST_DIR
	CMAKE_ARGS+=-DCLBlast_dir=$(CLBLAST_DIR)
endif

#
# Print build information
#

$(info I ggllm.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I CGO_LDFLAGS:  $(CGO_LDFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I BUILD_TYPE:  $(BUILD_TYPE))
$(info I CMAKE_ARGS:  $(CMAKE_ARGS))
$(info I EXTRA_TARGETS:  $(EXTRA_TARGETS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

# Use this if you want to set the default behavior


ggllm.cpp/ggml.o: prepare
	mkdir -p build
	cd build && cmake ../ggllm.cpp $(CMAKE_ARGS) && VERBOSE=1 cmake --build . --config Release && cp -rf CMakeFiles/ggml.dir/ggml.c.o ../ggllm.cpp/ggml.o

ggllm.cpp/ggml-cuda.o: ggllm.cpp/ggml.o
	cd build && cp -rf CMakeFiles/ggml.dir/ggml-cuda.cu.o ../ggllm.cpp/ggml-cuda.o

ggllm.cpp/ggml-opencl.o: ggllm.cpp/ggml.o
	cd build && cp -rf CMakeFiles/ggml.dir/ggml-opencl.cpp.o ../ggllm.cpp/ggml-opencl.o

ggllm.cpp/ggml-metal.o: ggllm.cpp/ggml.o
	cd build && cp -rf CMakeFiles/ggml.dir/ggml-metal.m.o ../ggllm.cpp/ggml-metal.o

ggllm.cpp/k_quants.o: ggllm.cpp/ggml.o
	cd build && cp -rf CMakeFiles/ggml.dir/k_quants.c.o ../ggllm.cpp/k_quants.o

ggllm.cpp/llama.o:
	cd build && cp -rf CMakeFiles/llama.dir/llama.cpp.o ../ggllm.cpp/llama.o

ggllm.cpp/libfalcon.o:
	cd build && cp -rf CMakeFiles/falcon.dir/libfalcon.cpp.o ../ggllm.cpp/libfalcon.o

ggllm.cpp/falcon_common.o:
	cd build && cp -rf examples/CMakeFiles/falcon_common.dir/falcon_common.cpp.o ../ggllm.cpp/falcon_common.o

ggllm.cpp/cmpnct_unicode.o:
	cd build && cp -rf CMakeFiles/cmpnct_unicode.dir/cmpnct_unicode.cpp.o ../ggllm.cpp/cmpnct_unicode.o

falcon_binding.o: prepare ggllm.cpp/ggml.o ggllm.cpp/cmpnct_unicode.o ggllm.cpp/llama.o ggllm.cpp/libfalcon.o ggllm.cpp/falcon_common.o
	$(CXX) $(CXXFLAGS) -I./ggllm.cpp -I./ggllm.cpp/examples falcon_binding.cpp -o falcon_binding.o -c $(LDFLAGS)

## https://github.com/ggerganov/llama.cpp/pull/1902
prepare:
	cd ggllm.cpp && patch -p1 < ../patches/1902-cuda.patch
	touch $@

libggllm.a: prepare falcon_binding.o ggllm.cpp/k_quants.o $(EXTRA_TARGETS)
	ar src libggllm.a ggllm.cpp/libfalcon.o ggllm.cpp/cmpnct_unicode.o ggllm.cpp/ggml.o ggllm.cpp/k_quants.o $(EXTRA_TARGETS) ggllm.cpp/falcon_common.o falcon_binding.o

clean:
	rm -rf *.o
	rm -rf *.a
	$(MAKE) -C ggllm.cpp clean
	rm -rf build

test: libggllm.a
	@C_INCLUDE_PATH=${INCLUDE_PATH} CGO_LDFLAGS=${CGO_LDFLAGS} LIBRARY_PATH=${LIBRARY_PATH} go test -v ./...
