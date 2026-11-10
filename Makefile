# Builds air2glsl against the system LLVM.
#
# LLVM 18 is what this was developed against (Ubuntu 24.04: `apt install
# llvm-18-dev`). `llvm-config-18` must be on PATH; override with
# `make LLVM_CONFIG=/usr/lib/llvm-20/bin/llvm-config`.

LLVM_CONFIG ?= llvm-config-18
CXX ?= clang++-18
CXXFLAGS += -std=c++17 -O2 -Wall -Wextra $(shell $(LLVM_CONFIG) --cxxflags) -fexceptions
LDFLAGS += $(shell $(LLVM_CONFIG) --ldflags --libs core bitreader support \
	--system-libs)

SOURCES := src/main.cpp src/metallib.cpp src/emit.cpp
HEADERS := src/metallib.h src/emit.h

air2glsl: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

.PHONY: clean check
check: air2glsl
	$(MAKE) -C . >/dev/null
	./air2glsl stats $(SHADER_DATA)

clean:
	rm -f air2glsl
