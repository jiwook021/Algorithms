###############################################################################
# Makefile for SimpleCppCompiler
###############################################################################

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -pedantic
LDFLAGS = -pthread

# Debug and release configurations
DEBUG_FLAGS = -g -O0 -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG

# Google Test configuration
GTEST_DIR = /usr/local/include/gtest
GTEST_LIBS = -lgtest -lgtest_main

# Directories
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = test
BIN_DIR = bin
DOC_DIR = doc

# Files
COMPILER_SRC = SimpleCppCompiler.cpp
TEST_SRC = CompilerTest.cpp
COMPILER_EXE = $(BIN_DIR)/compiler
TEST_EXE = $(BIN_DIR)/compiler_test

# Default target
all: release

# Create directories
$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(BIN_DIR) $(COMPILER_EXE)

# Release build
release: CXXFLAGS += $(RELEASE_FLAGS)
release: $(BIN_DIR) $(COMPILER_EXE)

# Build the compiler
$(COMPILER_EXE): $(COMPILER_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Build and run tests
test: $(TEST_EXE)
	$(TEST_EXE)

# Build the test executable
$(TEST_EXE): $(TEST_SRC) $(COMPILER_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $< -o $@ $(GTEST_LIBS) $(LDFLAGS)

# Run the compiler on the test C++ file
compile_test: $(COMPILER_EXE)
	$(COMPILER_EXE) test.cpp test.asm

# Run the compiler with the verbose flag
compile_verbose: $(COMPILER_EXE)
	$(COMPILER_EXE) test.cpp test.asm --verbose

# Run the compiler with different optimization levels
compile_o0: $(COMPILER_EXE)
	$(COMPILER_EXE) test.cpp test_o0.asm -O0

compile_o1: $(COMPILER_EXE)
	$(COMPILER_EXE) test.cpp test_o1.asm -O1

compile_o2: $(COMPILER_EXE)
	$(COMPILER_EXE) test.cpp test_o2.asm -O2

compile_o3: $(COMPILER_EXE)
	$(COMPILER_EXE) test.cpp test_o3.asm -O3

# Assemble and link the generated assembly code (requires nasm)
run_test: compile_test
	nasm -f elf64 test.asm -o test.o
	$(CXX) test.o -o test
	./test

# Generate documentation (requires Doxygen)
docs:
	doxygen Doxyfile

# Install the compiler
install: $(COMPILER_EXE)
	install -d $(DESTDIR)/usr/local/bin
	install -m 755 $(COMPILER_EXE) $(DESTDIR)/usr/local/bin/cppcompiler

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) *.o *.asm test

# Clean everything including documentation
distclean: clean
	rm -rf $(DOC_DIR)

# Phony targets
.PHONY: all debug release test compile_test compile_verbose compile_o0 compile_o1 compile_o2 compile_o3 run_test docs install clean distclean

###############################################################################
# Dependencies
###############################################################################

# Generate dependencies automatically
DEPFLAGS = -MT $@ -MMD -MP -MF $(BUILD_DIR)/$*.d

# Include generated dependencies
-include $(wildcard $(BUILD_DIR)/*.d)

###############################################################################
# Help
###############################################################################

help:
	@echo "SimpleCppCompiler Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build the compiler (release mode)"
	@echo "  debug       - Build the compiler with debug symbols"
	@echo "  release     - Build the compiler with optimizations"
	@echo "  test        - Build and run tests"
	@echo "  compile_test - Compile the test.cpp file"
	@echo "  compile_verbose - Compile the test.cpp file with verbose output"
	@echo "  compile_o0  - Compile with -O0 optimization"
	@echo "  compile_o1  - Compile with -O1 optimization"
	@echo "  compile_o2  - Compile with -O2 optimization"
	@echo "  compile_o3  - Compile with -O3 optimization"
	@echo "  run_test    - Compile, assemble, link, and run test.cpp"
	@echo "  docs        - Generate documentation"
	@echo "  install     - Install the compiler"
	@echo "  clean       - Remove build artifacts"
	@echo "  distclean   - Remove all generated files"
	@echo "  help        - Display this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CXX         - C++ compiler (default: g++)"
	@echo "  CXXFLAGS    - C++ compiler flags"
	@echo "  DESTDIR     - Installation destination root"

.PHONY: help