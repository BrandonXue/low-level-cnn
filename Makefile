# IMPORTANT:
# - Test code must include catch.hpp
# - Test files must be named test_<filename>.cu where <filename> is the header
#   file of the code being tested, without the ".cu.h" part.
# - Test files must include the header file of the code they are testing.
# - Project must follow the structure below.
# - Given this structure, things will generate properly.
#
# <project root>
# |--<SRC_DIR>
# |  |--headers
# |  |  |-- <header1>.cu.h
# |  |  |-- <header2>.cu.h
# |  |  `-- (internal headers)...
# |  |
# |  |-- <MAIN_NAME>.cu
# |  |-- <src_file1>.cu
# |  |-- <src_file2>.cu
# |  `-- (internal sources)...
# |
# |--<TEST_DIR>
# |  |--obj
# |  |  `-- (test .o outputs)...
# |  |--bin
# |  |  `-- (test executable output)...
# |  |
# |  |-- <TEST_MAIN_NAME>.cu
# |  |-- <test_header1>.cu
# |  |-- <test_header2>.cu
# |  `-- (other tests)...
# |
# |--external
# |  `-- (external dependency headers)...
# |
# |--<OBJ_DIR>
# |  `-- (main .o outputs)...
# |
# `--<BIN_DIR>
#    `-- (main executable output)...

# Variables ===================================================================

# Source file names without extension of where main functions are.
MAIN_NAME=main
TEST_MAIN_NAME=catch_main

# Output executable file names
TARGET_NAME=479_proj_2
TEST_TARGET_NAME=catch_tests

# Directory names (must exist in the project root)
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests

# Test directory names
TEST_OBJ_DIR = $(TEST_DIR)/obj
TEST_BIN_DIR = $(TEST_DIR)/bin

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/$(TARGET_NAME)
MAIN_OBJ = $(OBJ_DIR)/$(MAIN_NAME).o

TEST_SRCS = $(wildcard $(TEST_DIR)/*.cu)
TEST_OBJS = $(patsubst $(TEST_DIR)/%.cu,$(TEST_OBJ_DIR)/%.o,$(TEST_SRCS))
TEST_TARGET = $(TEST_BIN_DIR)/$(TEST_TARGET_NAME)
TEST_MAIN_OBJ=$(TEST_OBJ_DIR)/$(TEST_MAIN_NAME).o

CC = nvcc
NVCC_FLAGS = -Xcompiler -Wall -O3 --std=c++14 -Iexternal -Isrc/headers

# Phony targets ===============================================================

.PHONY: all clean run run-test run-tests test tests

all: $(TARGET)

test: $(TEST_TARGET)

tests: $(TEST_TARGET)

clean: $(BIN_DIR)
ifneq ($(wildcard $(BIN_DIR)/*), )
	@rm $(BIN_DIR)/*
endif
ifneq ($(wildcard $(OBJ_DIR)/*), )
	@rm $(OBJ_DIR)/*
endif
ifneq ($(wildcard $(TEST_BIN_DIR)/*), )
	@rm $(TEST_BIN_DIR)/*
endif
# For catch.hpp usage specifically, do not rm the object file with catch's
# main function. It takes a while to compile
ifneq ($(wildcard $(TEST_OBJ_DIR)/*), )
	@find ./$(TEST_OBJ_DIR) ! -name "$(TEST_MAIN_NAME).o" -type f -exec rm {} +
endif


run: $(TARGET)
	@./$(TARGET)

run-test: $(TEST_TARGET)
	@./$(TEST_TARGET)

run-tests: $(TEST_TARGET)
	@./$(TEST_TARGET)

# Directories =================================================================

$(BIN_DIR):
	@mkdir $(BIN_DIR)

$(OBJ_DIR):
	@mkdir $(OBJ_DIR)

$(TEST_BIN_DIR):
	@mkdir $(TEST_BIN_DIR)

$(TEST_OBJ_DIR):
	@mkdir $(TEST_OBJ_DIR)

# Other targets ===============================================================

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@$(CC) $(NVCC_FLAGS) $< -c -o $@

$(TARGET): $(OBJ_DIR) $(OBJS) $(BIN_DIR)
	$(CC) $(NVCC_FLAGS) $(OBJS) -o $(TARGET)

$(TEST_OBJ_DIR)/%.o: $(TEST_DIR)/%.cu
	@$(CC) $(NVCC_FLAGS) $< -c -o $@

# For catch.hpp usage specifically, exclude the object file with the main function.
# catch.hpp will define its own main function.
$(TEST_TARGET): $(TEST_OBJ_DIR) $(filter-out $(MAIN_OBJ),$(OBJS)) $(TEST_OBJS) $(TEST_BIN_DIR)
	$(CC) $(NVCC_FLAGS) $(filter-out $(MAIN_OBJ),$(OBJS)) $(TEST_OBJS) -o $@
