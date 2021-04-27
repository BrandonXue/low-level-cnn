# Variables ===================================================================

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests
TEST_OBJ_DIR = $(TEST_DIR)/obj
TEST_BIN_DIR = $(TEST_DIR)/bin

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/479-proj-2
MAIN_OBJ = $(OBJ_DIR)/main.o

TEST_SRCS = $(wildcard $(TEST_DIR)/*.cu)
TEST_OBJS = $(patsubst $(TEST_DIR)/%.cu,$(TEST_OBJ_DIR)/%.o,$(TEST_SRCS))
TEST_TARGET = $(TEST_BIN_DIR)/catch-tests

CC = nvcc
NVCC_FLAGS = -Xcompiler -Wall --std=c++14 -Ilib -Isrc/headers

# Phony targets ===============================================================

.PHONY: all clean run test

all: $(TARGET)

test: $(TEST_TARGET)

clean: $(BIN_DIR)
ifneq ($(wildcard $(BIN_DIR)/*), )
	rm $(BIN_DIR)/*
endif
ifneq ($(wildcard $(OBJ_DIR)/*), )
	rm $(OBJ_DIR)/*
endif
ifneq ($(wildcard $(TEST_BIN_DIR)/*), )
	rm $(TEST_BIN_DIR)/*
endif
ifneq ($(wildcard $(TEST_OBJ_DIR)/*), )
	rm $(TEST_OBJ_DIR)/*
endif


run: $(TARGET)
	./$(TARGET)

run-test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Directories =================================================================

$(BIN_DIR):
	mkdir $(BIN_DIR)

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(TEST_BIN_DIR):
	mkdir $(TEST_BIN_DIR)

$(TEST_OBJ_DIR):
	mkdir $(TEST_OBJ_DIR)

# Other targets ===============================================================

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CC) $(NVCC_FLAGS) $< -c -o $@

$(TARGET): $(OBJ_DIR) $(OBJS) $(BIN_DIR)
	$(CC) $(NVCC_FLAGS) $(OBJS) -o $(TARGET)

$(TEST_OBJ_DIR)/%.o: $(TEST_DIR)/%.cu
	$(CC) $(NVCC_FLAGS) $< -c -o $@

$(TEST_TARGET): $(TEST_OBJ_DIR) $(filter-out $(MAIN_OBJ),$(OBJS)) $(TEST_OBJS) $(TEST_BIN_DIR)
	$(CC) $(NVCC_FLAGS) $(word 2,$^) $(TEST_OBJS) -o $@
