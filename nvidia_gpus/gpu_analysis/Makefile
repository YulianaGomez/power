INCLUDES := -I/common/inc

all: build
build: matrixMul

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

matrixMul.o:matrixMul.cu
	$(EXEC) $(NVCC) $(INCLUDES) -o $@ 

run: build
	$(EXEC) ./matrixMul
	
