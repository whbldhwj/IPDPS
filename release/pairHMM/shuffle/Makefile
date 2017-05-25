COMMONFLAG = -arch=sm_35 -line-info -Xptxas -v
#COMMONFLAG = -arch=sm_50 -g -G
#CUDA_DIR = /usr/local/cuda
#CUDA_LIB_DIR := $(CUDA_DIR)/lib
CXXFLAGS = --use_fast_math

NVCC := nvcc
GCC := gcc
#INCLUDE := $(CUDA_DIR)/include
SRC = pairHMM.cu
EXE = pairHMM

#all:cpu.o gpu.o host.o const.o
all:cpu.o host.o const.o trans.o
	$(NVCC) -dlink host.o -o device.o
	$(GCC) -o $(EXE) device.o cpu.o host.o trans.o const.o -lm -lgomp -lstdc++ -L /usr/local/cuda/lib64 -lcudart

cpu.o:
	$(GCC) -g -o cpu.o -c pairHMM.cpp

host.o:
	$(NVCC) -O4 $(COMMONFLAG) $(CXXFLAGS) -o host.o -c pairHMM_test.cu

const.o:
	$(GCC) -O4 -ffast-math -o const.o -c pairHMM_const.cpp

trans.o:
	$(GCC) -O4 -fopenmp -o trans.o -c pairHMM_trans_cal.cpp

clean:
	rm *.o pairHMM
