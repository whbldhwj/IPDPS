CXXFLAGS = -ffast-math
#NVCCFLAGS = -arch=sm_50 -Xptxas -v,-abi=no,-dlcm=ca --use_fast_math
NVCCFLAGS = -arch=sm_35 -Xptxas -v,-abi=no --use_fast_math
#NVCCFLAGS = -arch=sm_50 -Xptxas -v,-abi=no --use_fast_math -lineinfo
EXE = sw

all: main.o sw_kernel.o sw_wrap.o
	nvcc -dlink sw_wrap.o sw_kernel.o -o device.o
	gcc -o $(EXE) device.o main.o sw_kernel.o sw_wrap.o -lcudart -lm -lstdc++
main.o:
	gcc -o main.o -c main.cpp
sw_kernel.o:
	nvcc -O4 $(NVCCFLAGS) -o sw_kernel.o -c sw_kernel.cu
sw_wrap.o:
	nvcc -O4 $(NVCCFLAGS) -o sw_wrap.o -c sw_wrap.cu
clean:
	rm *.o sw
