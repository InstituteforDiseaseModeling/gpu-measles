# GPU implementation

measles: measles-main.cpp measles-main.h measles-kernel.cu measles-kernel.h block-size.h check-return.h
	nvcc -gencode arch=compute_60,code=sm_60 -lineinfo -rdc=true -std=c++11 --maxrregcount=63 --ptxas-options=-v -o bin/$@ $^
