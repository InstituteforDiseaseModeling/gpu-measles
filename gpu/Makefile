# GPU implementation

measles: measles-main.cu measles-kernel.cu
	nvcc -gencode arch=compute_60,code=sm_60 -lineinfo -rdc=true -std=c++11 --maxrregcount=63 --ptxas-options=-v -o bin/$@ $^
